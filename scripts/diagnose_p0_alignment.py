"""Training-only P0 budget and predictor-gradient audit. No optimizer/test access."""
import os,sys,json,math,gc,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import setproctitle
setproctitle.setproctitle('yeonwoo');sys.dont_write_bytecode=True
import torch,numpy as np
import train_v3_study as w
import train_hybrid_jeju_forecast as h
from mhw import subset_preprocessing as p
from mhw.ocean_stencil import ocean_neighbor_support
from torch.utils.data import DataLoader,Subset
OLD=ROOT/'outputs/data_efficiency/v3_baseline_p0_20261008_r1'
OUT=ROOT/'outputs/p0_repair_20261009'

def integral(pred,tm,target,mask,batch,args,lead,we,rres):
 rhs,_=h._linear_rhs_per_day(pred,tm,mask,batch,args,lead,we)
 mean=rhs.reshape(lead,*pred.shape).mean(0)
 return h.compute_physics_loss((pred-tm)/lead-mean,ocean_neighbor_support(mask),args,rhs_per_day=mean)
def budget(base,starts,lead,device):
 support=ocean_neighbor_support(torch.tensor(base.ocean_mask,device=device)[None,None]);sea=support[0,0].bool();mask=torch.tensor(base.ocean_mask,device=device)[None,None]
 dx=torch.tensor(base.dx,device=device)[None,None];dy=torch.tensor(base.dy,device=device)[None,None]
 sums={};count=0
 for off in range(0,len(starts),64):
  now=np.asarray(starts[off:off+64])+29;idx=now[:,None]+np.arange(lead+1)[None,:]
  clean=lambda a:torch.tensor(np.nan_to_num(a,nan=0,posinf=0,neginf=0),device=device)
  observed=clean(base.target_array[idx]);linear=observed[:,0:1]+torch.arange(lead+1,device=device)[None,:,None,None]/lead*(observed[:,-1:]-observed[:,0:1])
  forcing={name:clean(base.physics_arrays[name][idx[:,:-1]]) for name in ['qnet','mlotst','u_ml','v_ml']}
  coeff=86400/forcing['mlotst'].clamp_min(5)
  for name,path in [('observed',observed),('linear',linear)]:
   grads=h.spatial_gradients_torch(path[:,:-1],dx,dy,mask)
   flux=forcing['qnet']/(1025*3990*forcing['mlotst'].clamp_min(5))*86400
   adv=-(forcing['u_ml']*grads[0]+forcing['v_ml']*grads[1])*86400
   r=(path[:,1:]-path[:,:-1])-flux-adv
   rv=r[:,:,sea];cv=coeff[:,:,sea]
   unconstrained=-(cv*rv).sum(1)/(cv.square().sum(1).clamp_min(1e-12));positive=unconstrained.clamp_min(0)
   quantities={'zero_we_sq':rv.square().sum(),'positive_oracle_we_sq':(rv+cv*positive[:,None]).square().sum(),'signed_oracle_we_sq':(rv+cv*unconstrained[:,None]).square().sum(),'positive_residual_count':(rv>0).sum(),'signed_oracle_we_negative_count':(unconstrained<0).sum()*lead,'mean_residual_sq':rv.mean(1).square().sum()*lead,'temporal_variance':rv.var(1,unbiased=False).sum()*lead,'mean_signed_residual':rv.sum()}
   for k,v in quantities.items():sums[name+'_'+k]=sums.get(name+'_'+k,0.)+float(v)
  count+=int(rv.numel())
 result={k:v/count for k,v in sums.items()}
 for name in ['observed','linear']:
  for key in ['zero_we','positive_oracle_we','signed_oracle_we']:result[name+'_'+key+'_rms_kday']=math.sqrt(result[name+'_'+key+'_sq'])
  result[name+'_variance_fraction']=result[name+'_temporal_variance']/result[name+'_zero_we_sq']
 return result

def grads(loss,parameters,retain=True):return torch.autograd.grad(loss,parameters,retain_graph=retain,allow_unused=True)
def compare(a,b):
 dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
 na=math.sqrt(sum(float(x.detach().double().square().sum()) for x in a if x is not None));nb=math.sqrt(sum(float(x.detach().double().square().sum()) for x in b if x is not None))
 return dict(data_norm=na,physics_norm=nb,cosine=dot/max(na*nb,1e-30),weighted_ratio_at_point1=.1*nb/max(na,1e-30))
def main():
 OUT.mkdir(exist_ok=True);device=torch.device('cuda:0');torch.set_num_threads(2);w.gpu_identity(OUT)
 assert os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','0')
 jobs=[j for j in json.loads((OLD/'protocol.json').read_text())['jobs'] if j['ratio']==10 and j['arm']=='p0'];results=[]
 for j in jobs:
  args,_=w.arguments(j);w.t.set_seed(args.seed);bundle,m=p.load_subset_bundle(j['manifest']);pa=w.t.build_physics_args(args);p.apply_manifest_physics(args,pa,m)
  result=dict(lead=j['lead'],seed=j['seed'],manifest=j['manifest'],manifest_sha256=w.sha(j['manifest']),budget=budget(bundle.train.base,bundle.train_indices,j['lead'],device),gradients=[])
  model=w.t.JejuHiLAM(graph=bundle.graph,input_dim=len(bundle.train.input_feature_names),hidden_dim=256,temporal_hidden_dim=256,processor_layers=3,mlp_hidden_layers=1,prediction_mode='direct',physics_heads=True,we_nonneg=True,we_scale=args.we_scale,rres_scale=args.rres_scale,we_softplus_shift=args.we_softplus_shift).to(device)
  indices=np.linspace(0,len(bundle.train)-1,64,dtype=int);loader=DataLoader(Subset(bundle.train,indices),batch_size=16,shuffle=False)
  for state in ['initial','p0_best']:
   if state=='p0_best':
    cp=torch.load(Path(j['run_dir'])/'best.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model_state']);del cp
   model.train();parameters=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
   for bi,batch in enumerate(loader):
    output=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
    pred=w.t.nodes_to_grid_torch(output['pred']*bundle.stats['target'][1]+bundle.stats['target'][0],bundle.graph)
    we=w.t.nodes_to_grid_torch(output['we'],bundle.graph);mask=batch['mask'].to(device);tm=batch['Tm'].to(device);target=batch['y_raw'].to(device)
    data=torch.nn.functional.mse_loss(output['pred'],batch['y_nodes'].to(device));gd=grads(data,parameters)
    for mode,fn in [('daily',h.daily_rollout_physics_loss),('integral_diagnostic',integral)]:
     loss=fn(pred,tm,target,mask,batch,pa,j['lead'],we,torch.zeros_like(we));gp=grads(loss,parameters)
     result['gradients'].append(dict(state=state,batch=bi,mode=mode,data_loss=float(data.detach()),physics_loss=float(loss.detach()),**compare(gd,gp)))
    del gd,gp,loss,data,output,pred,we
  results.append(result);w.save(OUT/'diagnosis.json',dict(scope='selected training samples only; no test/optimizer; integral is a diagnostic comparator',status='running',conditions=results));print(json.dumps(result),flush=True)
  bundle.ds.close();del model,bundle,parameters;gc.collect();torch.cuda.empty_cache()
 w.save(OUT/'diagnosis.json',dict(scope='selected training samples only; no test/optimizer; integral is a diagnostic comparator',status='PASS',conditions=results));w.log('p0_repair_20261009/diagnosis.json: 10%×2lead×3seed의선택train 실제/직선경로budget 및oraclewe제약, 초기/best모델의data/physics predictorgradient진단완료. optimizer/Test평가없음. GPU0미사용.')
if __name__=='__main__':main()
