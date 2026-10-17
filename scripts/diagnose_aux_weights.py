"""Initial shared-gradient audit across 3 seeds x 8 fixed training batches. No optimizer."""
import argparse,gc,math
import train_v3_ablation as a
from train_v3_study import *
from torch.utils.data import Subset,DataLoader

def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs if g is not None))
def cosine(g,h):return sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(g,h) if x is not None and y is not None)/(norm(g)*norm(h))
def choose_weights(q):
 median=float(np.median(q));base=10.**math.floor(math.log10(1./median))
 candidates=[float(f'{v:.12g}') for v in [base/10,base,3*base]]
 assert len(set(candidates))==3 and all(0<v<10 for v in candidates)
 return candidates

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);opt=ap.parse_args();out=opt.study;device=torch.device('cuda:0');torch.set_num_threads(2)
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3'];gpu_identity(out)
 source=ROOT/'outputs/v3_ablation_lead7_20261009';protocol=json.loads((source/'protocol.json').read_text());rows=[];indices={}
 for seed in [42,7,123]:
  j=next(j for j in protocol['jobs'] if j['seed']==seed and j['arm']=='auxiliary_only');j=dict(j,run_dir=str(out/f'diagnostic_seed{seed}'))
  args,bundle,model,pa,cfg=a.setup(j,protocol,device);model.train();initial=state_digest(model,shared=False)
  ids=np.linspace(0,len(bundle.train)-1,128,dtype=int);assert len(np.unique(ids))==128
  indices[str(seed)]=bundle.train_indices[ids].tolist();loader=DataLoader(Subset(bundle.train,ids),batch_size=16,shuffle=False)
  params=[p for n,p in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
  for bi,batch in enumerate(loader):
   output=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
   vals=t.compute_batch_losses(output,batch,bundle.graph,*bundle.stats['target'],pa,0.,model)
   gd=torch.autograd.grad(vals[1],params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(vals[3],params,allow_unused=True)
   nd,na=norm(gd),norm(ga);assert nd>0 and na>0 and math.isfinite(nd+na)
   row=dict(seed=seed,batch=bi,target_indices=batch['target_index'].tolist(),data_loss=float(vals[1].detach()),auxiliary_loss=float(vals[3].detach()),data_gradient_norm=nd,auxiliary_gradient_norm=na,unweighted_aux_data_ratio=na/nd,weighted_ratio_at_10=10*na/nd,loss_ratio_at_10=10*float(vals[3].detach())/float(vals[1].detach()),cosine=cosine(gd,ga))
   assert all(math.isfinite(v) for v in row.values() if isinstance(v,float));rows.append(row);print(json.dumps(row),flush=True)
   del vals,output,gd,ga
  assert state_digest(model,shared=False)==initial
  save(out/'gradient_progress.json',dict(rows=rows,selected_train_start_indices=indices));bundle.ds.close();del model,bundle,params;gc.collect();torch.cuda.empty_cache()
 q=np.array([r['unweighted_aux_data_ratio'] for r in rows]);candidates=choose_weights(q)
 def stats(v):return dict(min=float(np.min(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),max=float(np.max(v)))
 result=dict(status='PASS',scope='initial models; 3 seeds x8 batches x16 selected train windows; no optimizer; no Validation/Test gradients',batches=rows,selected_train_start_indices=indices,weighted_10_summary=stats(q*10),by_seed={str(s):stats([r['weighted_ratio_at_10'] for r in rows if r['seed']==s]) for s in [42,7,123]},opposing_batches=sum(r['cosine']<0 for r in rows),candidate_rule='base=10^floor(log10(1/median unweighted shared-gradient ratio)); candidates=base/10,base,3base; fixed before validation',candidates=candidates,candidate_gradient_summaries={str(beta):stats(q*beta) for beta in candidates},script_sha256=sha(__file__))
 save(out/'gradient_diagnostic.json',result);log('Auxiliary multi-batch initial gradient audit complete: '+str(result['weighted_10_summary'])+'; candidates='+str(candidates));print(json.dumps({k:v for k,v in result.items() if k not in ['batches','selected_train_start_indices']}),flush=True)
if __name__=='__main__':main()
