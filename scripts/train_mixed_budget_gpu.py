"""Manual v7 GPU worker. Reuses native model, optimizer schedule and epoch loop."""
from mixed_budget_common import *
import argparse,math,random,traceback
from dataclasses import replace
import numpy as np
import torch
import train_v3_study as w
import mixed_budget_physics as mb
t=w.t
ORIG_ARGS=w.arguments
ORIG_LOSS=t.daily_rollout_physics_loss
ORIG_BUDGET=t.physics_budget_diagnostics
ORIG_RAW=t.physics_raw_residual_stats
ORIG_FINAL=t.finalize_budget_metrics

def arguments(job):
 a,argv=ORIG_ARGS(dict(job,arm='p0'))
 a.lambda_physics=0. if job['model']=='B0' else .1;a.physics_warmup_epochs=0
 argv[argv.index('--lambda-physics')+1]=str(a.lambda_physics);argv+=['--physics-warmup-epochs','0']
 return a,argv

def setup(job,p,device):
 w.arguments=arguments
 a,b,model,pa,cfg=w.setup(dict(job,arm='p0'),p,device)
 fm=json.loads((FEATURES/'manifest.json').read_text());assert sha(FEATURES/'manifest.json')==p['feature_manifest_sha256']
 ar=mb.arrays();b=replace(b,**{s:mb.ForcingDataset(getattr(b,s),ar) for s in ['train','val','test']})
 pa.mixed_variant=job['model'];pa.lambda_physics=a.lambda_physics
 # Fixed common mean-square RHS scale is the unchanged v3 selected-training statistic.
 assert pa.physics_rhs_scale==cfg['rhs_scale'] and pa.lambda_rres==0 and pa.lambda_we_tie==0 and pa.zero_rres_physics
 cfg.update(job=job,architecture='Identical native predictor/we/rres heads in all arms; no extra head',normalization='common v3 train-only mean-square RHS; no per-equation rescaling',temperature='NOAA SST proxy, not prognostic GLORYS mixed-layer temperature',forcing='oracle loss-only; x_nodes unchanged',feature_manifest_sha256=p['feature_manifest_sha256'],selection='min full Validation SST RMSE every478 updates',test_evaluation='separate manual evaluation after all24 trained',lambda_schedule='fixed .1, physics warmup0; B0=0')
 cfg['active_objective_parameters']=cfg['parameters']['shared']+(cfg['parameters']['we_head'] if job['model']=='B1' else 0)
 cfg['settings']['lambda_physics']=a.lambda_physics;cfg['settings']['physics_warmup_epochs']=0
 cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg)
 return a,b,model,None if job['model']=='B0' else pa,cfg

def daily(*args,**kwargs):
 pa=args[5] if args else kwargs['args']
 if pa.mixed_variant in ['A','B']:return mb.loss(*args,**kwargs)
 xs=list(args)
 if xs:xs[3]=mb.common_mask(xs[3],xs[4])
 else:kwargs['mask']=mb.common_mask(kwargs['mask'],kwargs['batch'])
 return ORIG_LOSS(*xs,**kwargs)
def budget(**kw):
 if kw['args'].mixed_variant in ['A','B']:return mb.budget_stats(**kw)
 kw['mask']=mb.common_mask(kw['mask'],kw['batch']);return ORIG_BUDGET(**kw)
def raw(**kw):
 if kw['physics_args'].mixed_variant in ['A','B']:return mb.raw_stats(**kw)
 kw['mask']=mb.common_mask(kw['mask'],kw['batch']);return ORIG_RAW(**kw)
def final(s):return mb.finalize(s) if 'mixed_count' in s else ORIG_FINAL(s)
def install():
 t.daily_rollout_physics_loss=daily;t.physics_budget_diagnostics=budget;t.physics_raw_residual_stats=raw;t.finalize_budget_metrics=final

class Monitor(w.Monitor):
 def reset(self):super().reset();self.gradient_ratios=[]
 def losses(self,*args,**kw):
  vals=super().losses(*args,**kw)
  if self.model.training and torch.is_grad_enabled() and self.physics is not None and len(self.gradient_ratios)<2:
   pars=[v for n,v in self.model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
   gd=torch.autograd.grad(vals[1],pars,allow_unused=True,retain_graph=True);gp=torch.autograd.grad(vals[2],pars,allow_unused=True,retain_graph=True)
   nd=sum(float(x.detach().double().square().sum()) for x in gd if x is not None)**.5;np_=sum(float(x.detach().double().square().sum()) for x in gp if x is not None)**.5
   dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(gd,gp) if x is not None and y is not None)
   assert nd>0 and np_>0 and math.isfinite(np_)
   self.gradient_ratios.append(dict(data_norm=nd,physics_norm=np_,lambda_physics=.1,weighted_ratio=.1*np_/nd,cosine=dot/(nd*np_)))
  return vals
 def clip(self,parameters,max_norm,*args,**kw):
  n=super().clip(parameters,max_norm,*args,**kw)
  if self.physics is None or self.physics.mixed_variant in ['A','B']:
   assert all(v.grad is None for name,v in self.model.named_parameters() if name.startswith(('we_head.','rres_head.')))
  return n

def smoke(a,b,model,pa,cfg,d,device):
 loader=t.make_loader(b.train,a,True);opt=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95));mon=Monitor(model,pa)
 with mon:r=t.run_epoch(model,loader,b.graph,device,*b.stats['target'],optimizer=opt,max_batches=2,precision='fp32',physics_args=pa,lambda_physics=a.lambda_physics,epoch=1)
 assert len(mon.norms)==2 and all(math.isfinite(v) for v in r.values())
 assert mon.phys_calls==(0 if pa is None else 2) and mon.rres_calls==0
 sample=next(iter(t.make_loader(b.val,a,False)));model.eval()
 with torch.no_grad():p1=model(sample['x_nodes'].to(device),base_nodes=sample['tm_norm_nodes'].to(device)).detach().cpu()
 w.checkpoint(d/'smoke.pt',dict(model_state=model.state_dict(),config_sha256=cfg['config_sha256']))
 cp=torch.load(d/'smoke.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model_state'],strict=True)
 with torch.no_grad():p2=model(sample['x_nodes'].to(device),base_nodes=sample['tm_norm_nodes'].to(device)).detach().cpu()
 assert torch.equal(p1,p2)
 if pa is not None:assert all(x['physics_norm']>0 for x in mon.gradient_ratios)
 save(d/'status.json',dict(status='PASS',config_sha256=cfg['config_sha256'],forward=True,backward=True,optimizer_steps=2,checkpoint_roundtrip=True,metrics=r,gradient_norms=mon.norms,predictor_gradients=mon.gradient_ratios,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,timestamp=stamp()))

def train(a,b,model,pa,cfg,d,device):
 stream=w.Stream(t.make_loader(b.train,a,True));opt=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95));mon=Monitor(model,pa)
 common=dict(graph=b.graph,device=device,target_mean=b.stats['target'][0],target_std=b.stats['target'][1],precision='fp32',physics_args=pa,lambda_physics=a.lambda_physics)
 val=t.make_loader(b.val,a,False);history=[];best=float('inf');best_epoch=0;bad=0;start=1
 last=d/'last.pt'
 if last.exists():
  cp=torch.load(last,map_location=device,weights_only=False);assert cp['config_sha256']==cfg['config_sha256'];model.load_state_dict(cp['model_state']);opt.load_state_dict(cp['optimizer_state']);stream.restore({**cp['stream'],'start':cp['stream']['start'].cpu()});history=cp['history'];best=cp['best_value'];best_epoch=cp['best_epoch'];bad=cp['bad_epochs'];start=cp['epoch']+1
  torch.set_rng_state(cp['rng_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in cp['rng_cuda']]);np.random.set_state(cp['rng_numpy']);random.setstate(cp['rng_python']);mon.streak=cp['gradient_guard_streak'];del cp
 begun=time.time();torch.cuda.reset_peak_memory_stats()
 for epoch in range(start,a.epochs+1):
  if bad>=a.patience:break
  mon.reset();lr=t.set_epoch_lr(opt,a,epoch);save(d/'status.json',dict(status='training',pid=os.getpid(),epoch=epoch,steps=stream.total,timestamp=stamp()))
  with mon:
   tr=t.run_epoch(model,stream.take(478),optimizer=opt,epoch=epoch,**common)
   with torch.no_grad():va=t.run_epoch(model,val,epoch=epoch,**common)
  assert len(mon.norms)==478 and mon.rres_calls==0
  for r in [tr,va]:assert all(math.isfinite(v) for v in r.values())
  improved=va['rmse']<best
  if improved:best=va['rmse'];best_epoch=epoch;bad=0
  else:bad+=1
  row=dict(epoch=epoch,steps=stream.total,lr=lr,best_epoch=best_epoch,best_val_rmse=best,gradient_norm_mean=float(np.mean(mon.norms)),gradient_norm_max=max(mon.norms),gradient_clip_fraction=float(np.mean(np.array(mon.norms)>1)),predictor_gradient_samples=mon.gradient_ratios,batch_order_sha256=stream.chain,physics_calls=mon.phys_calls,rres_calls=mon.rres_calls,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.time()-begun,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()})
  history.append(row)
  cp=dict(model_state=model.state_dict(),config_sha256=cfg['config_sha256'],manifest_sha256=cfg['manifest_sha256'],best_epoch=best_epoch,best_value=best,stats=b.stats)
  if improved:w.checkpoint(d/'best.pt',cp)
  cp.update(epoch=epoch,optimizer_state=opt.state_dict(),history=history,bad_epochs=bad,stream=stream.state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),gradient_guard_streak=mon.streak)
  w.checkpoint(last,cp);save(d/'history.json',history);t.save_epoch_history(d/'history.csv',history);print(json.dumps(row),flush=True)
 cp=torch.load(d/'best.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model_state'],strict=True)
 with torch.no_grad():reval=t.run_epoch(model,val,epoch=1,**common)
 assert abs(reval['rmse']-best)<2e-6
 save(d/'status.json',dict(status='trained',steps=stream.total,epochs=len(history),best_epoch=best_epoch,best_val_rmse=best,early_stopping=bad>=a.patience,checkpoint_roundtrip='PASS',peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,timestamp=stamp()))

def evaluate(a,b,model,pa,cfg,d,device):
 cp=torch.load(d/'best.pt',map_location=device,weights_only=False);assert cp['config_sha256']==cfg['config_sha256'];model.load_state_dict(cp['model_state'],strict=True)
 for split in ['val','test']:
  out=d/'evaluation'/split;out.mkdir(parents=True,exist_ok=True)
  if (out/'metrics.json').exists():
   old=json.loads((out/'metrics.json').read_text());assert old['checkpoint_sha256']==sha(d/'best.pt') and old['config_sha256']==cfg['config_sha256'];continue
  f=t.collect_predictions(model,t.make_loader(getattr(b,split),a,False),b.graph,device,*b.stats['target'],precision='fp32',physics_args=pa)
  assert np.array_equal(f['target_index'],getattr(b,split+'_indices')+29+a.lead)
  pred=f['pred_raw'].squeeze(1) if f['pred_raw'].ndim==4 else f['pred_raw'];obs=f['target_raw'].squeeze(1) if f['target_raw'].ndim==4 else f['target_raw'];sea=np.squeeze(f['mask'][0])>0;er=pred[:,sea].astype('float64')-obs[:,sea]
  rmse=float(np.sqrt(np.mean(er**2)))
  if split=='val':assert abs(rmse-cp['best_value'])<2e-6
  f['target_dates']=b.ds.time.values[f['target_index']];np.savez_compressed(out/'fields.npz',**f)
  save(out/'metrics.json',dict(rmse=rmse,mae=float(np.mean(abs(er))),bias=float(er.mean()),checkpoint_sha256=sha(d/'best.pt'),config_sha256=cfg['config_sha256'],timestamp=stamp()))
 save(d/'evaluation_status.json',dict(status='completed',timestamp=stamp()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);ap.add_argument('--phase',choices=['smoke','train','evaluate'],required=True);o=ap.parse_args()
 gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'));assert gpu in [0,1,2],'Only one of physical GPU0/1/2 may be visible'
 expected_uuid=gpu_idle(gpu);p=protocol();verify_code(p);assert 0<=o.job<len(p['jobs']);job=p['jobs'][o.job];d=Path(job['run_dir']) if o.phase!='smoke' else STUDY/'smoke'/job['name'];d.mkdir(parents=True,exist_ok=True)
 save(d/f'{o.phase}_pid.json',dict(pid=os.getpid(),gpu=gpu,phase=o.phase,timestamp=stamp()))
 torch.set_num_threads(1);install();device=torch.device('cuda:0')
 try:
  w.gpu_identity(d);identity=json.loads((d/'gpu_identity.json').read_text());assert all(r[0].strip()==expected_uuid for r in identity['nvml']), 'CUDA physical GPU mapping mismatch'
  if o.phase=='train':
   assert json.loads((STUDY/'smoke_test_report.json').read_text())['status']=='PASS'
   if o.job>=4:assert json.loads((STUDY/'representative_report.json').read_text())['status']=='PASS'
  if o.phase=='evaluate':assert all(json.loads((Path(j['run_dir'])/'status.json').read_text())['status']=='trained' for j in p['jobs'])
  a,b,model,pa,cfg=setup(job,p,device)
  if (d/'config.json').exists():assert json.loads((d/'config.json').read_text())==cfg
  else:save(d/'config.json',cfg)
  log(f'{o.phase} 시작 {job["name"]}, GPU{gpu}, PID{os.getpid()}, λ={a.lambda_physics}; oracle forcing, NOAA proxy, no adaptive.')
  {'smoke':smoke,'train':train,'evaluate':evaluate}[o.phase](a,b,model,pa,cfg,d,device);b.ds.close();log(f'{o.phase} 완료 {job["name"]}')
 except BaseException:
  save(d/f'failure_{o.phase}_{time.time_ns()}.json',dict(error=traceback.format_exc(),timestamp=stamp()));save(d/'status.json',dict(status='failed',phase=o.phase,timestamp=stamp()));raise
if __name__=='__main__':main()
