"""Actual epochs; unchanged scientific model/loss modules; validation-only selection."""
from mixed_budget_common import *
import importlib.util,argparse,math,random,signal,traceback
from dataclasses import replace
import numpy as np
import torch
# Resolve the frozen oracle physics implementation against the new output context.
sys.path.insert(1,str(ROOT/'scripts/oracle96'))
spec=importlib.util.spec_from_file_location('frozen_oracle_worker',ROOT/'scripts/oracle96/train_mixed_budget_gpu.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
w=old.w;t=old.t;mb=old.mb

def arguments(job):
 argv=['trainer','--subset-manifest',job['manifest'],'--lead',str(job['lead']),'--seed',str(job['seed']),
 '--physics-mode','p0_original','--lambda-physics',str(job['lambda_physics']),
 '--lambda-rres','0','--lambda-we-tie','0','--batch-size','16','--epochs','80','--patience','81',
 '--physics-warmup-epochs','5','--warmup-epochs','5','--lr','0.0002','--initial-lr','0.00002',
 '--no-residual-feature','--device','cuda:0','--precision','fp32','--output-root',job['run_dir']]
 prev=sys.argv;sys.argv=argv
 try:a=t.parse_args()
 finally:sys.argv=prev
 a.virtual_epoch_steps=None
 return a,argv[1:]

def setup(job,p,device):
 old.arguments=arguments
 a,b,model,pa,c=old.setup(job,p,device)
 n=len(b.train);upe=math.ceil(n/a.batch_size)
 assert n==job['n_train'] and upe==job['updates_per_epoch']
 if pa is not None:assert pa.physics_warmup_epochs==5
 c.pop('fixed_updates',None)
 c.update(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},epoch_type='actual',updates_per_epoch=upe,total_updates=80*upe,
 selection='minimum full Validation SST RMSE over 80 actual epochs',post_warmup_selection='epoch >= 5, first epoch at full physics weight',
 test_evaluation='disabled; separate plan after all96 completed',lambda_schedule='config lambda * min(epoch/5,1), constant within actual epoch',
 early_stopping_enabled=False,normalization='unchanged v3 selected-training statistics and common P0 mean-square RHS scale')
 c.pop('config_sha256');c['config_sha256']=digest(c)
 return a,b,model,pa,c

class CheckedEpoch:
 """One finite native shuffled DataLoader pass; assert manifest sample identity."""
 def __init__(self,loader,expected):self.loader=loader;self.expected=list(map(int,expected));self.indices=[];self.batch_sizes=[]
 def __iter__(self):
  assert not self.indices,'An actual epoch may be consumed only once'
  for batch in self.loader:
   ids=list(map(int,batch['start_index'].tolist()));self.indices.extend(ids);self.batch_sizes.append(len(ids));yield batch
  assert len(self.indices)==len(self.expected) and sorted(self.indices)==sorted(self.expected),'duplicate/missing training window'
  assert len(set(self.indices))==len(self.indices)
  assert len(self.batch_sizes)==math.ceil(len(self.expected)/self.loader.batch_size)
  assert self.batch_sizes[-1]==(len(self.expected)-1)%self.loader.batch_size+1
 def evidence(self):return dict(train_samples=len(self.indices),epoch_updates=len(self.batch_sizes),last_batch_samples=self.batch_sizes[-1],sample_order_sha256=digest(self.indices),sampling_exact_once=True)

class Monitor(old.Monitor):
 def losses(self,*args,**kw):
  vals=w.Monitor.losses(self,*args,**kw)
  if self.model.training and torch.is_grad_enabled() and self.physics is not None and len(self.gradient_ratios)<2:
   effective=float(args[6] if args else kw['lambda_eff'])
   pars=[v for n,v in self.model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
   gd=torch.autograd.grad(vals[1],pars,allow_unused=True,retain_graph=True);gp=torch.autograd.grad(vals[2],pars,allow_unused=True,retain_graph=True)
   nd=sum(float(x.detach().double().square().sum()) for x in gd if x is not None)**.5;ng=sum(float(x.detach().double().square().sum()) for x in gp if x is not None)**.5
   dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(gd,gp) if x is not None and y is not None)
   assert nd>0 and ng>0 and math.isfinite(ng) and effective>0
   self.gradient_ratios.append(dict(data_norm=nd,physics_norm=ng,lambda_physics_effective=effective,weighted_ratio=effective*ng/nd,cosine=dot/(nd*ng)))
  return vals

def rng_state(loader):
 return dict(loader_rng=loader.generator.get_state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],rng_numpy=np.random.get_state(),rng_python=random.getstate())
def restore_rng(c,loader):
 loader.generator.set_state(c['loader_rng'].cpu());torch.set_rng_state(c['rng_cpu'].cpu())
 if c['rng_cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in c['rng_cuda']])
 np.random.set_state(c['rng_numpy']);random.setstate(c['rng_python'])

def run_training(a,b,model,pa,cfg,d,device,*,stop_after_epoch=None,invariance=True):
 """stop_after_epoch is CPU regression injection only, never exposed by worker CLI."""
 assert a.epochs==80 and a.virtual_epoch_steps is None and a.physics_warmup_epochs==5
 cpdir=Path(cfg['checkpoint_dir']);cpdir.mkdir(parents=True,exist_ok=True);d.mkdir(parents=True,exist_ok=True)
 loader=t.make_loader(b.train,a,True);assert not loader.drop_last and not loader.sampler.replacement
 opt=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95));mon=Monitor(model,pa)
 common=dict(graph=b.graph,device=device,target_mean=b.stats['target'][0],target_std=b.stats['target'][1],precision='fp32',physics_args=pa,lambda_physics=a.lambda_physics)
 val=t.make_loader(b.val,a,False);history=[];best=float('inf');best_ep=0;post=float('inf');post_ep=0;start=1;steps=0
 last=cpdir/'last.pt';expected_steps=80*len(loader)
 if last.exists():
  c=torch.load(last,map_location=device,weights_only=False);assert c['config_sha256']==cfg['config_sha256'] and c['epoch_type']=='actual'
  assert c['epoch']<=80 and c['steps']==c['epoch']*len(loader)
  model.load_state_dict(c['model_state'],strict=True);opt.load_state_dict(c['optimizer_state']);restore_rng(c,loader)
  history=c['history'];best=c['best_value'];best_ep=c['best_epoch'];post=c['post_best_value'];post_ep=c['post_best_epoch'];start=c['epoch']+1;steps=c['steps'];mon.streak=c['gradient_guard_streak'];del c
 elapsed=history[-1]['elapsed_seconds'] if history else 0.;begun=time.time()
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
 for epoch in range(start,81):
  mon.reset();lr=t.set_epoch_lr(opt,a,epoch);checked=CheckedEpoch(loader,b.train_indices)
  save(d/'status.json',dict(status='RUNNING',pid=os.getpid(),epoch=epoch,steps=steps,timestamp=stamp()))
  with mon:
   tr=t.run_epoch(model,checked,optimizer=opt,epoch=epoch,**common)
   assert model.training
   with torch.no_grad():va=t.run_epoch(model,val,epoch=epoch,**{**common,'physics_args':None,'lambda_physics':0.})
   assert not model.training
  evidence=checked.evidence();steps+=evidence['epoch_updates']
  assert len(mon.norms)==len(loader) and mon.rres_calls==0 and mon.phys_calls==(0 if pa is None else len(loader))
  assert all(math.isfinite(x) for r in [tr,va] for x in r.values())
  factor=t.physics_lambda_factor(epoch,5);effective=a.lambda_physics*factor if pa is not None else 0.
  assert abs(tr['lambda_physics_effective']-effective)<1e-12 and va['physics_loss']==0
  improved=va['rmse']<best;post_improved=epoch>=5 and va['rmse']<post
  if improved:best=va['rmse'];best_ep=epoch
  if post_improved:post=va['rmse'];post_ep=epoch
  row=dict(epoch=epoch,actual_epoch=epoch,steps=steps,lr=lr,physics_warmup_factor=factor,lambda_phys_effective=effective,
   best_epoch=best_ep,best_val_rmse=best,post_warmup_best_epoch=post_ep,post_warmup_best_val_rmse=post if post_ep else None,
   gradient_norm_mean=float(np.mean(mon.norms)),gradient_norm_max=max(mon.norms),gradient_clip_fraction=float(np.mean(np.array(mon.norms)>1)),
   predictor_gradient_samples=mon.gradient_ratios,physics_calls=mon.phys_calls,rres_calls=mon.rres_calls,adversarial_loss=0.,
   peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0.,elapsed_seconds=elapsed+time.time()-begun,
   **evidence,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()})
  history.append(row)
  c=dict(model_state=model.state_dict(),epoch_type='actual',epoch=epoch,steps=steps,config_sha256=cfg['config_sha256'],manifest_sha256=cfg['manifest_sha256'],
    best_epoch=best_ep,best_value=best,post_best_epoch=post_ep,post_best_value=post,stats=b.stats)
  if improved:w.checkpoint(cpdir/'best.pt',c)
  if post_improved:w.checkpoint(cpdir/'post_warmup_best.pt',c)
  c.update(optimizer_state=opt.state_dict(),history=history,gradient_guard_streak=mon.streak,**rng_state(loader))
  w.checkpoint(last,c);save(d/'history.json',history);t.save_epoch_history(d/'history.csv',history)
  print(json.dumps(row),flush=True)
  if stop_after_epoch==epoch:
   save(d/'status.json',dict(status='PAUSED',epoch=epoch,steps=steps));return history
 assert len(history)==80 and steps==expected_steps
 checks={}
 for name,value in [('post_warmup_best',post),('best',best)]:
  c=torch.load(cpdir/(name+'.pt'),map_location=device,weights_only=False);assert c['config_sha256']==cfg['config_sha256'];model.load_state_dict(c['model_state'],strict=True)
  with torch.no_grad():v=t.run_epoch(model,val,epoch=1,**{**common,'physics_args':None,'lambda_physics':0.})
  assert abs(v['rmse']-value)<2e-6
  checks[name]=dict(epoch=c['epoch'],expected_val_rmse=value,reloaded_val_rmse=v['rmse'],sha256=sha(cpdir/(name+'.pt')),status='PASS')
 if invariance:old.prediction_invariance(a,b,model,d,device,'val')
 save(d/'checkpoint_audit.json',checks)
 save(d/'status.json',dict(status='COMPLETED',epochs=80,validation_selection_calls=80,steps=steps,best_epoch=best_ep,best_val_rmse=best,
 post_warmup_best_epoch=post_ep,post_warmup_best_val_rmse=post,post_minus_overall_rmse=post-best,
 early_stopping=False,checkpoint_roundtrip='PASS',checkpoint_dir=str(cpdir),timestamp=stamp()))
 return history

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);ap.add_argument('--phase',choices=['smoke','train'],required=True);z=ap.parse_args()
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in [7,8,9]
 p=protocol();verify_code(p);assert json.loads((STUDY/'preflight.json').read_text())['status']=='PASS'
 job=p['jobs'][z.job];d=STUDY/'smoke'/job['name'] if z.phase=='smoke' else Path(job['run_dir']);d.mkdir(parents=True,exist_ok=True)
 gate=Path(os.environ['ACTUAL_START_GATE']);deadline=time.monotonic()+60
 while not gate.exists():
  assert time.monotonic()<deadline;time.sleep(.05)
 lease=json.loads((STUDY/'leases.json').read_text())['running'];assert any(x['pid']==os.getpid() and x['gpu']==gpu and x['job']==z.job for x in lease)
 assert sum(x['gpu']==gpu for x in lease)<=2
 save(d/'pid.json',dict(pid=os.getpid(),gpu=gpu,job=z.job,phase=z.phase,timestamp=stamp()))
 def interrupted(sig,frame):raise KeyboardInterrupt(f'user/scheduler signal {sig}')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  torch.set_num_threads(1);old.install();device=torch.device('cuda:0')
  torch.cuda.set_per_process_memory_fraction(14336*2**20/torch.cuda.get_device_properties(device).total_memory,device)
  w.gpu_identity(d)
  rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()));uuid=next(r[1].strip() for r in rows if int(r[0])==gpu)
  assert all(r[0].strip()==uuid for r in json.loads((d/'gpu_identity.json').read_text())['nvml'])
  a,b,model,pa,c=setup(job,p,device);c['checkpoint_dir']=str(STUDY/'checkpoints'/job['name']);c.pop('config_sha256');c['config_sha256']=digest(c)
  if (d/'config.json').exists():assert json.loads((d/'config.json').read_text())==c
  else:save(d/'config.json',c)
  if z.phase=='train':
   assert json.loads((STUDY/'smoke_gate.json').read_text())['status']=='PASS'
   if z.job>=4:assert json.loads((STUDY/'representative_gate.json').read_text())['status']=='PASS'
  log(f'{z.phase} start {job["name"]} GPU{gpu} PID{os.getpid()}, actual80 epochs, updates={job["total_updates"]}, warmup5')
  if z.phase=='smoke':
   old.Monitor=Monitor;old.smoke(a,b,model,pa,c,d,device)
   s=json.loads((d/'status.json').read_text());expected=job['lambda_physics']*.2
   assert abs(s['metrics']['lambda_physics_effective']-expected)<1e-12
   assert all(abs(x['lambda_physics_effective']-expected)<1e-12 for x in s['predictor_gradients'])
  else:run_training(a,b,model,pa,c,d,device)
  b.ds.close();log(f'{z.phase} complete {job["name"]}')
 except BaseException as e:
  save(d/'status.json',dict(status='PAUSED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=traceback.format_exc(),timestamp=stamp()))
  log(f'{z.phase} interrupted/failed {job["name"]}; no automatic retry');raise
if __name__=='__main__':main()
