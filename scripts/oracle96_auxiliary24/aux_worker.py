"""Native data + fixed zero-target Rres regularization; no Test entrypoint."""
from aux_common import *
import worker as W
import argparse,math,random,copy,signal,traceback
import numpy as np
import torch
from dataclasses import replace

def setup(job,device):
 original=protocol()['jobs'][job['baseline_id']]
 a,b,model,_,cfg=W.setup(original,protocol(),device)
 ref=json.loads((Path(original['run_dir'])/'config.json').read_text())
 for key in ['stats','split_sha256','tensor_fingerprints','parameters','rhs_scale','shared_initial_sha256']:
  assert cfg[key]==ref[key],key
 # Independent native construction; tensor comparison rather than seed assumption.
 from types import SimpleNamespace
 rng_holder=SimpleNamespace(generator=torch.Generator());state=W.rng_state(rng_holder);W.t.set_seed(a.seed)
 baseline=W.t.JejuHiLAM(graph=b.graph,input_dim=len(b.train.input_feature_names),hidden_dim=a.hidden_dim,temporal_hidden_dim=a.temporal_hidden_dim,processor_layers=a.processor_layers,mlp_hidden_layers=a.mlp_hidden_layers,prediction_mode=a.prediction_mode,physics_heads=True,we_nonneg=True,we_scale=a.we_scale,rres_scale=a.rres_scale,we_softplus_shift=a.we_softplus_shift).to(device)
 assert all(torch.equal(v,baseline.state_dict()[k]) for k,v in model.state_dict().items())
 assert W.w.state_digest(baseline)==ref['shared_initial_sha256'];del baseline;W.restore_rng(state,rng_holder)
 a.lambda_physics=0.;a.lambda_rres=.0001;a.physics_warmup_epochs=0;a.output_root=Path(job['run_dir'])
 pa=W.t.build_physics_args(a);W.w.p.apply_manifest_physics(a,pa,json.loads(Path(job['manifest']).read_text()))
 assert pa.lambda_rres==.0001 and pa.lambda_we_tie==0 and pa.zero_rres_physics and pa.rres_penalty_normalize=='rhs' and pa.physics_rhs_scale==cfg['rhs_scale']
 for k,v in vars(a).items():
  if k not in ['lambda_rres','physics_warmup_epochs','output_root']:
   assert (str(v) if isinstance(v,Path) else v)==ref['settings'][k],('setting',k)
 cfg.update(job=job,baseline_reference=original['run_dir'],baseline_config_sha256=ref['config_sha256'],settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},checkpoint_dir=str(AUX/'checkpoints'/job['name']),lambda_schedule='physics OFF; auxiliary 0.0001 constant, no auxiliary warmup',test_evaluation='FORBIDDEN',active_objective_parameters=cfg['parameters']['shared']+cfg['parameters']['rres_head'],initial_tensor_equality=True,normalization='unchanged fixed subset v3 RHS mean-square',diagnostic='fixed first16 Train samples; separate forward/autograd.grad; preserve RNG/state/grad/modes')
 for flag,value in [('--lambda-rres','.0001'),('--physics-warmup-epochs','0'),('--output-root',job['run_dir'])]:cfg['argv'][cfg['argv'].index(flag)+1]=value
 cfg.pop('post_warmup_selection',None);cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg)
 return a,b,model,pa,cfg,ref

class Monitor(W.w.Monitor):
 def reset(self):super().reset();self.rres_sq=0.;self.rres_count=0
 def losses(self,*args,**kw):
  vals=self.original_loss(*args,**kw)
  assert all(torch.isfinite(x).all() for x in vals) and float(vals[2])==0 and float(vals[4])==0
  out=args[0] if args else kw['output']
  if isinstance(out,dict):
   x=out['rres'].detach().double();self.rres_sq+=float(x.square().sum());self.rres_count+=x.numel()
  self.batches+=1;return vals
 def clip(self,parameters,max_norm,*args,**kw):
  pars=list(parameters);norm=self.original_clip(pars,max_norm,error_if_nonfinite=True);n=float(norm);self.norms.append(n);self.streak=self.streak+1 if n>1e6 else 0
  if self.streak>=100:raise FloatingPointError('100 consecutive preclip norms >1e6')
  assert all(v.grad is None for name,v in self.model.named_parameters() if name.startswith('we_head.'))
  return norm
 def phys(self,*a,**kw):raise AssertionError('Disabled physics loss called')
 def rres(self,*a,**kw):self.rres_calls+=1;return self.original_rres(*a,**kw)
 def __enter__(self):
  super().__enter__();self.budget=W.t.physics_budget_diagnostics;self.raw=W.t.physics_raw_residual_stats;self.final=W.t.finalize_budget_metrics
  # Native epoch otherwise computes unused physics diagnostics even with lambda=0.
  W.t.physics_budget_diagnostics=lambda **kw:{}
  W.t.physics_raw_residual_stats=lambda **kw:{}
  W.t.finalize_budget_metrics=lambda sums:{}
  return self
 def __exit__(self,*a):
  W.t.physics_budget_diagnostics=self.budget;W.t.physics_raw_residual_stats=self.raw;W.t.finalize_budget_metrics=self.final;super().__exit__(*a)

def rng_equal(x,y):
 return torch.equal(x['loader_rng'],y['loader_rng']) and torch.equal(x['rng_cpu'],y['rng_cpu']) and len(x['rng_cuda'])==len(y['rng_cuda']) and all(torch.equal(a,b) for a,b in zip(x['rng_cuda'],y['rng_cuda'])) and x['rng_python']==y['rng_python'] and x['rng_numpy'][0]==y['rng_numpy'][0] and np.array_equal(x['rng_numpy'][1],y['rng_numpy'][1]) and x['rng_numpy'][2:]==y['rng_numpy'][2:]
def grad_digest(model):
 h=hashlib.sha256()
 for n,p in model.named_parameters():h.update(n.encode());h.update(b'None' if p.grad is None else p.grad.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def diagnostic(a,b,model,pa,loader,device,epoch,reason):
 rng=W.rng_state(loader);before=W.w.state_digest(model,False);grads=grad_digest(model);modes=[(m,m.training) for m in model.modules()]
 try:
  model.train();batch=next(iter(W.t.make_loader(b.train,a,False)))
  out=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
  with Monitor(model,pa) as mon:vals=W.t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],pa,0.,model)
  pars=[p for n,p in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
  gd=torch.autograd.grad(vals[1],pars,allow_unused=True,retain_graph=True);ga=torch.autograd.grad(vals[3],pars,allow_unused=True)
  norm=lambda gs:sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5
  nd,na=norm(gd),norm(ga);dot=sum(float((d.detach().double()*u.detach().double()).sum()) for d,u in zip(gd,ga) if d is not None and u is not None)
  data,aux=float(vals[1].detach()),float(vals[3].detach());rr=float(out['rres'].detach().double().square().mean().sqrt())
  row=dict(epoch=epoch,reason=reason,batch_size=len(batch['start_index']),sample_indices=batch['start_index'].tolist(),rres_rms_ks=rr,rres_rms_kday=rr*86400,auxiliary_loss=aux,weighted_auxiliary_loss=.0001*aux,data_loss=data,weighted_aux_data_loss_ratio=.0001*aux/data if data else None,shared_aux_gradient_norm=na,shared_data_gradient_norm=nd,weighted_gradient_ratio=.0001*na/nd if nd else None,gradient_cosine=dot/(na*nd) if na and nd else None,rres_last_layer_parameter_norm=sum(float(v.detach().double().square().sum()) for v in model.rres_head[-1].parameters())**.5,optimizer_updates=0)
  assert all(math.isfinite(v) for v in [data,aux,rr,nd,na])
 finally:
  W.restore_rng(rng,loader)
  for m,flag in modes:m.training=flag
 assert W.w.state_digest(model,False)==before and grad_digest(model)==grads and rng_equal(rng,W.rng_state(loader))
 row.update(model_state_unchanged=True,parameter_gradients_unchanged=True,RNG_unchanged=True,modes_unchanged=True)
 return row

def smoke(a,b,model,pa,cfg,d,device,ref):
 loader=W.t.make_loader(b.train,a,True);batch=next(iter(W.t.make_loader(b.train,a,False)));initial=copy.deepcopy(model.state_dict())
 rng=W.rng_state(loader);outcomes=[];opts=[]
 # One genuine optimizer step with and without diagnostic, from identical state/RNG.
 for enabled in [False,True]:
  model.load_state_dict(initial);model.zero_grad(set_to_none=True);W.restore_rng(rng,loader)
  opt=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95));W.t.set_epoch_lr(opt,a,1)
  model.train()
  if enabled:diag=diagnostic(a,b,model,pa,loader,device,0,'smoke_noninterference')
  with Monitor(model,pa) as mon:metrics=W.t.run_epoch(model,[batch],b.graph,device,*b.stats['target'],optimizer=opt,precision='fp32',physics_args=pa,lambda_physics=0.,epoch=1)
  assert mon.phys_calls==0 and mon.rres_calls==1 and len(mon.norms)==1
  outcomes.append(copy.deepcopy(model.state_dict()));opts.append(copy.deepcopy(opt.state_dict()));del opt
 assert all(torch.equal(v,outcomes[1][k]) for k,v in outcomes[0].items())
 for k,s in opts[0]['state'].items():
  for n,v in s.items():assert torch.equal(v,opts[1]['state'][k][n]) if isinstance(v,torch.Tensor) else v==opts[1]['state'][k][n]
 save(d/'diagnostic_noninterference.json',dict(status='PASS',weights_bitwise_equal=True,optimizer_state_equal=True,diagnostic=diag,metrics=metrics))
 torch.save(dict(model_state=model.state_dict()),d/'roundtrip.pt');cp=torch.load(d/'roundtrip.pt',map_location=device,weights_only=False)
 assert all(torch.equal(v,cp['model_state'][k]) for k,v in model.state_dict().items())
 save(d/'status.json',dict(status='PASS',checkpoint_roundtrip='PASS',diagnostic_noninterference='PASS',formal_training=False,disposable_smoke_steps=2,timestamp=stamp()))

def train(a,b,model,pa,cfg,d,device,ref):
 cpdir=Path(cfg['checkpoint_dir']);cpdir.mkdir(parents=True,exist_ok=True)
 loader=W.t.make_loader(b.train,a,True);val=W.t.make_loader(b.val,a,False);mon=Monitor(model,pa)
 opt=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95))
 baseline=json.loads((Path(cfg['baseline_reference'])/'history.json').read_text());history=[];diagnostics=[];best=float('inf');best_ep=0;steps=0;start=1
 last=cpdir/'last.pt'
 if last.exists():
  c=torch.load(last,map_location=device,weights_only=False);assert c['config_sha256']==cfg['config_sha256']
  model.load_state_dict(c['model_state']);opt.load_state_dict(c['optimizer_state']);W.restore_rng(c,loader)
  history=c['history'];diagnostics=c['diagnostics'];best=c['best_value'];best_ep=c['best_epoch'];steps=c['steps'];start=c['epoch']+1;mon.streak=c['gradient_guard_streak'];del c
 common=dict(graph=b.graph,device=device,target_mean=b.stats['target'][0],target_std=b.stats['target'][1],precision='fp32',physics_args=pa,lambda_physics=0.)
 elapsed=history[-1]['elapsed_seconds'] if history else 0.;begun=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for epoch in range(start,81):
  mon.reset();lr=W.t.set_epoch_lr(opt,a,epoch);assert lr==baseline[epoch-1]['lr'];checked=W.CheckedEpoch(loader,b.train_indices)
  save(d/'status.json',dict(status='RUNNING',pid=os.getpid(),epoch=epoch,steps=steps,timestamp=stamp()))
  with mon:
   tr=W.t.run_epoch(model,checked,optimizer=opt,epoch=epoch,**common)
   with torch.no_grad():va=W.t.run_epoch(model,val,epoch=epoch,**{**common,'physics_args':None})
  ev=checked.evidence();assert ev['sample_order_sha256']==baseline[epoch-1]['sample_order_sha256']
  assert mon.phys_calls==0 and mon.rres_calls==len(loader) and len(mon.norms)==len(loader)
  assert all(math.isfinite(v) for r in [tr,va] for v in r.values()) and tr['physics_loss']==0 and va['physics_loss']==0
  steps+=len(loader);improved=va['rmse']<best
  if improved:best=va['rmse'];best_ep=epoch
  if epoch in [1,5,10,20,80] or improved:
   diagnostics.append(diagnostic(a,b,model,pa,loader,device,epoch,'validation_best_candidate' if improved else 'scheduled'))
   save(d/'diagnostics.json',diagnostics)
  row=dict(epoch=epoch,actual_epoch=epoch,steps=steps,lr=lr,best_epoch=best_ep,best_val_rmse=best,lambda_aux=.0001,physics_calls=mon.phys_calls,rres_calls=mon.rres_calls,gradient_norm_mean=float(np.mean(mon.norms)),gradient_norm_max=max(mon.norms),gradient_clip_fraction=float(np.mean(np.array(mon.norms)>1)),rres_rms_ks=(mon.rres_sq/max(mon.rres_count,1))**.5,rres_rms_kday=86400*(mon.rres_sq/max(mon.rres_count,1))**.5,weighted_auxiliary_loss=.0001*tr['rres_l2'],weighted_aux_data_loss_ratio=.0001*tr['rres_l2']/tr['data_loss'] if tr['data_loss'] else None,rres_last_layer_parameter_norm=sum(float(v.detach().double().square().sum()) for v in model.rres_head[-1].parameters())**.5,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=elapsed+time.monotonic()-begun,**ev,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()})
  history.append(row)
  c=dict(model_state=model.state_dict(),config_sha256=cfg['config_sha256'],manifest_sha256=cfg['manifest_sha256'],epoch_type='actual',epoch=epoch,steps=steps,best_value=best,best_epoch=best_ep,stats=b.stats)
  if improved:W.w.checkpoint(cpdir/'best.pt',c)
  c.update(optimizer_state=opt.state_dict(),history=history,diagnostics=diagnostics,gradient_guard_streak=mon.streak,**W.rng_state(loader));W.w.checkpoint(last,c)
  save(d/'history.json',history);W.t.save_epoch_history(d/'history.csv',history);print(json.dumps(row),flush=True)
 assert len(history)==80 and steps==80*len(loader)
 cp=torch.load(cpdir/'best.pt',map_location=device,weights_only=False);assert cp['config_sha256']==cfg['config_sha256'];model.load_state_dict(cp['model_state'])
 with torch.no_grad():v=W.t.run_epoch(model,val,epoch=1,**{**common,'physics_args':None})
 assert abs(v['rmse']-best)<2e-6
 bestdiag=next(x for x in diagnostics if x['epoch']==best_ep);save(d/'best_diagnostic.json',bestdiag)
 save(d/'checkpoint_audit.json',dict(status='PASS',epoch=best_ep,expected_val_rmse=best,reloaded_val_rmse=v['rmse'],sha256=sha(cpdir/'best.pt')))
 save(d/'status.json',dict(status='COMPLETED',epochs=80,steps=steps,best_epoch=best_ep,best_val_rmse=best,early_stopping=False,checkpoint_roundtrip='PASS',test_evaluation='NOT_RUN',timestamp=stamp()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);ap.add_argument('--phase',choices=['smoke','train'],required=True);z=ap.parse_args();p=plan();verify(p)
 j=p['jobs'][z.job];d=AUX/'smoke'/j['name'] if z.phase=='smoke' else Path(j['run_dir']);d.mkdir(parents=True,exist_ok=True)
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in GPUS
 gate=Path(os.environ['AUX_START_GATE']);end=time.monotonic()+60
 while not gate.exists():assert time.monotonic()<end;time.sleep(.05)
 lease=json.loads((AUX/'leases.json').read_text())['running'];assert any(x['pid']==os.getpid() and x['gpu']==gpu and x['job']==z.job for x in lease) and sum(x['gpu']==gpu for x in lease)<=2
 def stop(sig,frame):raise KeyboardInterrupt(str(sig))
 signal.signal(signal.SIGTERM,stop)
 try:
  torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(14336*2**20/torch.cuda.get_device_properties(device).total_memory)
  W.w.gpu_identity(d)
  rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()));uid=next(x[1].strip() for x in rows if int(x[0])==gpu)
  assert all(x[0].strip()==uid for x in json.loads((d/'gpu_identity.json').read_text())['nvml'])
  a,b,model,pa,cfg,ref=setup(j,device)
  if (d/'config.json').exists():assert json.loads((d/'config.json').read_text())==cfg
  else:save(d/'config.json',cfg)
  if z.phase=='train':assert json.loads((AUX/'smoke_gate.json').read_text())['status']=='PASS'
  journal(f'{z.phase} start {j["name"]} GPU{gpu} PID{os.getpid()} λaux=.0001 physicsOFF TestOFF')
  (smoke if z.phase=='smoke' else train)(a,b,model,pa,cfg,d,device,ref)
  b.ds.close();journal(f'{z.phase} complete {j["name"]}')
 except BaseException as e:
  save(d/'status.json',dict(status='PAUSED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=traceback.format_exc(),timestamp=stamp()));raise
if __name__=='__main__':main()
