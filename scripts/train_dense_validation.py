"""Dense validation with unchanged optimizer-update schedule and native P0 gradients."""
import os,json,math,time,random,traceback
from pathlib import Path
from dense_eval_common import *

def write_config(d,cfg,p,job):
 cfg.pop('config_sha256');cfg.update(protocol_sha256=w.digest(p),evaluation_interval=INTERVAL,schedule_block=BLOCK,max_optimizer_steps=MAX_STEPS,patience_optimizer_steps=PATIENCE,selection='validation SST RMSE, dense checkpoints; no Test inference')
 cfg['config_sha256']=w.digest(cfg)
 if (d/'config.json').exists():assert json.loads((d/'config.json').read_text())==cfg
 else:w.save(d/'config.json',cfg)
 return cfg

def verify_reference(cfg,job):
 ref=json.loads((Path(job['reference_run'])/'config.json').read_text())
 for k in ['shared_initial_sha256','stats','manifest_sha256','split_sha256','tensor_fingerprints','parameters']:assert cfg[k]==ref[k],k
 for k in cfg['settings']:
  if k!='output_root':assert cfg['settings'][k]==ref['settings'][k],k

def run_diagnostics(study,job,p,device):
 d=Path(job['run_dir']);results=[];a=b=m=None;current_arm=None
 for source in job['sources']:
  arm=source['arm']
  if arm!=current_arm:
   if b is not None:b.ds.close();del m,b;torch.cuda.empty_cache()
   setup_job={**job,'arm':arm,'lambda_physics':source['lambda_physics']}
   a,b,m,pa,cfg=w.setup(setup_job,p,device);current_arm=arm
  sd=Path(source['run_dir']);sc=json.loads((sd/'config.json').read_text());cp=torch.load(sd/'best.pt',map_location=device,weights_only=False)
  assert cp['config_sha256']==sc['config_sha256'] and cp['manifest_sha256']==job['manifest_sha256']
  for key in ['stats','manifest_sha256','split_sha256','tensor_fingerprints','shared_initial_sha256']:assert cfg[key]==sc[key]
  m.load_state_dict(cp['model_state'],strict=True);del cp
  diag=common_diagnostics(m,b,a,device,observations=not results)
  h=json.loads((sd/'history.json').read_text());best=min(h,key=lambda r:r['val_rmse'])
  assert abs(diag['val_rmse']-best['val_rmse'])<2e-6
  if arm!='baseline':assert abs(diag['learned_we']['rmse_kday']-best['val_common_res_rmse_kday'])<2e-6
  if results:assert diag['target_index_sha256']==results[0]['diagnostics']['target_index_sha256'] and diag['zero_we']['count']==results[0]['diagnostics']['zero_we']['count']
  results.append(dict(source=source,diagnostics=diag));w.save(d/'diagnostics.json',dict(status='running',results=results));print(source['lambda_physics'],diag['val_rmse'],diag['zero_we'],flush=True)
 b.ds.close();w.save(d/'diagnostics.json',dict(status='complete',results=results))
 w.save(d/'config.json',dict(protocol_sha256=w.digest(p),manifest_sha256=job['manifest_sha256'],job=job))
 w.save(d/'status.json',dict(status='trained',kind='diagnostics',timestamp=w.stamp(),checkpoints_evaluated=len(results)))
 w.log('共通残差診断完了 '+job['name']+'; validation only, we=0 common and learned-we separate.')

def run_training(study,job,p,device):
 d=Path(job['run_dir']);a,b,m,pa,cfg=w.setup(job,p,device);verify_reference(cfg,job);cfg=write_config(d,cfg,p,job)
 loader=t.make_loader(b.train,a,True);assert len(loader)==INTERVAL
 stream=w.Stream(loader);opt=torch.optim.AdamW(m.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95));monitor=w.Monitor(m,pa)
 common=dict(graph=b.graph,device=device,target_mean=b.stats['target'][0],target_std=b.stats['target'][1],physics_args=pa,lambda_physics=a.lambda_physics,precision='fp32')
 history=[];best=float('inf');best_step=0;coarse_best=float('inf');coarse_step=0;last=d/'last.pt'
 if last.exists():
  cp=torch.load(last,map_location=device,weights_only=False);assert cp['config_sha256']==cfg['config_sha256']
  m.load_state_dict(cp['model_state']);opt.load_state_dict(cp['optimizer_state']);stream.restore({**cp['stream'],'start':cp['stream']['start'].cpu()})
  history=cp['history'];best=cp['best_value'];best_step=cp['best_step'];coarse_best=cp['coarse_best'];coarse_step=cp['coarse_step'];monitor.streak=cp['gradient_guard_streak']
  torch.set_rng_state(cp['rng_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in cp['rng_cuda']]);np.random.set_state(cp['rng_numpy']);random.setstate(cp['rng_python']);del cp
 else:
  # Reproduce original worker's no-update P0 diagnostic before the first step.
  if pa is not None:
   batch=next(iter(t.make_loader(b.train,a,False)));m.train();out=m(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
   vals=t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],pa,.1,m)
   grads=torch.autograd.grad(vals[2],[v for n,v in m.named_parameters() if not n.startswith(('we_head.','rres_head.'))],allow_unused=True)
   norm=math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None));assert norm>0 and math.isfinite(norm)
   w.save(d/'initial_physics_gradient.json',dict(shared_norm=norm));del grads,vals,out,batch
 oldhist=json.loads((Path(job['reference_run'])/'history.json').read_text());reference={r['steps']:r for r in oldhist}
 begun=time.time();torch.cuda.reset_peak_memory_stats();w.log('dense validation training start '+job['name']+'; 24-step eval,478-step schedule,38240 max,5736 patience; GPU'+os.environ['CUDA_VISIBLE_DEVICES'])
 while stream.total<MAX_STEPS and stream.total-best_step<PATIENCE:
  epoch=stream.total//BLOCK+1;stop=next_event(stream.total,best_step);n=stop-stream.total;assert n>0
  lr=t.set_epoch_lr(opt,a,epoch);monitor.reset()
  w.save(d/'status.json',dict(status='training',steps=stream.total,next_evaluation_step=stop,best_step=best_step,pid=os.getpid(),timestamp=w.stamp()))
  with monitor:train=t.run_epoch(m,stream.take(n),optimizer=opt,epoch=epoch,**common)
  assert len(monitor.norms)==n and monitor.rres_calls==0 and monitor.phys_calls==(n if pa is not None else 0)
  val=sst_validation(m,b,a,device)
  improved=val['rmse']<best
  if improved:best=val['rmse'];best_step=stream.total
  coarse=stream.total%BLOCK==0
  if coarse and val['rmse']<coarse_best:coarse_best=val['rmse'];coarse_step=stream.total
  row=dict(steps=stream.total,train_steps=n,schedule_epoch=epoch,lr=lr,best_step=best_step,best_val_rmse=best,coarse_checkpoint=coarse,batch_order_sha256=stream.chain,physics_calls=monitor.phys_calls,rres_calls=monitor.rres_calls,gradient_norm_mean=float(np.mean(monitor.norms)),gradient_norm_max=max(monitor.norms),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.time()-begun,**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()})
  assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
  if stream.total in reference:
   old=reference[stream.total];assert stream.chain==old['batch_order_sha256']
   delta=abs(val['rmse']-old['val_rmse']);row['reference_val_rmse_difference']=delta
   assert delta<2e-6,('trajectory changed',job['name'],stream.total,delta)
  history.append(row)
  cp=dict(model_state=m.state_dict(),config_sha256=cfg['config_sha256'],manifest_sha256=job['manifest_sha256'],best_step=best_step,best_value=best,stats=b.stats)
  if improved:w.checkpoint(d/'best.pt',cp)
  w.save(d/'history.json',history)
  finished=stream.total==MAX_STEPS or stream.total-best_step>=PATIENCE
  if coarse or finished:
   cp.update(optimizer_state=opt.state_dict(),history=history,stream=stream.state(),coarse_best=coarse_best,coarse_step=coarse_step,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),gradient_guard_streak=monitor.streak)
   w.checkpoint(last,cp)
  print(json.dumps({k:row[k] for k in ['steps','best_step','best_val_rmse','val_rmse','lr','train_lambda_physics_effective']}),flush=True)
 cp=torch.load(d/'best.pt',map_location=device,weights_only=False);assert cp['best_step']==best_step;m.load_state_dict(cp['model_state'],strict=True)
 diag=common_diagnostics(m,b,a,device);assert abs(diag['val_rmse']-best)<2e-6
 w.save(d/'best_diagnostics.json',diag);w.save(d/'status.json',dict(status='trained',steps=stream.total,best_step=best_step,best_val_rmse=best,coarse_best_val_rmse=coarse_best,coarse_best_step=coarse_step,evaluations=len(history),checkpoint_roundtrip='PASS',peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,timestamp=w.stamp()))
 b.ds.close();w.log('dense validation training complete '+job['name']+f'; best_step={best_step}, val_RMSE={best:.6f}; Test not evaluated.')

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);o=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3'];lw.install()
 p=json.loads((o.study/'protocol.json').read_text());j=p['jobs'][o.job];d=Path(j['run_dir']);d.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 try:
  w.gpu_identity(d);device=torch.device('cuda:0')
  if j['kind']=='diagnostics':run_diagnostics(o.study,j,p,device)
  else:run_training(o.study,j,p,device)
 except Exception:
  w.save(d/'failure.json',dict(timestamp=w.stamp(),error=traceback.format_exc()));w.log('dense validation failure '+j['name']);raise
if __name__=='__main__':main()
