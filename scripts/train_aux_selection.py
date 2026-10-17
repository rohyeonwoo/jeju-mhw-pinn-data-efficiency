"""Bounded pilot / exact continuation; unchanged native model and all-head objectives."""
import train_v3_ablation as a
from train_v3_study import *
setup=a.setup
Monitor=a.Monitor
initial_diagnostic=a.initial_diagnostic

def worker(study,job_id,stage):
 protocol=json.loads((study/'protocol.json').read_text());job=protocol['jobs'][job_id];d=Path(job['run_dir']);d.mkdir(parents=True,exist_ok=True)
 for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
  dest=study/'runtime'/str(job_id)/key;dest.mkdir(parents=True,exist_ok=True);os.environ[key]=str(dest)
 torch.set_num_threads(2);device=torch.device('cuda:0');gpu_identity(d)
 args,bundle,model,physics,config=setup(job,protocol,device)
 path=d/'config.json'
 if path.exists():assert json.loads(path.read_text())==config,'Config drift'
 else:save(path,config)
 common=dict(graph=bundle.graph,device=device,target_mean=bundle.stats['target'][0],target_std=bundle.stats['target'][1],precision='fp32',physics_args=physics,lambda_physics=args.lambda_physics)
 val=t.make_loader(bundle.val,args,False)
 assert stage in ('pilot','final'), 'Test is disabled'
 stop_epoch=protocol['pilot_epochs'] if stage=='pilot' else args.epochs
 loader=t.make_loader(bundle.train,args,True);stream=Stream(loader)
 optimizer=torch.optim.AdamW(model.parameters(),lr=args.initial_lr,weight_decay=args.weight_decay,betas=(.9,.95))
 history=[];best=float('inf');best_epoch=0;bad=0;start_epoch=1;monitor=Monitor(model,physics)
 last=d/'last.pt'
 if last.exists():
  cp=torch.load(last,map_location=device,weights_only=False);assert cp['config_sha256']==config['config_sha256']
  model.load_state_dict(cp['model_state']);optimizer.load_state_dict(cp['optimizer_state']);stream.restore({**cp['stream'],'start':cp['stream']['start'].cpu()})
  history=cp['history'];best=cp['best_value'];best_epoch=cp['best_epoch'];bad=cp['bad_epochs'];start_epoch=cp['epoch']+1;monitor.streak=cp['gradient_guard_streak']
  torch.set_rng_state(cp['rng_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in cp['rng_cuda']]);np.random.set_state(cp['rng_numpy']);random.setstate(cp['rng_python'])
  del cp;log(f'epoch 경계 checkpoint 복구 {job["name"]}, next_epoch={start_epoch}, 설정/manifest 해시 확인.')
 else:
  initial_diagnostic(model,bundle,args,physics,device,d)
  log(f'Auxiliary selection start {job["name"]}; physics={args.lambda_physics}, auxiliary={args.lambda_rres}, identical heads; Validation only.')
 begun=time.time();torch.cuda.reset_peak_memory_stats()
 for epoch in range(start_epoch,stop_epoch+1):
  if bad>=args.patience:break
  monitor.reset();lr=t.set_epoch_lr(optimizer,args,epoch)
  save(d/'status.json',dict(status='training',epoch=epoch,pid=os.getpid(),steps=stream.total,best_epoch=best_epoch,timestamp=stamp()))
  with monitor:
   train=t.run_epoch(model,stream.take(478),optimizer=optimizer,epoch=epoch,**common)
   train_calls=monitor.phys_calls
   with torch.no_grad():validation=t.run_epoch(model,val,epoch=epoch,**common)
  assert monitor.rres_calls==(monitor.batches if args.lambda_rres else 0)
  assert monitor.phys_calls==(monitor.batches if args.lambda_physics else 0)
  assert train_calls==(478 if args.lambda_physics else 0)
  refhist=json.loads((Path(job['reference_p0_run'])/'history.json').read_text())
  if epoch<=len(refhist):assert stream.chain==refhist[epoch-1]['batch_order_sha256'],'Batch order mismatch'
  for values in [train,validation]:
   for k in ['loss','data_loss','physics_loss','total_loss','rmse']:assert math.isfinite(values[k]),(k,values[k])
  assert len(monitor.norms)==478
  if validation['rmse']<best:best=validation['rmse'];best_epoch=epoch;bad=0;improved=True
  else:bad+=1;improved=False
  row=dict(epoch=epoch,steps=stream.total,lr=lr,best_epoch=best_epoch,best_val_rmse=best,early_stopping=bad>=args.patience,gradient_norm_mean=float(np.mean(monitor.norms)),gradient_norm_max=max(monitor.norms),gradient_clipped_batches=sum(n>1 for n in monitor.norms),gradient_clip_fraction=float(np.mean(np.array(monitor.norms)>1)),batch_order_sha256=stream.chain,first128_batches_sha256=digest(stream.first),physics_calls=monitor.phys_calls,rres_calls=monitor.rres_calls,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_seconds=time.time()-begun,**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in validation.items()})
  history.append(row)
  cp=dict(model_state=model.state_dict(),config_sha256=config['config_sha256'],manifest_sha256=job['manifest_sha256'],best_epoch=best_epoch,best_value=best,stats=bundle.stats)
  if improved:checkpoint(d/'best.pt',cp)
  cp.update(epoch=epoch,optimizer_state=optimizer.state_dict(),bad_epochs=bad,history=history,stream=stream.state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),gradient_guard_streak=monitor.streak)
  checkpoint(last,cp);t.save_epoch_history(d/'history.csv',history);save(d/'history.json',history)
  save(d/'status.json',dict(status='training',epoch=epoch,steps=stream.total,best_epoch=best_epoch,best_val_rmse=best,pid=os.getpid(),peak_reserved_mib=row['peak_reserved_mib'],timestamp=stamp()))
  print(json.dumps(row),flush=True)
 state=torch.load(d/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state'],strict=True)
 assert state['config_sha256']==config['config_sha256'] and all(torch.isfinite(v).all() for v in state['model_state'].values())
 save(d/'status.json',dict(status='pilot_complete' if stage=='pilot' else 'trained',stage=stage,epochs=len(history),steps=stream.total,best_epoch=best_epoch,best_val_rmse=best,early_stopping=bad>=args.patience,checkpoint_roundtrip='PASS',config_sha256=config['config_sha256'],manifest_sha256=job['manifest_sha256'],peak_reserved_mib=max(r['peak_reserved_mib'] for r in history),timestamp=stamp()))
 log(f'학습 완료 {job["name"]}, best_epoch={best_epoch}, val_RMSE={best:.6f}, epochs={len(history)}; Auxiliary selection stage={stage}; Validation only; Test disabled.');bundle.ds.close()



if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);ap.add_argument('--stage',choices=['pilot','final'],required=True);o=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3']
 try:worker(o.study,o.job,o.stage)
 except Exception:
  j=json.loads((o.study/'protocol.json').read_text())['jobs'][o.job];d=Path(j['run_dir']);d.mkdir(parents=True,exist_ok=True)
  save(d/('failure_'+o.stage+'.json'),dict(error=traceback.format_exc(),timestamp=stamp()));raise
