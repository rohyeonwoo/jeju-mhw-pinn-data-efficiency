"""Matched all-head four-arm ablation; native losses, no Test phase."""
import train_v3_study as w
from train_v3_study import *
_original_arguments=w.arguments
_original_setup=w.setup

def arguments(job):
 # Every arm has exactly the same native predictor/we/rres architecture.
 args,argv=_original_arguments(dict(job,arm='p0'))
 args.lambda_physics=float(job['lambda_physics'])
 argv[argv.index('--lambda-physics')+1]=str(args.lambda_physics)
 # Original setup validates zero here; set actual Rres weight after construction.
 return args,argv

def setup(job,protocol,device):
 w.arguments=arguments
 args,bundle,model,physics,cfg=_original_setup(job,protocol,device)
 args.lambda_rres=physics.lambda_rres=float(job['lambda_rres'])
 physics.lambda_physics=args.lambda_physics
 assert args.zero_rres_physics and args.lambda_we_tie==0 and args.rres_penalty_normalize=='rhs'
 cfg['settings']['lambda_rres']=args.lambda_rres
 cfg['argv'][cfg['argv'].index('--lambda-rres')+1]=str(args.lambda_rres)
 cfg['architecture']='identical native predictor + we_head + rres_head in all four arms'
 cfg['test_evaluation']='disabled; validation selection only'
 cfg['optimization']='native data + lambda_effective*physics + lambda_rres*rres; no gradient surgery'
 cfg['selection']='minimum validation SST RMSE at original 478-update boundaries'
 ref=json.loads((Path(job['reference_p0_run'])/'config.json').read_text())
 for key in ['shared_initial_sha256','stats','manifest_sha256','split_sha256','tensor_fingerprints','parameters','trainable_parameters','rhs_scale','applied_rhs_scale']:
  assert cfg[key]==ref[key],key
 for k,v in cfg['settings'].items():
  if k=='device':assert v==str(device)
  elif k not in ['lambda_rres','lambda_physics','output_root']:assert v==ref['settings'][k],k
 cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg)
 return args,bundle,model,physics,cfg

class Monitor(w.Monitor):
 def clip(self,parameters,max_norm,*args,**kw):
  norm=self.original_clip(list(parameters),max_norm,error_if_nonfinite=True)
  n=float(norm);self.norms.append(n);self.streak=self.streak+1 if n>1e6 else 0
  if self.streak>=100:raise FloatingPointError('100 consecutive preclip gradient norms > 1e6')
  for prefix,weight in [('we_head.',self.physics.lambda_physics),('rres_head.',self.physics.lambda_rres)]:
   if weight==0:assert all(v.grad is None for name,v in self.model.named_parameters() if name.startswith(prefix)),prefix
  return norm
 def losses(self,*a,**kw):
  vals=self.original_loss(*a,**kw)
  assert all(bool(torch.isfinite(v).all()) for v in vals)
  assert float(vals[4])==0
  if self.physics.lambda_physics==0:assert float(vals[2])==0
  if self.physics.lambda_rres==0:assert float(vals[3])==0
  self.batches+=1
  return vals
 def phys(self,*a,**kw):
  assert self.physics.lambda_physics!=0,'Inactive physics invoked'
  self.phys_calls+=1;return self.original_phys(*a,**kw)
 def rres(self,*a,**kw):
  assert self.physics.lambda_rres!=0,'Inactive auxiliary invoked'
  self.rres_calls+=1;return self.original_rres(*a,**kw)

def initial_diagnostic(model,bundle,args,physics,device,d):
 batch=next(iter(t.make_loader(bundle.train,args,False)));model.train()
 out=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
 effective=args.lambda_physics*.2
 vals=t.compute_batch_losses(out,batch,bundle.graph,*bundle.stats['target'],physics,effective,model)
 shared=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
 rows={};vectors={}
 for name,index,weight in [('data',1,1.),('physics',2,effective),('auxiliary',3,args.lambda_rres)]:
  if weight==0:rows[name]=dict(active=False);continue
  gs=torch.autograd.grad(vals[index],shared,allow_unused=True,retain_graph=True)
  v=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(shared,gs)]).double()
  norm=float(v.norm());assert math.isfinite(norm) and norm>0
  vectors[name]=v;rows[name]=dict(active=True,loss=float(vals[index].detach()),weight=weight,shared_gradient_norm=norm)
 for name in ['physics','auxiliary']:
  if name in vectors:
   a,b=vectors['data'],vectors[name];rows[name].update(cosine_to_data=float(a.dot(b)/(a.norm()*b.norm())),weighted_gradient_ratio=rows[name]['weight']*rows[name]['shared_gradient_norm']/rows['data']['shared_gradient_norm'])
 save(d/'initial_objective_gradients.json',dict(scope='first train batch, epoch1 effective lambda; no optimizer update',objectives=rows))

def worker(study,job_id,phase):
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
 assert phase=='train', 'Test is disabled'
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
  log(f'Ablation start {job["name"]}; physics={args.lambda_physics}, auxiliary={args.lambda_rres}, identical heads; Validation only.')
 begun=time.time();torch.cuda.reset_peak_memory_stats()
 for epoch in range(start_epoch,args.epochs+1):
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
 save(d/'status.json',dict(status='trained',epochs=len(history),steps=stream.total,best_epoch=best_epoch,best_val_rmse=best,early_stopping=bad>=args.patience,checkpoint_roundtrip='PASS',config_sha256=config['config_sha256'],manifest_sha256=job['manifest_sha256'],peak_reserved_mib=max(r['peak_reserved_mib'] for r in history),timestamp=stamp()))
 log(f'학습 완료 {job["name"]}, best_epoch={best_epoch}, val_RMSE={best:.6f}, epochs={len(history)}; Ablation Validation only; Test disabled.');bundle.ds.close()


if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);opt=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3']
 try:worker(opt.study,opt.job,'train')
 except Exception:
  job=json.loads((opt.study/'protocol.json').read_text())['jobs'][opt.job];d=Path(job['run_dir']);d.mkdir(parents=True,exist_ok=True)
  save(d/'failure.json',dict(error=traceback.format_exc(),timestamp=stamp()));raise
