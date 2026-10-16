"""v3 study worker: native model/loss, exact epoch-boundary resume, held-out test phase."""
import os,sys,json,hashlib,math,time,csv,subprocess,traceback,random
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import setproctitle
setproctitle.setproctitle('yeonwoo')
assert setproctitle.getproctitle()=='yeonwoo'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import train_jeju_hilam as t
from mhw import subset_preprocessing as p
from mhw.ocean_stencil import ocean_neighbor_support

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def save(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(path)
def checkpoint(path,obj):
 tmp=path.with_suffix('.tmp.pt');torch.save(obj,tmp);tmp.replace(path)
def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
def log(message):
 import fcntl
 with (ROOT/'docs/research_log.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(f'\n- {stamp()} — v3 study: {message}\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
def state_digest(model,shared=True):
 h=hashlib.sha256()
 for name,v in model.state_dict().items():
  if shared and name.startswith(('we_head.','rres_head.')):continue
  h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def gpu_identity(directory):
 torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
 for _ in range(20):
  rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()))
  own=[r for r in rows if len(r)==4 and r[1].strip()==str(os.getpid())]
  if own:break
  time.sleep(.1)
 assert own and all(r[2].strip()=='yeonwoo' for r in own),own
 save(directory/'gpu_identity.json',dict(pid=os.getpid(),nvml=own,title=setproctitle.getproctitle(),comm=Path('/proc/self/comm').read_text().strip()))

def arguments(job):
 argv=['trainer','--subset-manifest',job['manifest'],'--lead',str(job['lead']),'--seed',str(job['seed']),'--physics-mode','none' if job['arm']=='baseline' else 'p0_original','--lambda-physics','0' if job['arm']=='baseline' else '.1','--lambda-rres','0','--lambda-we-tie','0','--batch-size','16','--epochs','80','--patience','12','--virtual-epoch-steps','478','--no-residual-feature','--device','cuda:0','--precision','fp32','--output-root',job['run_dir']]
 old=sys.argv;sys.argv=argv
 try:return t.parse_args(),argv[1:]
 finally:sys.argv=old

class Stream:
 """Preserve native DataLoader RNG and partially consumed shuffled passes."""
 def __init__(self,loader):
  self.loader=loader;self.iterator=None;self.offset=0;self.total=0;self.chain='';self.first=[];self.start=None
 def new_pass(self):
  self.start=self.loader.generator.get_state().clone();self.iterator=iter(self.loader);self.offset=0
 def take(self,n):
  for _ in range(n):
   if self.iterator is None:self.new_pass()
   try:b=next(self.iterator)
   except StopIteration:self.new_pass();b=next(self.iterator)
   self.offset+=1;self.total+=1
   indices=b['target_index'].tolist()
   self.chain=digest([self.chain,indices])
   if len(self.first)<128:self.first.append(indices)
   yield b
 def state(self):return dict(start=self.start,offset=self.offset,total=self.total,chain=self.chain,first=self.first)
 def restore(self,s):
  self.loader.generator.set_state(s['start']);self.new_pass()
  for _ in range(s['offset']):next(self.iterator)
  self.offset=s['offset'];self.total=s['total'];self.chain=s['chain'];self.first=s['first']

class Monitor:
 def __init__(self,model,physics):self.model=model;self.physics=physics;self.streak=0;self.reset()
 def reset(self):self.norms=[];self.phys_calls=0;self.rres_calls=0;self.batches=0
 def clip(self,parameters,max_norm,*args,**kw):
  values=list(parameters)
  norm=self.original_clip(values,max_norm,error_if_nonfinite=True)
  n=float(norm);self.norms.append(n)
  self.streak=self.streak+1 if n>1e6 else 0
  if self.streak>=100:raise FloatingPointError('100 consecutive preclip gradient norms > 1e6')
  assert all(v.grad is None for name,v in self.model.named_parameters() if name.startswith('rres_head.'))
  return norm
 def losses(self,*a,**kw):
  vals=self.original_loss(*a,**kw)
  if not all(bool(torch.isfinite(v).all()) for v in vals):raise FloatingPointError('Nonfinite prediction/loss')
  assert float(vals[3])==0 and float(vals[4])==0
  if self.physics is None:assert float(vals[2])==0
  self.batches+=1
  return vals
 def phys(self,*a,**kw):self.phys_calls+=1;return self.original_phys(*a,**kw)
 def rres(self,*a,**kw):self.rres_calls+=1;raise AssertionError('Inactive Rres penalty called')
 def __enter__(self):
  self.original_clip=torch.nn.utils.clip_grad_norm_;self.original_loss=t.compute_batch_losses;self.original_phys=t.daily_rollout_physics_loss;self.original_rres=t.rres_penalty
  torch.nn.utils.clip_grad_norm_=self.clip;t.compute_batch_losses=self.losses;t.daily_rollout_physics_loss=self.phys;t.rres_penalty=self.rres
  return self
 def __exit__(self,*a):
  torch.nn.utils.clip_grad_norm_=self.original_clip;t.compute_batch_losses=self.original_loss;t.daily_rollout_physics_loss=self.original_phys;t.rres_penalty=self.original_rres

def setup(job,protocol,device):
 for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected,name
 assert sha(job['manifest'])==job['manifest_sha256']
 args,argv=arguments(job);args.device=str(device)
 t.set_seed(args.seed)
 bundle,m=p.load_subset_bundle(args.subset_manifest)
 assert m['schema']=='subset_normalization_v3'
 physics=t.build_physics_args(args);p.apply_manifest_physics(args,physics,m)
 if physics is not None:assert physics.physics_rhs_scale==m['physics_rhs_scale'] and physics.lambda_rres==0
 model=t.JejuHiLAM(graph=bundle.graph,input_dim=len(bundle.train.input_feature_names),hidden_dim=args.hidden_dim,temporal_hidden_dim=args.temporal_hidden_dim,processor_layers=args.processor_layers,mlp_hidden_layers=args.mlp_hidden_layers,prediction_mode=args.prediction_mode,physics_heads=physics is not None,we_nonneg=True,we_scale=args.we_scale,rres_scale=args.rres_scale,we_softplus_shift=args.we_softplus_shift).to(device)
 mask=bundle.train[0]['mask'];assert int(mask.sum())==661 and int(ocean_neighbor_support(mask).sum())==660
 assert len(bundle.train)=={10:376,25:987,50:1974,100:3995}[job['ratio']]
 assert bundle.stats=={k:tuple(v) for k,v in m['statistics'].items()}
 # No fit/cache lookup is performed: the authoritative v3 loader provides every split.
 counts={key:sum(v.numel() for name,v in model.named_parameters() if v.requires_grad and (not name.startswith(('we_head.','rres_head.')) if key=='shared' else name.startswith(key+'.'))) for key in ['shared','we_head','rres_head']}
 fingerprints={}
 for split in ['train','val','test']:
  ds=getattr(bundle,split);h=hashlib.sha256()
  for i in sorted(set([0,len(ds)//2,len(ds)-1])):
   sample=ds[i]
   for k in ['x_nodes','y_nodes','target_index']:
    h.update(k.encode());h.update(sample[k].numpy().tobytes())
  fingerprints[split]=h.hexdigest()
 cfg={'job':job,'protocol_sha256':digest(protocol),'argv':argv,'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'manifest_sha256':sha(job['manifest']),'stats':m['statistics'],'rhs_scale':m['physics_rhs_scale'],'applied_rhs_scale':None if physics is None else physics.physics_rhs_scale,'physics_stencil':m['physics_stencil'],'shared_initial_sha256':state_digest(model),'parameters':counts,'trainable_parameters':sum(counts.values()),'split_sha256':{k:digest(getattr(bundle,k+'_indices').tolist()) for k in ['train','val','test']},'tensor_fingerprints':fingerprints,'evaluation_mask_cells':661,'physics_mask_cells':660,'selection':'validation SST RMSE only','test_evaluation':'deferred until all 48 training runs complete'}
 cfg['config_sha256']=digest(cfg)
 return args,bundle,model,physics,cfg

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
 if phase=='evaluate':
  state=torch.load(d/'best.pt',map_location=device,weights_only=False)
  assert state['config_sha256']==config['config_sha256'];model.load_state_dict(state['model_state'])
  test=t.make_loader(bundle.test,args,False)
  with torch.no_grad():metrics=t.run_epoch(model,test,epoch=5,**common)
  fields=t.collect_predictions(model,test,**{k:v for k,v in common.items() if k!='lambda_physics'})
  assert np.array_equal(fields['target_index'],bundle.test_indices+29+job['lead'])
  assert np.isfinite(fields['pred_raw']).all()
  fields['target_dates']=bundle.ds.time.values[fields['target_index']]
  sea=fields['mask'][0]>0;errors=fields['pred_raw'][:,sea].astype(np.float64)-fields['target_raw'][:,sea]
  fields['daily_mse']=np.mean(errors**2,axis=1);fields['daily_mae']=np.mean(np.abs(errors),axis=1)
  np.savez_compressed(d/'test_fields.npz',**fields)
  result=t.compute_forecast_metrics(fields['pred_raw'],fields['target_raw'],fields['mask'])
  result.update({'test_'+k:v for k,v in metrics.items()});result.update(job=job,best_epoch=state['best_epoch'],best_val_rmse=state['best_value'],config_sha256=config['config_sha256'])
  save(d/'test_metrics.json',result);save(d/'evaluation_status.json',dict(status='completed',timestamp=stamp()));log(f'평가 완료 {job["name"]}; test_metrics.json/test_fields.npz 저장.');bundle.ds.close();return
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
  # Shared predictor physics contribution, with no RNG/data-stream/optimizer change.
  if physics is not None:
   batch=next(iter(t.make_loader(bundle.train,args,False)));model.train();out=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
   vals=t.compute_batch_losses(out,batch,bundle.graph,*bundle.stats['target'],physics,.1,model)
   shared=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
   grads=torch.autograd.grad(vals[2],shared,allow_unused=True);norm=math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None));assert norm>0 and math.isfinite(norm)
   save(d/'initial_physics_gradient.json',dict(shared_physics_gradient_norm=norm,physics_loss=float(vals[2].detach()),predictor_contribution=True));del grads,shared,out,vals,batch
  log(f'정식 학습 시작 {job["name"]}, PID {os.getpid()}, manifest={job["manifest"]}, λ={args.lambda_physics}, Rres=0.')
 begun=time.time();torch.cuda.reset_peak_memory_stats()
 for epoch in range(start_epoch,args.epochs+1):
  if bad>=args.patience:break
  monitor.reset();lr=t.set_epoch_lr(optimizer,args,epoch)
  save(d/'status.json',dict(status='training',epoch=epoch,pid=os.getpid(),steps=stream.total,best_epoch=best_epoch,timestamp=stamp()))
  with monitor:
   train=t.run_epoch(model,stream.take(478),optimizer=optimizer,epoch=epoch,**common)
   train_calls=monitor.phys_calls
   with torch.no_grad():validation=t.run_epoch(model,val,epoch=epoch,**common)
  assert monitor.rres_calls==0
  assert (monitor.phys_calls==0) if physics is None else (train_calls==478)
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
 log(f'학습 완료 {job["name"]}, best_epoch={best_epoch}, val_RMSE={best:.6f}, epochs={len(history)}; Test는 전체48회 학습 완료 후 평가.');bundle.ds.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);ap.add_argument('--phase',choices=['train','evaluate'],default='train');opt=ap.parse_args()
 try:worker(opt.study,opt.job,opt.phase)
 except Exception:
  detail=traceback.format_exc();print(detail,flush=True)
  job=json.loads((opt.study/'protocol.json').read_text())['jobs'][opt.job];d=Path(job['run_dir']);d.mkdir(parents=True,exist_ok=True)
  save(d/('failure_'+opt.phase+'_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),dict(error=detail,phase=opt.phase,timestamp=stamp(),pid=os.getpid()))
  log(f'실행 실패 {job["name"]} phase={opt.phase}; 오류 로그 보존.');raise
