"""Authorized v3 smoke only: two batches and at most one update per model."""
import os,sys,json,csv,io,subprocess,time,gc,ast,hashlib
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
REPORT=ROOT/'outputs/v3_preflight_20261008';V3=ROOT/'data/subsets/preprocessing_subset_v3'
# Name is set before importing torch or doing any CUDA work.
import setproctitle
setproctitle.setproctitle('yeonwoo')
assert setproctitle.getproctitle()=='yeonwoo'
for key in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR'):
 path=REPORT/'runtime_cache'/key;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)

def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True)
def parse(text):return list(csv.reader(io.StringIO(text)))
def save(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 gate=json.loads((V3/'validation_report.json').read_text());assert all(v=='PASS' for v in gate['gates'].values())
 assert '\nOK\n' in (REPORT/'regression_tests.log').read_text()
 manifest=V3/'ratio_10_lead7_seed42.json';m=json.loads(manifest.read_text())
 for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
 attempt=REPORT/'gpu_smoke_attempt_001';attempt.mkdir(exist_ok=False)
 before=query([]);inventory=parse(query(['--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']))
 apps=parse(query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']))
 busy={r[0].strip() for r in apps if len(r)>=3}
 available=[r for r in inventory if r[1].strip() not in busy and int(r[2])<100 and int(r[3])>30000 and int(r[4])==0]
 if not available:raise RuntimeError('No idle GPU; no other process will be disturbed')
 selected=available[0];gpu=selected[0].strip();uuid=selected[1].strip();os.environ['CUDA_VISIBLE_DEVICES']=gpu
 save(attempt/'gpu_before.json',dict(snapshot=before,inventory=inventory,applications=apps,chosen_index=gpu,chosen_uuid=uuid,pid=os.getpid()))
 import torch
 from mhw import subset_preprocessing as p
 import train_jeju_hilam as t
 torch.set_num_threads(4)
 bundle,metadata=p.load_subset_bundle(manifest)
 # Establish context, then verify the actual NVML process name before model work.
 allocation=torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
 own=[]
 for _ in range(20):
  records=parse(query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']))
  own=[r for r in records if len(r)>=4 and r[1].strip()==str(os.getpid())]
  if own:break
  time.sleep(.1)
 if not own or any(r[2].strip()!='yeonwoo' for r in own):raise RuntimeError(f'GPU title verification failed: {own}')
 save(attempt/'gpu_process_identity.json',dict(pid=os.getpid(),title=setproctitle.getproctitle(),nvml=own,proc_comm=Path('/proc/self/comm').read_text().strip()))
 del allocation
 def ensure_exclusive():
  records=parse(query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']))
  if any(r[0].strip()==uuid and r[1].strip()!=str(os.getpid()) for r in records if len(r)>=3):raise RuntimeError('Another process entered selected GPU; stop our smoke')
 # Execute the real main setup with real model, device and manifest dispatch.
 tree=ast.parse((ROOT/'src/train_jeju_hilam.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');body=[]
 for s in fn.body:
  if isinstance(s,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='config' for x in s.targets):break
  body.append(s)
 fn.name='setup_only';fn.body=body+[ast.Return(ast.Call(ast.Name('locals',ast.Load()),[],[]))]
 code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_main_setup','exec')
 results=[];initial_shared=[];batch_hashes=[]
 class OneUpdateAdamW(torch.optim.AdamW):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);self.updates=0
  def step(self,closure=None):
   if self.updates==0:self.updates+=1;return super().step(closure)
 def gnorm(gs):return sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5
 for arm,mode,lp in [('baseline','none',0.),('physics_only','p0_original',.1)]:
  ensure_exclusive();torch.cuda.reset_peak_memory_stats();env=dict(t.__dict__);exec(code,env)
  argv=['trainer','--subset-manifest',str(manifest),'--lead','7','--seed','42','--device','cuda:0','--physics-mode',mode,'--lambda-physics',str(lp),'--lambda-rres','0','--batch-size','16','--output-root',str(attempt/arm)]
  with patch.object(sys,'argv',argv),patch.object(p,'load_subset_bundle',return_value=(bundle,metadata)):state=env['setup_only']()
  model=state['model'];pa=state['physics_args'];shared=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
  digest=hashlib.sha256()
  for name,v in model.named_parameters():
   if not name.startswith(('we_head.','rres_head.')):digest.update(name.encode());digest.update(v.detach().cpu().numpy().tobytes())
  initial_shared.append(digest.hexdigest())
  counts={part:sum(v.numel() for name,v in model.named_parameters() if (name.startswith(part+'.') if part!='shared' else not name.startswith(('we_head.','rres_head.')))) for part in ('shared','we_head','rres_head')}
  optimizer=OneUpdateAdamW(model.parameters(),lr=state['args'].initial_lr,weight_decay=.1,betas=(.9,.95))
  measurements=[];clipping=[];targets=[];original_loss=t.compute_batch_losses;original_clip=torch.nn.utils.clip_grad_norm_
  def observed_loss(output,batch,*args,**kw):
   vals=original_loss(output,batch,*args,**kw);pred,data,phys,rres,tie,total=vals
   assert pred.shape==batch['y_nodes'].shape and torch.isfinite(pred).all()
   assert all(torch.isfinite(loss) for loss in vals[1:])
   gd=torch.autograd.grad(data,shared,retain_graph=True,allow_unused=True)
   gp=torch.autograd.grad(phys,shared,retain_graph=True,allow_unused=True) if lp else []
   assert all(torch.isfinite(g).all() for g in [*gd,*gp] if g is not None)
   if lp:assert gnorm(gp)>0
   else:assert float(phys)==0
   assert float(rres)==0
   targets.append(batch['target_index'].tolist())
   measurements.append(dict(shape=list(pred.shape),output_min=float(pred.detach().min()),output_max=float(pred.detach().max()),data_loss=float(data.detach()),physics_loss=float(phys.detach()),total_loss=float(total.detach()),data_gradient_norm=gnorm(gd),physics_gradient_norm=gnorm(gp),rhs_scale=None if pa is None else pa.physics_rhs_scale,loss_finite=True,forward=True))
   return vals
  def observed_clip(parameters,max_norm,*args,**kw):
   params=list(parameters);grads=[v.grad for v in params if v.grad is not None]
   assert all(torch.isfinite(g).all() for g in grads)
   before=gnorm(grads);res=original_clip(params,max_norm,*args,**kw);after=gnorm([v.grad for v in params if v.grad is not None]);assert after<=1.00001
   if lp:assert all(v.grad is None for n,v in model.named_parameters() if n.startswith('rres_head.'))
   clipping.append(dict(before=before,after=after,max_norm=max_norm,all_gradients_finite=True));return res
  with patch.object(t,'compute_batch_losses',observed_loss),patch.object(torch.nn.utils,'clip_grad_norm_',observed_clip):
   metrics=t.run_epoch(model,state['train_loader'],bundle.graph,torch.device('cuda:0'),*bundle.stats['target'],optimizer=optimizer,max_batches=2,physics_args=pa,lambda_physics=lp,epoch=5)
  assert len(measurements)==len(clipping)==2 and optimizer.updates==1
  torch.cuda.synchronize();peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
  checkpoint=attempt/arm/'smoke_checkpoint.pt'
  torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),stats=bundle.stats,manifest=str(manifest),manifest_sha256=sha(manifest),optimizer_steps=1,smoke_only=True),checkpoint)
  loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
  restored=t.JejuHiLAM(graph=bundle.graph,input_dim=len(bundle.train.input_feature_names),hidden_dim=256,temporal_hidden_dim=256,processor_layers=3,mlp_hidden_layers=1,prediction_mode='direct',physics_heads=pa is not None,we_scale=state['args'].we_scale,rres_scale=state['args'].rres_scale,we_softplus_shift=state['args'].we_softplus_shift)
  restored.load_state_dict(loaded['model_state'],strict=True)
  assert all(torch.equal(v.detach().cpu(),restored.state_dict()[k]) for k,v in model.state_dict().items())
  assert loaded['stats']==bundle.stats and loaded['manifest_sha256']==sha(manifest)
  result=dict(model=arm,status='PASS',forward_batches=2,backward_batches=2,optimizer_steps=1,batch_size=16,parameters=counts,trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),measurements=measurements,gradient_clipping=clipping,peak_allocated_mib=peak_alloc/2**20,peak_reserved_mib=peak_reserved/2**20,checkpoint=str(checkpoint),checkpoint_roundtrip='PASS_EXACT_STATE',native_metrics=metrics,target_indices=targets)
  results.append(result);batch_hashes.append(hashlib.sha256(json.dumps(targets).encode()).hexdigest());save(attempt/'results.json',dict(status='IN_PROGRESS',models=results))
  print(json.dumps(result),flush=True)
  del model,optimizer,shared,loaded,restored,state;gc.collect();torch.cuda.empty_cache()
 assert initial_shared[0]==initial_shared[1] and batch_hashes[0]==batch_hashes[1]
 final=dict(status='PASS',models=results,manifest=str(manifest),shared_initialization_identical=True,batch_order_identical=True,process_name='yeonwoo',formal_training_started=False)
 save(attempt/'results.json',final);save(REPORT/'gpu_smoke.json',final)
 (attempt/'gpu_after.txt').write_text(query([]));bundle.ds.close()
 stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
 with (ROOT/'docs/research_log.md').open('a') as f:f.write(f'\n\n## {stamp} — v3 제한 GPU smoke 완료\n\n- scripts/smoke_subset_v3_gpu.py 및 outputs/v3_preflight_20261008/gpu_smoke_attempt_001. 유휴 GPU1개에서 Baseline/P0 Physics only 각2batch16 forward/backward 및 각1회optimizer update만 실행했다.\n- 실제 v3 loader/main setup/native run_epoch, finite loss/gradient, physics shared gradient 기여, norm1 clipping, checkpoint 상태 복원을 검증했다. NVML PID process name yeonwoo 확인. 초기 predictor/배치 순서 일치. 본 학습 미실행.\n')
 print('GPU smoke PASS; no formal training started',flush=True)

if __name__=='__main__':main()
