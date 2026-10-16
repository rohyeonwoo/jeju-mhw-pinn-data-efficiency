"""CPU preflight: finite passes, native loss gradients, schedule, real loop resume."""
from mixed_budget_common import *
import ast,math,copy,contextlib,io
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
import worker as W

def doc(name,text): (STUDY/name).write_text(text+'\n')
def test_loop_resume():
 class Toy(Dataset):
  def __len__(self):return 17
  def __getitem__(self,i):
   x=torch.tensor([i/17.,(i%3)/3.]);y=torch.tensor([.3*x[0]+.1])
   return dict(start_index=torch.tensor(i),x_nodes=x,y_nodes=y,y_raw_nodes=y,tm_norm_nodes=y*0)
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1))
  def forward(self,x,base_nodes=None,return_aux=False):return self.net(x)
 def make(name):
  W.t.set_seed(42);model=Model();args=SimpleNamespace(seed=42,batch_size=16,num_workers=0,epochs=80,virtual_epoch_steps=None,physics_warmup_epochs=5,initial_lr=2e-5,lr=2e-4,warmup_epochs=5,weight_decay=.1,lambda_physics=0.)
  b=SimpleNamespace(train=Toy(),val=Toy(),graph=None,stats={'target':(0.,1.)},train_indices=list(range(17)))
  d=STUDY/'cpu_tests'/name;cfg=dict(config_sha256='cpu-loop-regression',manifest_sha256='synthetic',checkpoint_dir=str(d/'checkpoints'))
  return args,b,model,cfg,d
 with (STUDY/'cpu_tests/loop.log').open('w') as f,contextlib.redirect_stdout(f):
  a,b,model,c,d=make('uninterrupted');h1=W.run_training(a,b,model,None,c,d,torch.device('cpu'),invariance=False)
  a,b,model,c,d=make('resumed');W.run_training(a,b,model,None,c,d,torch.device('cpu'),stop_after_epoch=3,invariance=False)
  a,b,model,c,d=make('resumed');h2=W.run_training(a,b,model,None,c,d,torch.device('cpu'),invariance=False)
 c1=torch.load(STUDY/'cpu_tests/uninterrupted/checkpoints/last.pt',map_location='cpu',weights_only=False);c2=torch.load(STUDY/'cpu_tests/resumed/checkpoints/last.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,c2['model_state'][k]) for k,v in c1['model_state'].items())
 for x,y in zip(h1,h2):
  for k in ['sample_order_sha256','train_total_loss','train_rmse','val_rmse','lr','best_epoch','post_warmup_best_epoch']:assert x[k]==y[k],k
 assert len(h1)==80 and h1[-1]['steps']==160 and c1['post_best_epoch']>=5
 # Explicit non-finite loss guard; this is the same Monitor used by real workers.
 guard=W.Monitor(model,None);guard.original_loss=lambda *a: (torch.tensor(float('nan')),)*6
 try:guard.losses();raise AssertionError('NaN not rejected')
 except FloatingPointError:pass
 return dict(status='PASS',epochs=80,updates=160,resume_after_epoch=3,weights_bitwise_equal=True,history_and_sampler_equal=True,both_checkpoint_reload=True,nonfinite_guard=True)

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 for folder in ['logs','launches','cpu_tests','figures','checkpoints','smoke']: (STUDY/folder).mkdir(parents=True,exist_ok=True)
 assert not (STUDY/'experiment_protocol.json').exists(),'Do not overwrite frozen actual protocol'
 torch.set_num_threads(1)
 prev=json.loads((OLD_STUDY/'experiment_protocol.json').read_text());verify_code(prev)
 jobs=[];sampling=[];boundaries=[];schedules=[];sets={}
 class Indices(Dataset):
  def __init__(self,ids):self.ids=ids
  def __len__(self):return len(self.ids)
  def __getitem__(self,i):return dict(start_index=torch.tensor(self.ids[i]))
 for j in prev['jobs']:
  m=json.loads(Path(j['manifest']).read_text());assert sha(j['manifest'])==j['manifest_sha256'];n=len(m['train_start_indices']);upe=math.ceil(n/16)
  c0=json.loads((OLD_STUDY/'runs'/f'{j["label"]}_r10_lead7_seed42'/'config.json').read_text());lam=c0['settings']['lambda_physics'];assert lam==(0 if j['model']=='B0' else .1)
  job=dict(j,run_dir=str(STUDY/'runs'/j['name']),lambda_physics=lam,n_train=n,updates_per_epoch=upe,total_updates=80*upe,warmup_steps=5*upe)
  jobs.append(job)
  if j['model']!='B0':continue
  key=(j['ratio'],j['lead'],j['seed']);sets[key]=set(m['train_start_indices'])
  s=np.asarray(m['train_start_indices']);assert len(s)==len(set(s.tolist()));train_end=min(m['val_start_indices'])+29+j['lead']-1
  assert np.all(s+29+j['lead']<=train_end) and np.all(s+29+j['lead']-1<=train_end)
  assert np.array_equal(np.unique(s[:,None]+np.arange(30)),m['input_time_indices'])
  boundaries.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],last_train_target=int(max(s)+29+j['lead']),train_end=train_end,first_val_target=min(m['val_start_indices'])+29+j['lead'],status='PASS'))
  a,_=W.arguments(job);orders=[]
  for arm in range(4):
   loader=W.t.make_loader(Indices(m['train_start_indices']),a,True);checks=[]
   for epoch in range(1,81):
    run=W.CheckedEpoch(loader,m['train_start_indices'])
    for batch in run:pass
    checks.append(run.evidence()['sample_order_sha256'])
   assert len(set(checks))==80,'epoch shuffle did not change'
   orders.append(checks)
  assert all(x==orders[0] for x in orders)
  # Reproducibility across fresh generator / same seed.
  retry=W.CheckedEpoch(W.t.make_loader(Indices(m['train_start_indices']),a,True),m['train_start_indices'])
  for batch in retry:pass
  assert retry.evidence()['sample_order_sha256']==orders[0][0]
  expected={10:376,25:987,50:1974,100:3995}[j['ratio']]
  sampling.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],samples=n,expected_samples=expected,matches_expected=n==expected,updates_per_epoch=upe,total_updates=80*upe,last_batch_samples=(n-1)%16+1,epochs_checked=80,models_checked=4,status='PASS',sample_order_sha256=orders[0]))
  dummy=torch.nn.Linear(1,1);opt=torch.optim.AdamW(dummy.parameters(),lr=a.initial_lr)
  for epoch in range(1,81):
   lr=W.t.set_epoch_lr(opt,a,epoch);factor=W.t.physics_lambda_factor(epoch,5)
   expected_lr=2e-5+(epoch/5)*(2e-4-2e-5) if epoch<=5 else .5*2e-4*(1+math.cos(math.pi*(epoch-5)/75))
   assert abs(lr-expected_lr)<1e-15 and factor==min(epoch/5,1)
   schedules.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],epoch=epoch,lr=lr,physics_factor=factor,physics_lambda=.1*factor,epoch_end_updates=epoch*upe))
  print('sampling/schedules PASS',key,n,upe,flush=True)
 for (ratio,lead,seed),values in sets.items():
  for lower in [r for r in [10,25,50,100] if r<ratio]:assert sets[lower,lead,seed]<=values
 # Edge sizes include smaller than batch, exact multiple and incomplete final batches.
 for n in [1,15,16,17,32,33]:
  a=SimpleNamespace(seed=7,batch_size=16,num_workers=0);run=W.CheckedEpoch(W.t.make_loader(Indices(list(range(n))),a,True),list(range(n)))
  for batch in run:pass
  assert run.evidence()['train_samples']==n
 save(STUDY/'sampling_audit.json',sampling);save(STUDY/'schedule_audit.json',schedules);save(STUDY/'boundary_audit.json',boundaries)
 p={**prev,'version':'actual_epoch_v1','created_at':stamp(),'jobs':jobs,'max_processes_per_gpu':2,'allowed_physical_gpus':[7,8,9],
 'epoch_type':'actual','epochs':80,'early_stopping':False,'physics_warmup_epochs':5,'lr_warmup_epochs':5,'fixed_updates':None,
 'max_updates':'80*ceil(n_train/16)','updates_per_validation':'ceil(n_train/16)','training_policy':'Training-only Oracle Physics Supervision',
 'selection':'minimum full Validation SST RMSE across 80 actual epochs','post_warmup_selection':'epoch >= 5 (first epoch at full weight)',
 'expansion_gates':['CPU','4 smokes','4 representatives integrity','remaining92','Validation report only'],
 'test_evaluation':'disabled; plan only after completion','reuse_previous_runs':False,'old_protocol_sha256':sha(OLD_STUDY/'experiment_protocol.json'),
 'code_sha256':{**prev['code_sha256'],**{str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')}}}
 for key in ['optimizer','MHW','normalization','lambda_physics','kd','lambda_rres','lambda_we_tie']:assert p[key]==prev[key]
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
 W.old.install();entry=[]
 for L in [7,14]:
  j=next(j for j in jobs if j['ratio']==10 and j['lead']==L and j['seed']==42 and j['model']=='A')
  a,b,model,pa,c=W.setup(j,p,torch.device('cpu'));a.batch_size=2
  batch=next(iter(W.t.make_loader(b.train,a,False)));model.eval();out=model(batch['x_nodes'],base_nodes=batch['tm_norm_nodes'],return_aux=True)
  pars=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
  for arm in ['B0','B1','A','B']:
   physics=copy.copy(pa);physics.mixed_variant=arm;physics.lambda_physics=0 if arm=='B0' else .1
   factor=W.t.physics_lambda_factor(1,physics.physics_warmup_epochs);effective=physics.lambda_physics*factor
   vals=W.t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],None if arm=='B0' else physics,effective,model)
   assert all(torch.isfinite(v).all() for v in vals)
   gd=torch.autograd.grad(vals[1],pars,retain_graph=True,allow_unused=True);nd=sum(float(g.double().square().sum()) for g in gd if g is not None)**.5
   gp=[] if arm=='B0' else torch.autograd.grad(vals[2],pars,retain_graph=True,allow_unused=True);ng=sum(float(g.double().square().sum()) for g in gp if g is not None)**.5
   assert arm=='B0' or ng>0
   assert torch.allclose(vals[-1],vals[1]+effective*vals[2])
   entry.append(dict(lead=L,model=arm,epoch=1,lambda_effective=effective,data_loss=float(vals[1].detach()),physics_loss=float(vals[2].detach()),physics_predictor_gradient=ng,weighted_ratio=effective*ng/nd,status='PASS'))
  vb=next(iter(W.t.make_loader(b.val,a,False)));assert not any(k.startswith('mixed_') or k.endswith('_seq') for k in vb)
  b.ds.close();del out,vals,gd,gp,model,b,batch,vb
  print('native CPU physics gradient PASS',L,flush=True)
 save(STUDY/'physics_entrypoint_cpu.json',entry)
 loop=test_loop_resume();save(STUDY/'loop_regression.json',loop);print('80 epoch native loop resume PASS',flush=True)
 # Preserve old outputs, including incomplete checkpoints, and verify scientific inputs.
 protected=json.loads((OLD_STUDY/'protected_input_hashes.json').read_text())
 for f in OLD_STUDY.rglob('*'):
  if f.is_file() and 'runtime' not in f.parts:protected.setdefault(str(f),sha(f))
 mismatches=[]
 for path,h in protected.items():
  if sha(path)!=h:mismatches.append(path)
 assert not mismatches,mismatches
 save(STUDY/'protected_input_hashes.json',protected);save(STUDY/'protected_integrity_preflight.json',dict(status='PASS',files=len(protected),timestamp=stamp()))
 save(STUDY/'experiment_protocol.json',p);verify_code(p)
 with (STUDY/'run_manifest.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(jobs[0]));writer.writeheader();writer.writerows(jobs)
 save(STUDY/'run_state.json',dict(status='PENDING',completed=0,runs=[dict(job=j['id'],name=j['name'],status='PENDING') for j in jobs],timestamp=stamp()))
 doc('protocol.md','''# Actual Epoch Oracle96 protocol\n\n80 actual epochs; one shuffled no-replacement pass per epoch; batch16/drop_last=False. Early stopping OFF. LR initial2e-5, peak2e-4, five actual-epoch linear warmup then original cosine to epoch80 LR=0. Physics lambda is the original Oracle96 config (Baseline0, P0/NoAdv/Adv0.1) times min(epoch/5,1). NoAdv/Adv refer to absence/presence of horizontal advection, not adversarial losses. No auxiliary/adversarial schedule is added.\n\nOfficial checkpoint: minimum Validation RMSE across epochs1–80. Diagnostic post-warmup best: epoch>=5 (epoch5 already uses full lambda on every update). Validation every actual epoch, no physics or forcing in selection; extra validation calls at completion only verify checkpoints and forcing independence. Test execution disabled.\n\nSame initialization, subset, statistics, predictor, physics terms, optimizer AdamW betas(.9,.95), weight_decay.1 and clipnorm1 as frozen Oracle96. 96 runs initialized from seed; no old checkpoint reuse. Each GPU7/8/9 max two workers, 14GiB allocator budget each plus4GiB headroom; pause new launches if insufficient memory. Old/new locks prevent overlap.\n\nCompared with old: actual epochs replace478-update cycles; physics warmup old0 -> new5 actual epochs; LR formula was epoch-based, not a per-update scheduler, and is now called once per actual epoch. Epoch80 LR0 remains intentional per unchanged formula; total updates count optimizer calls. Training RMSE is online batch aggregation.\n\nSST-label efficiency with training-only oracle, not equal total-information efficiency. Same prior Test has informed earlier research decisions; future evaluation is exploratory. Test will require a separate plan/authorization after96 completion.''')
 doc('sampling_audit.md','# Sampling PASS\n\n24 manifests × 4 model-order replicas × 80 epochs: exact once, no missing/duplicate, last batch retained, reproducible shuffle. Synthetic sizes1/15/16/17/32/33 PASS. Details sampling_audit.json.\n\n10%376→24→1920;25%987→62→4960;50%1974→124→9920;100%3995→250→20000. 100% is blocked subset, not Full train.\n\nImplementation: scripts/oracle96_actual/worker.py CheckedEpoch; native src/train_jeju_hilam.py:215 DataLoader; scripts/train_v3_study.py:53 old cycling iterator is not called by new training loop.')
 doc('physics_warmup_audit.md','# Physics warmup PASS\n\nOriginal src/train_hybrid_jeju_forecast.py:524 factor=min(epoch/warmup,1); src/train_jeju_hilam.py:323 applies lambda once, :288 adds weighted loss once. Original Oracle96 worker:79 overrides warmup0. New config5 gives factors0.2/0.4/0.6/0.8/1; effective lambda0.02/0.04/0.06/0.08/0.1. Epoch6–80 remains0.1. Baseline0 skips physics. CPU real predictor/loss gradient checks on leads7/14 PASS; physics_entrypoint_cpu.json. NoAdv/Adv invoke same effective lambda, no other warmup. GPU confirmation PENDING until smoke gate.')
 doc('lr_scheduler_audit.md','# LR scheduler PASS\n\nOriginal src/train_jeju_hilam.py:230 formula unchanged. Called once BEFORE actual-epoch optimizer loop; no scheduler.step duplicate. Initial2e-5→epoch1 5.6e-5→epoch5 2e-4→epoch80 0. Ratio warmup/total step calls:10%=120/1920,25%=310/4960,50%=620/9920,100%=1250/20000. schedule_audit.json validates all80 epochs×24 manifests. Original fixed478/38240 paths remain preserved but are not called by worker.run_training. New args virtual_epoch_steps=None. Last epoch has zero LR; no reinterpretation as nonzero parameter updates.')
 doc('preflight.md','# CPU preflight PASS\n\nSampling24 conditions×80×4 PASS; nested/boundary manifests PASS; physics gradient both leads PASS; actual native run_epoch training loop80 epochs PASS; interrupted afterepoch3 / fresh-process-equivalent restore reproduces weights bitwise and histories/order exactly; best and post-warmup reload PASS; NaN rejection PASS. Protected input/old-output hashes PASS. New worker has no evaluate phase and controller never executes Test.\n\nGPU smoke and representative actual data training are still PENDING; not claimed as PASS. See preflight.json and regression artifacts. Original science source hashes preserved; changed protocol code isolated in scripts/oracle96_actual/.')
 save(STUDY/'preflight.json',dict(status='PASS',sampling='PASS',loop_resume=loop,physics_entrypoint='PASS',boundary='PASS',protected_files=len(protected),protocol_sha256=digest(p),gpu_smoke='PENDING',timestamp=stamp()))
 log('新 Actual Epoch専用 scripts/oracle96_actual 追加: v8_actual_epoch_oracle96_20261011 별도96조건, early stopping OFF, actual80, LR/physics warmup5, old λ=.1 보존. CPU sampling/경계/실제loss gradient/80epoch resume/보호hash PASS. 기존 v8 및 Test 평가 미변경. GPU smoke 후 대표4개, 검증 PASS 후92개 자동확대 승인 반영.')
 print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
