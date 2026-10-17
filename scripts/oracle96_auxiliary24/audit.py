"""Pretraining audit only. Never launches a training queue or evaluates Test."""
import os,sys,json,time,hashlib,csv,copy,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/ceph_data/yeonwoo/MHW_subset');OUT=ROOT/'outputs/v8_actual_epoch_auxiliary24_20261015'
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 d=OUT/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import setproctitle
setproctitle.setproctitle('yeonwoo');assert setproctitle.getproctitle()=='yeonwoo'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'scripts/oracle96_actual'))
import worker as W
# Imported legacy utilities use the old STUDY for path constants; redirect caches back.
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:os.environ[k]=str(OUT/'runtime'/k)
import torch
import numpy as np
TRAIN=ROOT/'outputs/v8_actual_epoch_oracle96_20261011';OLD=ROOT/'outputs/aux_weight_selection_20261009'
def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
def save(name,obj):W.save(OUT/name,obj)
def record(s):
 import fcntl
 with (ROOT/'docs/research_log.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n- '+stamp()+' — Auxiliary24 preflight: '+s+'\n');f.flush()
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
 torch.set_num_threads(1)
 p=json.loads((TRAIN/'experiment_protocol.json').read_text());W.verify_code(p)
 selected=json.loads((OLD/'selection.json').read_text());weight=selected['selected_weight'];assert selected['status']=='SELECTED' and weight==.0001 and not selected['test_used']
 jobs=[j for j in p['jobs'] if j['label']=='Baseline'];assert len(jobs)==24
 protected={str(TRAIN/'experiment_protocol.json'):W.sha(TRAIN/'experiment_protocol.json'),str(OLD/'selection.json'):W.sha(OLD/'selection.json')}
 for j in jobs:
  for f in [Path(j['manifest']),Path(j['run_dir'])/'config.json',Path(j['run_dir'])/'history.json',Path(j['run_dir'])/'status.json']:protected[str(f)]=W.sha(f)
 save('protected_audit_inputs.json',protected)
 device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(14336*2**20/torch.cuda.get_device_properties(device).total_memory)
 W.w.gpu_identity(OUT)
 gpu=json.loads((OUT/'gpu_identity.json').read_text());assert all(x[0].strip()=='GPU-2a1fc5e5-245d-a3f9-9fbc-1cd16185cbc4' for x in gpu['nvml'])
 job=jobs[0];assert (job['ratio'],job['lead'],job['seed'])==(10,7,42)
 print('Building original Oracle96 bundle/model',flush=True)
 a,b,model,unused,cfg=W.setup(job,p,device)
 ref=json.loads((Path(job['run_dir'])/'config.json').read_text())
 assert cfg['shared_initial_sha256']==ref['shared_initial_sha256']
 for key in ['stats','split_sha256','tensor_fingerprints','parameters','rhs_scale']:assert cfg[key]==ref[key],key
 # Independent reconstruction of the same native Baseline; compare actual tensors.
 W.t.set_seed(a.seed)
 baseline=W.t.JejuHiLAM(graph=b.graph,input_dim=len(b.train.input_feature_names),hidden_dim=a.hidden_dim,temporal_hidden_dim=a.temporal_hidden_dim,processor_layers=a.processor_layers,mlp_hidden_layers=a.mlp_hidden_layers,prediction_mode=a.prediction_mode,physics_heads=True,we_nonneg=True,we_scale=a.we_scale,rres_scale=a.rres_scale,we_softplus_shift=a.we_softplus_shift).to(device)
 assert all(torch.equal(v,baseline.state_dict()[k]) for k,v in model.state_dict().items())
 assert W.w.state_digest(baseline)==ref['shared_initial_sha256'];del baseline
 initial=W.w.state_digest(model,False)
 a.lambda_physics=0.;a.lambda_rres=weight;a.physics_warmup_epochs=0
 pa=W.t.build_physics_args(a);W.w.p.apply_manifest_physics(a,pa,json.loads(Path(job['manifest']).read_text()))
 assert pa.lambda_rres==weight and pa.lambda_we_tie==0 and pa.zero_rres_physics and pa.rres_penalty_normalize=='rhs' and pa.physics_rhs_scale==cfg['rhs_scale']
 # Sampling: native loader traverses each selected window exactly once.
 checked=W.CheckedEpoch(W.t.make_loader(b.train,a,True),b.train_indices);batches=[]
 for batch in checked:
  if len(batches)<3:batches.append(batch)
 ev=checked.evidence();rh=json.loads((Path(job['run_dir'])/'history.json').read_text())
 assert ev['sample_order_sha256']==rh[0]['sample_order_sha256']
 opt=torch.optim.AdamW(model.parameters(),lr=a.initial_lr,weight_decay=a.weight_decay,betas=(.9,.95))
 lrs=[W.t.set_epoch_lr(opt,a,i) for i in range(1,81)];assert all(abs(x-r['lr'])<1e-14 for x,r in zip(lrs,rh))
 del opt
 calls=dict(physics=0,auxiliary=0,tie=0)
 native_aux=W.t.rres_penalty;native_phys=W.t.daily_rollout_physics_loss;native_tie=W.t.entrainment_tie_penalty
 def forbidden(*args,**kw):calls['physics']+=1;raise AssertionError('Inactive physics/tie called')
 def auxiliary(*args,**kw):calls['auxiliary']+=1;return native_aux(*args,**kw)
 W.t.daily_rollout_physics_loss=forbidden;W.t.entrainment_tie_penalty=forbidden;W.t.rres_penalty=auxiliary
 names,params=zip(*model.named_parameters())
 def norm(gs,sel):return sum(float(g.detach().double().square().sum()) for n,g in zip(names,gs) if g is not None and sel(n))**.5
 shared=lambda n:not n.startswith(('we_head.','rres_head.'))
 head=lambda n:n.startswith('rres_head.')
 evidence=[];preds=[];model.train()
 for i,batch in enumerate(batches):
  out=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
  vals=W.t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],pa,0.,model)
  gd=torch.autograd.grad(vals[1],params,allow_unused=True,retain_graph=True);ga=torch.autograd.grad(vals[3],params,allow_unused=True)
  nd,na,nh=norm(gd,shared),norm(ga,shared),norm(ga,head)
  assert all(torch.isfinite(g).all() for g in ga if g is not None) and nd>0 and na>0 and nh>0
  assert all(g is None for n,g in zip(names,ga) if n.startswith(('we_head.','output_head.')))
  assert float(vals[2])==0 and float(vals[4])==0
  dot=sum(float((x.detach().double()*y.detach().double()).sum()) for n,x,y in zip(names,gd,ga) if shared(n) and x is not None and y is not None)
  evidence.append(dict(batch=i,batch_size=len(batch['start_index']),data_loss=float(vals[1].detach()),auxiliary_loss=float(vals[3].detach()),total_loss=float(vals[5].detach()),shared_data_grad=nd,shared_aux_grad=na,rres_head_grad=nh,weighted_ratio=weight*na/nd,cosine=dot/(nd*na),physics_loss=0.,we_head_aux_grad_absent=True,sst_output_head_aux_grad_absent=True))
  preds.append(out['pred'].detach().clone());del out,vals,gd,ga
  print('Gradient batch',i,evidence[-1],flush=True)
 # Counterfactual only: change 257 last-head parameters in memory, no optimization.
 last=model.rres_head[-1];saved={k:v.detach().clone() for k,v in last.state_dict().items()}
 with torch.no_grad():last.weight.zero_();last.bias.zero_()
 zero=[]
 for i,batch in enumerate(batches):
  out=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=True)
  vals=W.t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],pa,0.,model)
  ga=torch.autograd.grad(vals[3],params,allow_unused=True)
  assert torch.equal(out['pred'],preds[i]) and float(vals[3])==0 and norm(ga,shared)==0 and norm(ga,head)==0
  zero.append(dict(batch=i,auxiliary_loss=float(vals[3]),shared_aux_grad=norm(ga,shared),rres_head_grad=norm(ga,head),SST_prediction_bitwise_unchanged=True,data_loss_unchanged=float(vals[1].detach())==evidence[i]['data_loss'],total_equals_data=torch.equal(vals[1],vals[5])))
  del out,vals,ga
 last.load_state_dict(saved);assert W.w.state_digest(model,False)==initial
 W.t.daily_rollout_physics_loss=native_phys;W.t.entrainment_tie_penalty=native_tie;W.t.rres_penalty=native_aux
 cp=OUT/'checkpoints/audit_only/initial_roundtrip.pt';assert not cp.exists();torch.save(dict(model_state=model.state_dict(),purpose='CPU/GPU pretraining roundtrip only; no optimizer updates',initial_sha=initial),cp)
 loaded=torch.load(cp,map_location=device,weights_only=False);assert all(torch.equal(v,loaded['model_state'][k]) for k,v in model.state_dict().items());del loaded
 unchanged=all(W.sha(f)==h for f,h in protected.items());assert unchanged
 result=dict(timestamp=stamp(),status='BLOCKED_TRIVIAL_ZERO_SOLUTION',scope='3 actual Train batches, ratio10 lead7 seed42, batch16; no optimizer updates',selected_aux_weight=weight,definition='data MSE + 0.0001 * ocean mean[(86400 * rres_scale * tanh(rres_head(shared_features)))^2] / fixed v3 RHS mean-square',aux_warmup=False,original_definition_clear=True,initial_nonzero_shared_gradient=True,initial_rres_head_trainable=True,zero_head_counterexample=True,zeroed_parameter_count=sum(v.numel() for v in last.parameters()),parameters=cfg['parameters'],extra_parameters_vs_oracle_baseline=0,initial_predictor_tensor_equality=True,initial_hash_matches_oracle_baseline=True,shared_initial_sha256=ref['shared_initial_sha256'],stats_splits_fingerprints_match=True,actual_epoch_sampling=ev,lr_schedule_matches_80_actual_epochs=True,physics_calls=calls['physics'],auxiliary_calls=calls['auxiliary'],initial_batches=evidence,zero_head_batches=zero,checkpoint_roundtrip='PASS',protected_inputs_unchanged=unchanged,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,formal_training_started=False,optimizer_updates=0,test_evaluation=False)
 save('gradient_audit.json',result)
 save('run_state.json',dict(status='BLOCKED_PREFLIGHT',completed=0,total=24,running=[],reason='Existing zero-target disconnected Rres head admits exact zero auxiliary loss and zero shared auxiliary gradient without changing SST prediction; user requested stop on this degeneracy.',gpus=[5,6,7],max_processes_per_gpu=2,training_started=False,test_evaluation='NOT_RUN',timestamp=stamp()))
 with (OUT/'run_manifest.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['model','ratio','lead','seed','manifest','baseline_run','lambda_physics','lambda_aux','status']);w.writeheader()
  for j in jobs:w.writerow(dict(model='Auxiliary-only',ratio=j['ratio'],lead=j['lead'],seed=j['seed'],manifest=j['manifest'],baseline_run=j['run_dir'],lambda_physics=0,lambda_aux=weight,status='BLOCKED_PREFLIGHT'))
 b.ds.close();record('scripts/oracle96_auxiliary24/audit.py 및 별도 outputs/v8_actual_epoch_auxiliary24_20261015 생성. 기존 λaux=0.0001 정의 감사와 실제 Train 3 batch GPU5 gradient/초기값/sampling/LR/체크포인트 검증. Rres 마지막257개 parameter만0으로 하면 SST예측 동일·auxloss/공유gradient=0 확인. 첨부 중단 조건에 따라 24개 본학습 미시작, Test 미실행. 기존 코드/데이터/실험 결과 보존.')
 print('Preflight blocked by exact zero-head counterexample; 24 runs NOT STARTED',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  save('audit_failure.json',dict(timestamp=stamp(),traceback=traceback.format_exc()));raise
