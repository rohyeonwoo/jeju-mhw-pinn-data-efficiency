"""Final audit: frozen Auxiliary24 Test inference only, no optimizer."""
import os,sys,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path('/ceph_data/yeonwoo/MHW_subset');OUT=ROOT/'outputs/paper_final_verification_20261015'
sys.dont_write_bytecode=True
for path in ['scripts/oracle96_test','scripts/oracle96_auxiliary24']:sys.path.insert(0,str(ROOT/path))
import aux_worker as A
import metrics as M
import numpy as np
import torch
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 p=OUT/'runtime'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
A.verify(A.plan());M.verify_evaluation(M.evaluation_protocol())
gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in [5,6,7]
jobid=int(sys.argv[1]);j=A.plan()['jobs'][jobid];d=OUT/'auxiliary_test'/'runs'/j['name'];d.mkdir(parents=True,exist_ok=True)
frozen=json.loads((OUT/'auxiliary_test/protocol.json').read_text());entry=frozen['jobs'][jobid]
try:
 torch.set_num_threads(1);device=torch.device('cuda:0')
 torch.cuda.set_per_process_memory_fraction(14336*2**20/torch.cuda.get_device_properties(device).total_memory)
 A.W.w.gpu_identity(d)
 ident=json.loads((d/'gpu_identity.json').read_text());uuids=A.subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines();uid=next(x.split(',')[1].strip() for x in uuids if int(x.split(',')[0])==gpu)
 assert all(x[0].strip()==uid for x in ident['nvml'])
 A.save(d/'status.json',dict(status='RUNNING',gpu=gpu,pid=os.getpid()))
 a,b,model,pa,cfg,ref=A.setup(j,device)
 assert cfg==json.loads((Path(j['run_dir'])/'config.json').read_text())
 cp=Path(entry['checkpoint']);assert A.sha(cp)==entry['sha256']
 c=torch.load(cp,map_location=device,weights_only=False)
 assert c['config_sha256']==cfg['config_sha256'] and c['stats']==b.stats and c['epoch']==entry['epoch'] and c['best_value']==entry['val_rmse']
 model.load_state_dict(c['model_state'],strict=True);del c
 for v in model.parameters():v.requires_grad_(False)
 model.eval();before=A.W.w.state_digest(model,False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 collect=lambda ds:A.W.t.collect_predictions(model,A.W.t.make_loader(ds,a,False),b.graph,device,*b.stats['target'],precision='fp32',physics_args=None)
 with torch.no_grad():
  f=collect(b.test);oracle=A.W.mb.ForcingDataset(b.test.base,A.W.mb.arrays())
  for ds in [oracle,A.W.old.PoisonForcing(oracle)]:
   other=collect(ds)
   for k in ['pred_raw','target_raw','target_index','mask']:assert np.array_equal(f[k],other[k]),k
 assert before==A.W.w.state_digest(model,False) and all(x.grad is None for x in model.parameters())
 assert np.array_equal(f['target_index'],b.test_indices+29+a.lead)
 dates=b.ds.time.values[f['target_index']]
 reference=A.ORACLE/'evaluation_reference.npz';assert A.sha(reference)==frozen['reference_sha256']
 with np.load(reference) as z:measured,daily=M.score(f,dates,z['clim'],z['threshold'])
 base=A.protocol()['jobs'][j['baseline_id']]
 with np.load(A.ORACLE/'test_evaluation_20261012/runs'/base['name']/'fields.npz') as z:
  for k in ['target_index','target_raw','mask']:assert np.array_equal(f[k],z[k]),k
 measured.update(model='Auxiliary',model_label='Auxiliary',ratio=j['ratio'],lead=j['lead'],seed=j['seed'],split='test',best_epoch=entry['epoch'],validation_rmse=entry['val_rmse'],checkpoint_sha256=entry['sha256'],optimizer_steps=0,forcing_invariance='PASS',peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,evaluation_seconds=time.monotonic()-start)
 assert A.sha(cp)==entry['sha256']
 np.savez_compressed(d/'fields.npz',**f,target_dates=dates);np.savez_compressed(d/'daily_metrics.npz',**daily)
 A.save(d/'metrics.json',measured);A.save(d/'forcing_invariance_test.json',dict(status='PASS',modes=['removed','oracle_present','NaN_poisoned'],prediction_bitwise_identical=True,model_state_unchanged=True,baseline_truth_identical=True,optimizer_steps=0))
 A.save(d/'status.json',dict(status='COMPLETED',gpu=gpu,pid=os.getpid(),timestamp=A.stamp()));b.ds.close()
 print(j['name'],measured['rmse'],flush=True)
except BaseException:
 A.save(d/'status.json',dict(status='FAILED',error=traceback.format_exc()));raise
