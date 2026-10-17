"""Fresh v3 48-run protocol, first-pair gate, <=3 GPUs and <=4 workers per GPU."""
import os,sys,json,time,csv,subprocess,math,fcntl,traceback
from pathlib import Path
from datetime import datetime
import train_v3_study as w
ROOT=w.ROOT;PYTHON=sys.executable;V3=ROOT/'data/subsets/preprocessing_subset_v3'
def query(kind):
 fields='index,uuid,memory.total,memory.used,memory.free,utilization.gpu' if kind=='gpu' else 'gpu_uuid,pid,process_name,used_memory'
 return [[c.strip() for c in r] for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-'+('gpu' if kind=='gpu' else 'compute-apps')+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()) if r]
def prepare(study):
 assert study.resolve().is_relative_to(ROOT/'outputs')
 study.mkdir(parents=True,exist_ok=False)
 index=json.loads((V3/'index.json').read_text());assert len(index['datasets'])==24
 validation=json.loads((V3/'validation_report.json').read_text());assert all(v=='PASS' for v in validation['gates'].values())
 smoke=json.loads((ROOT/'outputs/v3_preflight_20261008/gpu_smoke.json').read_text());assert smoke['status']=='PASS'
 hashes={};manifests={}
 def checked(path,expected):
  path=Path(path).resolve()
  if str(path) not in hashes:hashes[str(path)]=w.sha(path)
  assert hashes[str(path)]==expected,str(path)
 for item in index['datasets']:
  path=V3/item['manifest'];checked(path,item['sha256']);m=json.loads(path.read_text());assert m['schema']=='subset_normalization_v3'
  checked(m['data_path'],m['data_sha256']);checked(m['subset_path'],m['subset_sha256']);checked(path.parent/m['climatology_file'],m['climatology_sha256'])
  for k,v in m['source_sha256'].items():checked(ROOT/k,v)
  assert math.isfinite(m['physics_rhs_scale']) and m['physics_rhs_scale']>0
  manifests[item['ratio'],item['lead'],item['seed']]=path
 # Fresh protocol references the preserved v3 artifacts and current unchanged smoke-tested src.
 names=sorted([str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')])+['scripts/train_v3_study.py','scripts/manage_v3_study.py','scripts/analyze_v3_study.py']
 jobs=[]
 for ratio in [10,25,50,100]:
  for lead in [7,14]:
   for seed in [42,7,123]:
    for arm in ['baseline','p0']:
     name=f'ratio_{ratio}/lead{lead}_seed{seed}/{arm}';path=manifests[ratio,lead,seed]
     jobs.append(dict(id=len(jobs),name=name,ratio=ratio,lead=lead,seed=seed,arm=arm,manifest=str(path),manifest_sha256=w.sha(path),run_dir=str(study/'runs'/name)))
 protocol=dict(created_at=w.stamp(),jobs=jobs,code_sha256={n:w.sha(ROOT/n) for n in names},settings=dict(batch_size=16,precision='fp32',epochs=80,virtual_epoch_steps=478,patience=12,optimizer='AdamW',betas=[.9,.95],weight_decay=.1,lr=2e-4,initial_lr=2e-5,lr_warmup_epochs=5,physics_warmup_epochs=5,lambda_physics_p0=.1,lambda_rres=0,lambda_we_tie=0,clip_grad_norm=1.,catastrophic_gradient_guard='stop after 100 consecutive preclip norms > 1e6; ordinary clipping is not failure'),resources=dict(max_gpus=3,max_workers_per_gpu=4,minimum_free_reserve_mib=4096,admission_multiplier=1.15,first_pair_jobs=[0,1]),evaluation=dict(test_after_all_training=True,selection='val SST RMSE',mhw_clim_frac=.70,mhw_quantile=.90,mhw_window=5,mhw_min_duration=5,mhw_area_frac=.15,mhw_onset_tolerance=3,bootstrap_blocks=[30,90,180],bootstrap_replicates=4000,primary_block=90,bootstrap_definition='circular paired time blocks shared across arms and seeds; CI conditional on fitted models; separate seed SD'),prior_protocol=str(ROOT/'outputs/data_efficiency/physics_pilot_lead7_20260925_4plus4/_queue/protocol.json'),preflight_artifact_hashes=hashes)
 w.save(study/'protocol.json',protocol)
 (study/'logs').mkdir();(study/'runtime').mkdir()
 w.save(study/'preflight.json',dict(status='ARTIFACT_HASH_PASS_RUNTIME_PENDING',manifests=24,code_identical_to_v3_smoke=True,existing_v3_gates=validation['gates'],smoke='PASS'))
 w.log(f'{study}: 48조건 protocol 저장. Baseline/P0 λ=.1/Rres0, 478step×80/patience12 유지. 최대3GPU/각4프로세스, 4GiB 및1.15배 메모리 여유 확인.')
 print(study,flush=True)
def pair_gate(study,jobs):
 a,b=[Path(j['run_dir']) for j in jobs]
 configs=[json.loads((d/'config.json').read_text()) for d in [a,b]]
 for key in ['manifest_sha256','stats','rhs_scale','shared_initial_sha256','split_sha256','tensor_fingerprints']:
  assert configs[0][key]==configs[1][key],key
 assert configs[0]['applied_rhs_scale'] is None and configs[1]['applied_rhs_scale']==configs[1]['rhs_scale']
 histories=[json.loads((d/'history.json').read_text()) for d in [a,b]]
 for ra,rb in zip(*histories):assert ra['batch_order_sha256']==rb['batch_order_sha256']
 for history in histories:
  assert all(r['rres_calls']==0 and math.isfinite(r['gradient_norm_max']) for r in history)
 assert all(r['physics_calls']==0 for r in histories[0]);assert all(r['physics_calls']>0 for r in histories[1])
 assert json.loads((b/'initial_physics_gradient.json').read_text())['predictor_contribution']
 for d in [a,b]:assert json.loads((d/'status.json').read_text())['checkpoint_roundtrip']=='PASS'
 return dict(status='PASS',jobs=[j['id'] for j in jobs],same_predictor_initialization=True,same_batch_order=True,same_data_statistics=True,finite_losses_gradients=True,validation_selection=True,checkpoint_roundtrip=True,physics_gradient_contribution=True,parameters=[c['parameters'] for c in configs],performance_not_a_gate=True)
def supervise(study):
 lock=(study/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((study/'protocol.json').read_text());jobs=protocol['jobs'];procs={};attempts=[]
 assert json.loads((study/'preflight.json').read_text())['status']=='PASS'
 for name,value in protocol['code_sha256'].items():assert w.sha(ROOT/name)==value,name
 statefile=study/'queue_state.json'
 if statefile.exists():
  previous=json.loads(statefile.read_text());attempts=previous.get('attempts',[])
  for active in previous.get('active',[]):
   pid=active['pid']
   try:started=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
   except (FileNotFoundError,PermissionError):continue
   if started==active.get('proc_start_ticks') and Path(f'/proc/{pid}').stat().st_uid==os.getuid():procs[active['job']]=dict(**active,process=None)
 else:previous={}
 apps=query('apps');busy={r[0] for r in apps};inventory=query('gpu')
 if previous.get('gpus'):gpus=previous['gpus']
 else:gpus=[dict(index=r[0],uuid=r[1],total_mib=int(r[2])) for r in inventory if r[1] not in busy and int(r[3])<100 and int(r[5])==0][:3]
 if len(gpus)<1:raise RuntimeError('No idle GPU available; no training launched')
 assert len(gpus)<=3
 w.save(study/'gpu_inventory_at_start.json',dict(gpus=gpus,inventory=inventory,apps=apps,timestamp=w.stamp()))
 phase=previous.get('phase','train');peak=max(x['peak_reserved_mib'] for x in json.loads((ROOT/'outputs/v3_preflight_20261008/gpu_smoke.json').read_text())['models'])
 failed=[]
 while True:
  done=[]
  for job in jobs:
   d=Path(job['run_dir']);s=d/('status.json' if phase=='train' else 'evaluation_status.json')
   if s.exists():
    value=json.loads(s.read_text())
    if value['status']==('trained' if phase=='train' else 'completed'):
     config=json.loads((d/'config.json').read_text())
     assert config['protocol_sha256']==w.digest(protocol) and config['manifest_sha256']==job['manifest_sha256']
     assert w.sha(job['manifest'])==job['manifest_sha256']
     expected=config['config_sha256'];assert w.digest({k:v for k,v in config.items() if k!='config_sha256'})==expected
     if phase=='train':assert value['config_sha256']==expected
     done.append(job['id'])
    peak=max(peak,value.get('peak_reserved_mib',0))
  for jid,record in list(procs.items()):
   proc=record.get('process');rc=proc.poll() if proc else (None if Path(f'/proc/{record["pid"]}').exists() else 0)
   if rc is None:continue
   del procs[jid]
   if rc!=0 or jid not in done:failed.append(dict(job=jid,phase=phase,returncode=rc))
  gate=study/'first_pair_gate.json'
  if phase=='train' and not gate.exists() and 0 in done and 1 in done:
   w.save(gate,pair_gate(study,jobs[:2]));w.log('첫 정식 Baseline/P0 한 쌍 실행검증 PASS. 성능과 무관하게 나머지46조건 확장.')
  state=dict(timestamp=w.stamp(),supervisor_pid=os.getpid(),phase=phase,planned=48,completed=len(done),failed=failed,attempts=attempts,gpus=gpus,peak_reserved_mib=peak,first_pair_gate=gate.exists(),active=[{k:v for k,v in r.items() if k!='process'} for r in procs.values()],completed_ids=done)
  w.save(statefile,state)
  if failed:
   w.save(study/'execution_failure.json',state);raise RuntimeError(f'Worker failures; no new launches: {failed}')
  if len(done)==48 and not procs:
   if phase=='train':
    for i in range(0,48,2):pair_gate(study,jobs[i:i+2])
    w.save(study/'all_pairs_gate.json',dict(status='PASS',pairs=24));phase='evaluate';continue
   w.log('48개 학습 및48개 best-checkpoint Test 평가 완료. 통계/그림 집계 시작.')
   subprocess.run([PYTHON,'-B',str(ROOT/'scripts/analyze_v3_study.py'),'--study',str(study)],check=True)
   state.update(phase='completed',completed=48);w.save(statefile,state);w.log('v3 정식48회 실험·집계 완료; final_report.md 참조.');return
  permitted=list(range(48)) if phase=='evaluate' or gate.exists() else [0,1]
  pending=[j for j in jobs if j['id'] in permitted and j['id'] not in done and j['id'] not in procs]
  apps=query('apps');inventory={r[1]:r for r in query('gpu')};owned={str(r['pid']) for r in procs.values()}
  for job in pending:
   candidates=[]
   for gpu in gpus:
    foreign=[r for r in apps if r[0]==gpu['uuid'] and r[1] not in owned]
    current=sum(r['gpu']==gpu['index'] for r in procs.values())
    free=int(inventory[gpu['uuid']][4]);total=gpu['total_mib']
    cap=min(4,int((total-4096)//(peak*1.15)))
    if not foreign and current<cap and free>=4096+peak*1.15:
     # Reserve memory for workers still in dataset/model setup as well.
     if (current+1)*peak*1.15+4096<=total:candidates.append((current,gpu))
   if not candidates:break
   _,gpu=min(candidates,key=lambda x:(x[0],x[1]['index']))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['index'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MHW_V3_STUDY=str(study),MHW_V3_JOB=str(job['id']))
   for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
    dest=study/'runtime'/str(job['id'])/key;dest.mkdir(parents=True,exist_ok=True);env[key]=str(dest)
   logfile=study/'logs'/f'{phase}_{job["id"]:02d}_{time.time_ns()}.log'
   with logfile.open('w') as loghandle:
    proc=subprocess.Popen([PYTHON,'-u','-B',str(ROOT/'scripts/train_v3_study.py'),'--study',str(study),'--job',str(job['id']),'--phase',phase],stdout=loghandle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   entry=dict(job=job['id'],gpu=gpu['index'],pid=proc.pid,proc_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],phase=phase,log=str(logfile),started_at=w.stamp())
   attempts.append(entry);procs[job['id']]=dict(**entry,process=proc);owned.add(str(proc.pid))
   print(json.dumps(entry),flush=True)
  state.update(attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in procs.values()]);w.save(statefile,state)
  time.sleep(20)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);ap.add_argument('--study',type=Path,required=True);opt=ap.parse_args()
 if opt.action=='prepare':prepare(opt.study)
 else:supervise(opt.study)
