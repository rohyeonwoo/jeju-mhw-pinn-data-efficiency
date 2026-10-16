"""Bounded six-run validation pilot; GPU0 excluded, no test inference or auto expansion."""
import os,sys,json,csv,subprocess,time,fcntl,math
from pathlib import Path
import train_v3_study as w
ROOT=w.ROOT;OUT=ROOT/'outputs/p0_repair_20261009'
def query(kind):
 fields='index,uuid,memory.total,memory.free' if kind=='gpu' else 'gpu_uuid,pid,process_name,used_memory'
 return [[x.strip() for x in r] for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-'+('gpu' if kind=='gpu' else 'compute-apps')+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()) if r]
def finish(protocol):
 rows=[];groups=[];checks=[]
 for job in protocol['jobs']:
  d=Path(job['run_dir']);status=json.loads((d/'status.json').read_text());cfg=json.loads((d/'config.json').read_text());hist=json.loads((d/'history.json').read_text())
  assert status['status']=='trained'
  assert status['best_epoch']==min(hist,key=lambda x:x['val_rmse'])['epoch']
  assert all(r['guard_weighted_ratio_max']<=r['train_lambda_physics_effective']+1e-5 and r['guard_min_cosine_after']>=-1e-4 and r['rres_calls']==0 for r in hist)
  bdir=Path(job['reference_baseline_run']);pdir=Path(job['reference_p0_run'])
  bc=json.loads((bdir/'config.json').read_text());pc=json.loads((pdir/'config.json').read_text());bh=json.loads((bdir/'history.json').read_text());ph=json.loads((pdir/'history.json').read_text())
  for key in ['shared_initial_sha256','manifest_sha256','stats','split_sha256','tensor_fingerprints']:assert cfg[key]==bc[key]==pc[key]
  for reference in [bh,ph]:
   for new,old in zip(hist,reference):assert new['batch_order_sha256']==old['batch_order_sha256']
  bv=min(r['val_rmse'] for r in bh);pv=min(r['val_rmse'] for r in ph);nv=status['best_val_rmse']
  rows.append(dict(lead=job['lead'],seed=job['seed'],baseline_val_rmse=bv,original_p0_val_rmse=pv,guarded_val_rmse=nv,improvement_vs_baseline_percent=100*(bv-nv)/bv,improvement_vs_original_p0_percent=100*(pv-nv)/pv,best_epoch=status['best_epoch'],epochs=status['epochs'],gradient_conflict_fraction=sum(r['guard_conflict_mean'] for r in hist)/len(hist),max_weighted_physics_data_norm=max(r['guard_weighted_ratio_max'] for r in hist),status='completed'))
 for lead in [7,14]:
  rs=[r for r in rows if r['lead']==lead]
  avg=lambda k:sum(r[k] for r in rs)/3
  count=sum(r['guarded_val_rmse']<r['baseline_val_rmse'] for r in rs)
  accepted=avg('guarded_val_rmse')<min(avg('baseline_val_rmse'),avg('original_p0_val_rmse')) and count>=2
  groups.append(dict(lead=lead,baseline_val_rmse=avg('baseline_val_rmse'),original_p0_val_rmse=avg('original_p0_val_rmse'),guarded_val_rmse=avg('guarded_val_rmse'),mean_paired_improvement_vs_baseline_percent=avg('improvement_vs_baseline_percent'),mean_paired_improvement_vs_original_p0_percent=avg('improvement_vs_original_p0_percent'),improved_baseline_seeds=count,accepted=accepted))
 accepted=all(g['accepted'] for g in groups)
 w.save(OUT/'validation_result.json',dict(technical='PASS',performance='PASS_PILOT_CRITERION' if accepted else 'NOT_ESTABLISHED',test_evaluated=False,runs=rows,by_lead=groups,scope='exploratory validation-only pilot; no proof of test/generalization improvement; no additional training automatically'))
 with (OUT/'validation_comparison.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# P0 gradient correction — validation-only pilot','',f'Technical safeguards: PASS. Performance criterion: {"PASS" if accepted else "NOT ESTABLISHED"}. Test inference: none.','', 'Same P0 daily physics and nonnegative entrainment; only the optimizer gradient combination is changed. Original v3 statistics, initial predictor and common batch-order prefixes match the reference experiments.','', '| Lead | Seed | Baseline val RMSE | Original P0 val RMSE | Guarded val RMSE | vs Baseline (%) | vs P0 (%) |','|---|---|---|---|---|---|---|']
 for r in rows:lines.append(f"| {r['lead']} | {r['seed']} | {r['baseline_val_rmse']:.6f} | {r['original_p0_val_rmse']:.6f} | {r['guarded_val_rmse']:.6f} | {r['improvement_vs_baseline_percent']:+.3f} | {r['improvement_vs_original_p0_percent']:+.3f} |")
 lines+=['','## Prespecified acceptance','', 'For EACH lead, three-seed mean best validation RMSE must be below both Baseline and original P0, with at least two seeds improving over Baseline. No additional arms or Test evaluation are run automatically.','',json.dumps(groups,indent=2),'', 'The scalar native loss is recorded for comparison; gradient projection is not its derivative. AdamW and finite-step dynamics do not guarantee improved generalization. Physical-model mismatch remains a scientific limitation, rather than being re-labelled as resolved.']
 (OUT/'validation_report.md').write_text('\n'.join(lines)+'\n');w.log(f'P0guarded6회검증완료: technicalPASS, performance={accepted}. validation_result.json/validation_report.md. Test미평가·48조건확장없음.')
def main():
 lock=(OUT/'pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'gpu_smoke/result.json').read_text())['status']=='PASS';jobs=protocol['jobs']
 statefile=OUT/'pilot_state.json';previous=json.loads(statefile.read_text()) if statefile.exists() else {};active={};attempts=previous.get('attempts',[])
 for row in previous.get('active',[]):
  pid=row['pid']
  try:token=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
  except FileNotFoundError:continue
  if token==row['start_ticks'] and Path(f'/proc/{pid}').stat().st_uid==os.getuid():active[row['job']]=dict(row,process=None)
 peak=json.loads((OUT/'gpu_smoke/result.json').read_text())['peak_reserved_mib'];reserve=4096;failure=[]
 while True:
  done=[]
  for j in jobs:
   d=Path(j['run_dir']);status=d/'status.json'
   if status.exists() and json.loads(status.read_text())['status']=='trained':
    cfg=json.loads((d/'config.json').read_text());assert cfg['protocol_sha256']==w.digest(protocol) and cfg['manifest_sha256']==w.sha(j['manifest']);done.append(j['id'])
  for jid,row in list(active.items()):
   proc=row['process'];rc=proc.poll() if proc else (None if Path(f'/proc/{row["pid"]}').exists() else 0)
   if rc is not None:
    del active[jid]
    if rc!=0 or jid not in done:failure.append(dict(job=jid,returncode=rc))
  state=dict(timestamp=w.stamp(),planned=6,completed=len(done),failed=failure,attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],phase='training_validation',gpu0_excluded=True)
  w.save(statefile,state)
  if failure:raise RuntimeError(failure)
  if len(done)==6 and not active:
   finish(protocol);state.update(phase='completed',timestamp=w.stamp());w.save(statefile,state);return
  inventory=[r for r in query('gpu') if r[0] in ['1','2','3']];apps=query('apps');owned={str(r['pid']) for r in active.values()}
  for j in jobs:
   if j['id'] in done or j['id'] in active:continue
   choices=[]
   for gpu in inventory:
    if any(r[0]==gpu[1] and r[1] not in owned for r in apps):continue
    count=sum(r['gpu']==gpu[0] for r in active.values());cap=min(3,int((int(gpu[2])-reserve)//(peak*1.15)))
    if count<cap and int(gpu[3])>=reserve+peak*1.15:choices.append((int(gpu[0]),gpu))
   if not choices:break
   _,gpu=min(choices);env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu[0],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
    cache=OUT/'runtime'/str(j['id'])/key;cache.mkdir(parents=True,exist_ok=True);env[key]=str(cache)
   logfile=OUT/f'pilot_{j["id"]}_{time.time_ns()}.log'
   with logfile.open('w') as f:proc=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/train_p0_guarded.py'),'--study',str(OUT),'--job',str(j['id'])],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   row=dict(job=j['id'],pid=proc.pid,gpu=gpu[0],start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],log=str(logfile));attempts.append(row);active[j['id']]=dict(row,process=proc);owned.add(str(proc.pid));print(json.dumps(row),flush=True)
  state.update(attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()]);w.save(statefile,state);time.sleep(20)
if __name__=='__main__':main()
