"""Matched-head four-arm validation study; waits for dense study; GPU1/2/3 only."""
import os,sys,json,csv,subprocess,time,fcntl,math
from pathlib import Path
import train_v3_study as w
ROOT=w.ROOT;OUT=ROOT/'outputs/v3_ablation_lead7_20261009'
def query(kind):
 fields='index,uuid,memory.total,memory.free' if kind=='gpu' else 'gpu_uuid,pid,process_name,used_memory'
 return [[x.strip() for x in r] for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-'+('gpu' if kind=='gpu' else 'compute-apps')+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()) if r]
def finish(protocol):
 from analyze_v3_ablation import summarize
 result=summarize(OUT,require_complete=True)
 w.log('Four-arm v3 ablation complete: results.json/report.md; no Test inference. '+str(result['groups']))
def main():
 lock=(OUT/'pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/'cpu_smoke/result.json').read_text())['status']=='PASS'
 protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'gpu_smoke/result.json').read_text())['status']=='PASS';jobs=protocol['jobs']
 statefile=OUT/'pilot_state.json';previous=json.loads(statefile.read_text()) if statefile.exists() else {};active={};attempts=previous.get('attempts',[])
 for row in previous.get('active',[]):
  pid=row['pid']
  try:token=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
  except FileNotFoundError:continue
  if token==row['start_ticks'] and Path(f'/proc/{pid}').stat().st_uid==os.getuid():active[row['job']]=dict(row,process=None)
 peak=json.loads((OUT/'gpu_smoke/result.json').read_text())['peak_reserved_mib'];reserve=4096;failure=[]
 from analyze_v3_ablation import summarize
 reported=-1
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
  state=dict(timestamp=w.stamp(),planned=len(jobs),completed=len(done),failed=failure,attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],phase='training_validation',gpu0_excluded=True)
  w.save(statefile,state)
  if len(done)!=reported:summarize(OUT);reported=len(done)
  if failure:raise RuntimeError(failure)
  if len(done)==len(jobs) and not active:
   finish(protocol);state.update(phase='completed',timestamp=w.stamp());w.save(statefile,state);return
  inventory=[r for r in query('gpu') if r[0] in ['1','2','3']];apps=query('apps');owned={str(r['pid']) for r in active.values()}
  dense=ROOT/'outputs/p0_dense_validation_20261009/pilot_state.json'
  if dense.exists() and json.loads(dense.read_text()).get('phase')!='completed':
   state.update(waiting_for='existing dense-validation study to complete');w.save(statefile,state);time.sleep(20);continue
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
   with logfile.open('w') as f:proc=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/train_v3_ablation.py'),'--study',str(OUT),'--job',str(j['id'])],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   row=dict(job=j['id'],pid=proc.pid,gpu=gpu[0],start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],log=str(logfile));attempts.append(row);active[j['id']]=dict(row,process=proc);owned.add(str(proc.pid));print(json.dumps(row),flush=True)
  state.update(attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()]);w.save(statefile,state);time.sleep(20)
if __name__=='__main__':main()
