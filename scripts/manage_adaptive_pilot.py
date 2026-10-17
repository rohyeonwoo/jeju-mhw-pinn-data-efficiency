"""Gated residual -> gradient -> smoke -> six trainings -> Validation only."""
from adaptive_common import *
import subprocess,fcntl
PHASES=['residual','gradient','preflight','train','evaluate','report']
def query(kind):
 fields='index,uuid,memory.free' if kind=='gpu' else 'gpu_uuid,pid,process_name,used_memory'
 return [[v.strip() for v in row] for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-'+('gpu' if kind=='gpu' else 'compute-apps')+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()) if row]
def alive(row):
 try:
  p=Path('/proc')/str(row['pid']);v=(p/'stat').read_text().rsplit(')',1)[1].split();return p.stat().st_uid==os.getuid() and v[0]!='Z' and v[19]==row['start_ticks']
 except FileNotFoundError:return False

def status_path(j,phase):
 if phase in ['residual','gradient']:return OUT/'diagnostics'/str(j['id'])/(phase+'_status.json')
 if phase=='preflight':return OUT/'preflight'/str(j['id'])/'status.json'
 if phase=='train':return Path(j['run_dir'])/'status.json'
 return OUT/'evaluations'/str(j['id'])/'status.json'
def snapshot(jobs,phase,active,failed):
 rows=[]
 for j in jobs:
  f=status_path(j,phase);v=json.loads(f.read_text()) if f.exists() else {};rows.append(dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model='adaptive',phase=phase,status='running' if j['id'] in active else v.get('status','waiting'),run_dir=j['run_dir'],best_epoch=v.get('best_epoch'),best_val_rmse=v.get('best_val_rmse')))
 for r in protocol()['reuse']:rows.append(dict(job='reuse',ratio=r['ratio'],lead=r['lead'],seed=r['seed'],model=r['arm'],phase='reused_validation_only',status='verified',run_dir=r['run_dir'],best_epoch=r['best_epoch'],best_val_rmse=r['validation_rmse']))
 csvsave(OUT/'run_manifest.csv',rows)
def main():
 lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pr=protocol();jobs=pr['jobs']
 for n,hsh in pr['code_sha256'].items():assert sha(ROOT/n)==hsh
 for n,hsh in json.loads((OUT/'operator_sources.json').read_text()).items():assert sha(ROOT/n)==hsh
 statefile=OUT/'queue_state.json';old=json.loads(statefile.read_text()) if statefile.exists() else {};phase=old.get('phase','residual');assert not old.get('failed');active={r['job']:dict(**r,process=None) for r in old.get('active',[]) if alive(r)};attempts=old.get('attempts',[]);failed=[]
 while True:
  if phase=='completed':return
  if phase=='report':
   with (OUT/'analysis.log').open('a') as logf:subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts/analyze_adaptive_pilot.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=logf,stderr=subprocess.STDOUT,check=True)
   save(statefile,dict(phase='completed',timestamp=stamp(),active=[],waiting=[],failed=[],attempts=attempts));log('Adaptive pilot complete: six new runs, Validation only; no new Test or expansion.');return
  done=[]
  for j in jobs:
   f=status_path(j,phase)
   if f.exists():
    s=json.loads(f.read_text())
    if s.get('status')==('trained' if phase=='train' else 'PASS'):done.append(j['id'])
  for jid,r in list(active.items()):
   process=r['process'];rc=process.poll() if process else (None if alive(r) else 0)
   if rc is not None:
    del active[jid]
    if rc!=0 or jid not in done:failed.append(dict(job=jid,phase=phase,returncode=rc,log=r['log']))
  state=dict(timestamp=stamp(),phase=phase,completed=len(done),done=done,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],waiting=[j['id'] for j in jobs if j['id'] not in done and j['id'] not in active],failed=failed,attempts=attempts,gpu_ids=[1,2,3],max_processes_per_gpu=4)
  save(statefile,state);snapshot(jobs,phase,active,failed)
  if failed:raise RuntimeError(f'Stopped admissions; diagnose phase failure: {failed}')
  if len(done)==6 and not active:
   if phase in ['residual','gradient','preflight']:
    with (OUT/'phase_reports.log').open('a') as f:subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts/summarize_adaptive_pilot.py'),'--phase',phase],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
   if phase=='train':
    frozen=[]
    for j in jobs:
     d=Path(j['run_dir']);s=json.loads((d/'status.json').read_text());hist=json.loads((d/'history.json').read_text());assert len(hist)==80 or len(hist)-s['best_epoch']==12
     assert s['best_epoch']==min(hist,key=lambda r:r['val_rmse'])['epoch'];frozen.append(dict(job=j['id'],best_sha256=sha(d/'best.pt'),config_sha256=json.loads((d/'config.json').read_text())['config_sha256'],best_val_rmse=s['best_val_rmse']))
    save(OUT/'checkpoint_lock.json',dict(status='PASS',rows=frozen,selection='Validation only',timestamp=stamp()))
   phase=PHASES[PHASES.index(phase)+1];continue
  peak=json.loads((OUT/'preflight.json').read_text())['peak_reserved_mib'] if (OUT/'preflight.json').exists() else 12000
  # At most1 diagnostic/smoke perGPU, at most2 production workers perGPU; user maximum4 includes existing processes.
  cap=2 if phase in ['train','evaluate'] else 1
  for j in jobs:
   if j['id'] in done or j['id'] in active:continue
   inventory=[] if phase in ['residual','gradient'] else [g for g in query('gpu') if g[0] in ['1','2','3']];apps=[] if phase in ['residual','gradient'] else query('apps');own={str(r['pid']) for r in active.values()};choices=[(0,0,'cpu')] if phase in ['residual','gradient'] and not active else []
   for g in inventory:
    foreign=[x for x in apps if x[0]==g[1] and x[1] not in own]
    try:safe=all(x[2]=='yeonwoo' and Path('/proc/'+x[1]).stat().st_uid==os.getuid() for x in foreign)
    except FileNotFoundError:continue
    if not safe:continue
    count=sum(r['gpu']==g[0] for r in active.values());pending=sum(r['gpu']==g[0] and str(r['pid']) not in {x[1] for x in apps} for r in active.values())
    if count>=cap or count+len(foreign)>=4:continue
    if int(g[2])-pending*peak*1.2<4096+peak*1.2:continue
    choices.append((count+len(foreign),-int(g[2]),g[0]))
   if not choices:break
   gpu=min(choices)[2];env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu=='cpu' else gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
    d=OUT/'runtime'/str(j['id'])/k;d.mkdir(parents=True,exist_ok=True);env[k]=str(d)
   script={'residual':'diagnose_adaptive_pilot.py','gradient':'diagnose_adaptive_pilot.py','preflight':'preflight_adaptive_pilot.py','train':'train_adaptive_pilot.py','evaluate':'evaluate_adaptive_pilot.py'}[phase]
   argv=[sys.executable,'-u','-B',str(ROOT/'scripts'/script),'--job',str(j['id'])]
   if phase in ['residual','gradient']:argv+=['--phase',phase]
   if phase=='residual':argv+=['--cpu-reuse']
   if phase=='gradient':argv+=['--cpu-gradient']
   logfile=OUT/'logs'/f'{phase}_{j["id"]}_{time.time_ns()}.log'
   with logfile.open('w') as f:proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   row=dict(job=j['id'],phase=phase,pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=gpu,log=str(logfile));active[j['id']]=dict(**row,process=proc);attempts.append(row);print(json.dumps(row),flush=True);break
  state.update(active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],attempts=attempts,waiting=[j['id'] for j in jobs if j['id'] not in done and j['id'] not in active]);save(statefile,state);time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  save(OUT/'supervisor_failure.json',dict(error=traceback.format_exc(),timestamp=stamp()));raise
