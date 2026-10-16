"""Bounded two-stage auxiliary study. GPU1/2/3 only, max4 total/max3 own processes per GPU."""
import os,sys,json,csv,subprocess,time,fcntl
from pathlib import Path
import train_v3_study as w
from analyze_aux_selection import select,final_jobs,summarize_final
ROOT=w.ROOT;OUT=ROOT/'outputs/aux_weight_selection_20261009'
def query(kind):
 fields='index,uuid,memory.total,memory.free' if kind=='gpu' else 'gpu_uuid,pid,process_name,used_memory'
 return [[x.strip() for x in r] for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-'+('gpu' if kind=='gpu' else 'compute-apps')+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()) if r]
def main():
 lock=(OUT/'study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((OUT/'protocol.json').read_text());smoke=json.loads((OUT/'gpu_smoke/result.json').read_text());assert smoke['status']=='PASS'
 assert json.loads((OUT/'gradient_diagnostic.json').read_text())['status']=='PASS'
 for name,h in protocol['code_sha256'].items():assert w.sha(ROOT/name)==h
 statefile=OUT/'study_state.json';previous=json.loads(statefile.read_text()) if statefile.exists() else {};active={};attempts=previous.get('attempts',[]);failures=previous.get('failed',[])
 assert not failures,'Prior failed jobs require diagnosis; no automatic retry'
 for row in previous.get('active',[]):
  q=Path('/proc')/str(row['pid'])
  if q.exists() and q.stat().st_uid==os.getuid() and q.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks']:active[row['job']]=dict(row,process=None)
 peak=smoke['peak_reserved_mib'];reserve=4096
 while True:
  selection=json.loads((OUT/'selection.json').read_text()) if (OUT/'selection.json').exists() else None
  stage='final' if selection else 'pilot';jobs=final_jobs(protocol,selection) if selection else [j for j in protocol['jobs'] if j['pilot']]
  done=[]
  for j in jobs:
   sf=Path(j['run_dir'])/'status.json'
   if sf.exists():
    s=json.loads(sf.read_text())
    if s['status'] in (['trained'] if stage=='final' else ['pilot_complete','trained']):
     cfg=json.loads((Path(j['run_dir'])/'config.json').read_text());assert cfg['protocol_sha256']==w.digest(protocol);done.append(j['id'])
  for jid,row in list(active.items()):
   proc=row['process'];q=Path('/proc')/str(row['pid']);dead=not q.exists() or q.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
   rc=proc.poll() if proc else (0 if dead else None)
   if rc is not None:
    del active[jid]
    if rc!=0 or jid not in done:failures.append(dict(job=jid,stage=row['stage'],returncode=rc))
  state=dict(timestamp=w.stamp(),stage=stage,planned_stage=len(jobs),completed_stage=len(done),failed=failures,attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],gpu_ids=[1,2,3],max_total_processes_per_gpu=4,selected_weight=selection['selected_weight'] if selection else None)
  w.save(statefile,state)
  if failures:raise RuntimeError(failures)
  if len(done)==len(jobs) and not active:
   if stage=='pilot':
    selection=select(OUT);w.log('Auxiliary pilot Validation selection: '+str(selection['selected_weight'])+'; '+selection['claim']);continue
   result=summarize_final(OUT,require_complete=True);state['stage']='completed';w.save(statefile,state);w.log('Auxiliary selected-weight four-arm comparison complete: '+str(result['comparisons']));return
  if stage=='final':summarize_final(OUT)
  for j in jobs:
   if j['id'] in done or j['id'] in active:continue
   inventory=[r for r in query('gpu') if r[0] in ['1','2','3']];apps=query('apps');owned={str(v['pid']) for v in active.values()};choices=[]
   for gpu in inventory:
    other=[r for r in apps if r[0]==gpu[1] and r[1] not in owned]
    # Never share with another user; same user's existing jobs remain untouched.
    try:safe=all(r[2]=='yeonwoo' and Path('/proc/'+r[1]).stat().st_uid==os.getuid() for r in other)
    except FileNotFoundError:continue
    if not safe:continue
    count=sum(v['gpu']==gpu[0] for v in active.values())
    if len(other)+count>=4 or count>=3:continue
    if int(gpu[3])<reserve+peak*1.2:continue
    # Account for workers launched but not visible in NVML yet.
    pending=sum(v['gpu']==gpu[0] and not any(r[1]==str(v['pid']) for r in apps) for v in active.values())
    if int(gpu[3])-pending*peak*1.2<reserve+peak*1.2:continue
    choices.append((len(other)+count,-int(gpu[3]),gpu[0],gpu))
   if not choices:break
   gpu=min(choices)[3];env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu[0],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
    d=OUT/'runtime'/str(j['id'])/k;d.mkdir(parents=True,exist_ok=True);env[k]=str(d)
   logfile=OUT/f'{stage}_{j["id"]}_{time.time_ns()}.log'
   with logfile.open('w') as f:proc=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/train_aux_selection.py'),'--study',str(OUT),'--job',str(j['id']),'--stage',stage],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   row=dict(job=j['id'],stage=stage,pid=proc.pid,gpu=gpu[0],start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],log=str(logfile));attempts.append(row);active[j['id']]=dict(row,process=proc);print(json.dumps(row),flush=True)
  state.update(attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()]);w.save(statefile,state);time.sleep(20)
if __name__=='__main__':main()
