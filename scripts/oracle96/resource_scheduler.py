"""Adopt running representative workers, then schedule 3 workers/GPU on7/8/9."""
from mixed_budget_common import *
import signal,traceback
from run_all import status,matched_gate
HERE=Path(__file__).parent

def identity(pid):
 try:
  x=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,state=x[0],ppid=int(x[1]),start_ticks=x[19],exit_status=int(x[-1]))
 except FileNotFoundError:return None

def capacity(used,total,apps,registered,reserved_new,slots):
 return slots<3 and set(apps)<=set(registered) and total-used-reserved_new>=14336+2048

def eligible(gpu,running):
 rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()));g=next(r for r in rows if int(r[0])==gpu);uuid=g[1].strip()
 apps=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()));appids={int(r[1]) for r in apps if r[0].strip()==uuid}
 registered={r['pid'] for r in running if r['gpu']==gpu};reserved=sum(14336 for r in running if r['gpu']==gpu and r['pid'] not in appids)
 return capacity(int(g[2]),int(g[3]),appids,registered,reserved,len(registered))

def snapshot_state(name,running,queue,done):
 records=[{k:r[k] for k in ['gpu','job','pid','start_ticks','adopted']} for r in running]
 save(STUDY/'resource_leases.json',dict(running=records,timestamp=stamp()))
 save(STUDY/'training_status.json',dict(status='running',stage=name,running=records,queued=queue,completed_this_stage=done,representative_gpus=[6,7,8,9],later_gpus=[7,8,9],max_processes_per_gpu=1 if name=='representative' else 3,timestamp=stamp()))

def stage(p,name,ids,phase,adopted=None):
 verify_code(p);guard_hashes('resource_launch_'+name)
 done_status='trained' if phase=='train' else 'completed';running=list(adopted or []);done=[]
 queue=[i for i in ids if status(p['jobs'][i],phase)!=done_status and i not in [r['job'] for r in running]]
 assert all(status(p['jobs'][i],phase)=='not_started' for i in queue),'Failure/interruption requires inspection, no automatic retries'
 try:
  while queue or running:
   for r in list(running):
    ident=identity(r['pid'])
    if ident and ident['start_ticks']!=r['start_ticks']:raise RuntimeError('PID reuse detected')
    rc=r['proc'].poll() if r.get('proc') else (None if ident and ident['state']!='Z' else 0 if ident and ident['exit_status']==0 else -1)
    if rc is not None:
     if r.get('fh'):r['fh'].close()
     running.remove(r)
     if rc!=0 or status(p['jobs'][r['job']],phase)!=done_status:raise RuntimeError(f'Job{r["job"]} failed rc={rc}; no automatic expansion')
     done.append(r['job'])
   for gpu in ([6] if name=='representative' else [7,8,9]):
    limit=1 if name=='representative' else 3
    while queue and sum(r['gpu']==gpu for r in running)<limit and eligible(gpu,running):
     i=queue.pop(0);j=p['jobs'][i];lp=STUDY/'logs'/f'resource_{phase}_{j["name"]}_{time.time_ns()}.log';fh=lp.open('x')
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
     # Python imports take time; a start gate guarantees the lease exists before CUDA.
     gate=STUDY/'runtime'/f'worker_start_{i}_{time.time_ns()}.gate'
     env['ORACLE_START_GATE']=str(gate)
     proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'resource_worker.py'),'--job',str(i),'--phase',phase],env=env,stdout=fh,stderr=subprocess.STDOUT)
     ident=identity(proc.pid);assert ident is not None
     running.append(dict(gpu=gpu,job=i,pid=proc.pid,start_ticks=ident['start_ticks'],adopted=False,proc=proc,fh=fh))
     snapshot_state(name,running,queue,done);gate.write_text('ready\n')
     save(STUDY/f'launch_resource_{phase}_{i}_{proc.pid}.json',dict(job=i,gpu=gpu,pid=proc.pid,log=str(lp),timestamp=stamp()));log(f'resource scheduler {name}: {j["name"]} GPU{gpu} PID{proc.pid}')
   snapshot_state(name,running,queue,done);time.sleep(2)
 except BaseException:
  for r in running:
   ident=identity(r['pid'])
   if ident and ident['start_ticks']==r['start_ticks'] and ident['state']!='Z':os.kill(r['pid'],signal.SIGTERM)
  raise
 save(STUDY/f'stage_{name}.json',dict(status='PASS',ids=list(ids),timestamp=stamp()))

def main():
 locks=[];policy=json.loads((STUDY/'resource_policy.json').read_text());old=policy['suspended_controller'];released=False
 try:
  for path,h in policy['source_sha256'].items():assert sha(path)==h,path
  f=(STUDY/'runtime/resource_controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  oldid=identity(old['pid']);assert oldid and oldid['start_ticks']==old['start_ticks'] and oldid['state']=='T'
  # Original manager holds7/8/9 locks while suspended; keep its PTY and children intact.
  for root in [STUDY,ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011']:
   f=(root/'runtime/gpu6.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  p=protocol();verify_code(p);adopted=policy['adopted_workers']
  for r in adopted:
   ident=identity(r['pid']);assert ident and ident['start_ticks']==r['start_ticks'] and ident['ppid']==old['pid']
  stage(p,'representative',range(4),'train',adopted);matched_gate(p,range(4),'representative_report.json')
  # All original GPU children have exited; retiring only their suspended manager is now safe.
  oldid=identity(old['pid']);assert oldid and oldid['start_ticks']==old['start_ticks'];os.kill(old['pid'],signal.SIGKILL);released=True
  for f in locks[1:]:f.close()
  locks=locks[:1]
  for gpu in [7,8,9]:
   for root in [STUDY,ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011']:
    f=(root/'runtime'/f'gpu{gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  log('대표4조건 검증 PASS, GPU6 해제. GPU7/8/9 각최대3 worker로 확대; 기존 과학 설정 유지.')
  stage(p,'remaining_10_lead7_seeds',range(4,12),'train');matched_gate(p,range(12),'seed_gate_report.json')
  stage(p,'remaining_84',range(12,96),'train');matched_gate(p,range(96),'all_training_gate_report.json')
  stage(p,'evaluation',range(96),'evaluate')
  save(STUDY/'training_status.json',dict(status='CPU_summary_running',timestamp=stamp()))
  with (STUDY/'logs/final_summary.log').open('x') as f:subprocess.run([str(PYTHON),'-B',str(HERE/'summarize_mixed_budget_gpu.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
  guard_hashes('all_done');save(STUDY/'training_status.json',dict(status='completed',trained=96,evaluated=192,timestamp=stamp()));log('96조건 학습·평가·보고 완료')
 except BaseException:
  save(STUDY/'training_status.json',dict(status='failed_or_interrupted',error=traceback.format_exc(),old_controller_suspended=not released,timestamp=stamp()));log('resource scheduler 실패, 자동 확대/재시도 금지; 기존 관리자는 재개하지 않음.');raise
 finally:
  for f in locks:f.close()
if __name__=='__main__':main()
