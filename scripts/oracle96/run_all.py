"""Authorized gated 96-run controller. Stops on failure; never resumes old managers."""
from mixed_budget_common import *
import signal,traceback
HERE=Path(__file__).parent

def status(j,phase):
 d=STUDY/'smoke'/j['name'] if phase=='smoke' else Path(j['run_dir'])
 f=d/('evaluation_status.json' if phase=='evaluate' else 'status.json')
 return json.loads(f.read_text()).get('status') if f.exists() else 'not_started'

def stage(p,name,ids,phase):
 verify_code(p);guard_hashes('launch_'+name)
 done={'smoke':'PASS','train':'trained','evaluate':'completed'}[phase]
 queue=[i for i in ids if status(p['jobs'][i],phase)!=done]
 assert all(status(p['jobs'][i],phase)=='not_started' for i in queue),'Inspect failure/interruption before any retry'
 running={};completed=[]
 try:
  while queue or running:
   for gpu,(proc,i,fh) in list(running.items()):
    rc=proc.poll()
    if rc is not None:
     fh.close();del running[gpu]
     if rc!=0 or status(p['jobs'][i],phase)!=done:raise RuntimeError(f'{name}: job{i} rc={rc}; expansion STOPPED')
     completed.append(i)
   for gpu in [7,8,9]:
    if not queue or gpu in running:continue
    gpu_idle(gpu);i=queue.pop(0);j=p['jobs'][i];lp=STUDY/'logs'/f'{phase}_{j["name"]}_{time.time_ns()}.log';fh=lp.open('x')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'train_mixed_budget_gpu.py'),'--job',str(i),'--phase',phase],env=env,stdout=fh,stderr=subprocess.STDOUT)
    running[gpu]=(proc,i,fh);save(STUDY/f'launch_{phase}_{i}_{proc.pid}.json',dict(job=i,gpu=gpu,pid=proc.pid,log=str(lp),timestamp=stamp()));log(f'{name}: launch {j["name"]} GPU{gpu} PID{proc.pid}')
   save(STUDY/'training_status.json',dict(status='running',stage=name,running=[dict(gpu=g,job=i,name=p['jobs'][i]['name'],pid=pr.pid) for g,(pr,i,fh) in running.items()],queued=queue,completed_this_stage=completed,timestamp=stamp()))
   time.sleep(2)
 except BaseException:
  for pr,i,fh in running.values():
   if pr.poll() is None:pr.terminate()
  for pr,i,fh in running.values():
   try:pr.wait(timeout=15)
   except subprocess.TimeoutExpired:pr.kill();pr.wait()
   fh.close()
  raise
 save(STUDY/f'stage_{name}.json',dict(status='PASS',ids=list(ids),timestamp=stamp()))

def matched_gate(p,ids,path,smoke=False):
 jobs=[p['jobs'][i] for i in ids]
 configs=[]
 for j in jobs:
  d=STUDY/'smoke'/j['name'] if smoke else Path(j['run_dir'])
  configs.append(json.loads((d/'config.json').read_text()))
  inv=json.loads((d/'forcing_invariance_val.json').read_text());assert inv['status']=='PASS'
  if not smoke:
   s=json.loads((d/'status.json').read_text());assert s['steps']==38240 and s['epochs']==80 and not s['early_stopping']
 # Compare only within an identical ratio/lead/seed group.
 for key in set((j['ratio'],j['lead'],j['seed']) for j in jobs):
  cs=[c for j,c in zip(jobs,configs) if (j['ratio'],j['lead'],j['seed'])==key]
  for field in ['shared_initial_sha256','stats','split_sha256','tensor_fingerprints','parameters','rhs_scale']:
   assert all(c[field]==cs[0][field] for c in cs),field
  if not smoke:
   histories=[json.loads((Path(j['run_dir'])/'history.json').read_text()) for j in jobs if (j['ratio'],j['lead'],j['seed'])==key]
   assert all(len(h)==80 for h in histories)
   for epoch in range(80):assert len({h[epoch]['batch_order_sha256'] for h in histories})==1
 save(STUDY/path,dict(status='PASS',ids=list(ids),setup_matched=True,forcing_invariance='PASS',fixed_updates=not smoke,timestamp=stamp()))

def main():
 locks=[]
 try:
  (STUDY/'logs').mkdir(exist_ok=True)
  f=(STUDY/'runtime/controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  p=protocol();verify_code(p);pre=json.loads((STUDY/'preflight_report.json').read_text());assert pre['status']=='PASS' and pre['protocol_sha256']==digest(p)
  for gpu in [7,8,9]:
   for root in [STUDY,ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011']:
    f=(root/'runtime'/f'gpu{gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
   gpu_idle(gpu)
  record=ROOT/'outputs/v4_cpu_physics_audit_20261011/gpu_stop_record.json'
  if record.exists():
   for r in json.loads(record.read_text())['processes']:
    f=Path('/proc')/str(r['pid'])/'stat'
    if f.exists():
     s=f.read_text().rsplit(')',1)[1].split();assert s[19]!=r['start_ticks'] or s[0]=='Z','Old adaptive process still active'
  save(STUDY/'resources_at_launch.json',snapshot())
  stage(p,'smoke',range(4),'smoke');matched_gate(p,range(4),'smoke_test_report.json',True)
  stage(p,'representative',range(4),'train');matched_gate(p,range(4),'representative_report.json')
  stage(p,'remaining_10_lead7_seeds',range(4,12),'train');matched_gate(p,range(12),'seed_gate_report.json')
  stage(p,'remaining_84',range(12,96),'train');matched_gate(p,range(96),'all_training_gate_report.json')
  stage(p,'evaluation',range(96),'evaluate')
  save(STUDY/'training_status.json',dict(status='CPU_summary_running',timestamp=stamp()))
  with (STUDY/'logs/final_summary.log').open('x') as f:subprocess.run([str(PYTHON),'-B',str(HERE/'summarize_mixed_budget_gpu.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
  guard_hashes('all_done');save(STUDY/'training_status.json',dict(status='completed',trained=96,evaluated=192,timestamp=stamp()));log('96조건 학습·독립 예측/사후진단·SST/MHW/통계 보고 완료')
 except BaseException:
  save(STUDY/'training_status.json',dict(status='failed_or_interrupted',error=traceback.format_exc(),timestamp=stamp()));log('96조건 관리자 중단: 자동 재시도/확대 금지. training_status.json 및 개별 로그 점검 필요.');raise
 finally:
  for f in locks:f.close()
if __name__=='__main__':main()
