"""Detached bounded GPU queue: CPU PASS -> four smokes -> four representatives -> 92."""
from mixed_budget_common import *
import signal,traceback
HERE=Path(__file__).parent

def ident(pid):
 try:
  v=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,state=v[0],start_ticks=v[19])
 except FileNotFoundError:return None

def job_status(j):
 f=Path(j['run_dir'])/'status.json';return json.loads(f.read_text()) if f.exists() else dict(status='PENDING')

def gate(p,ids,name,smoke=False):
 cs=[];hs=[]
 for i in ids:
  j=p['jobs'][i];d=STUDY/'smoke'/j['name'] if smoke else Path(j['run_dir'])
  c=json.loads((d/'config.json').read_text());s=json.loads((d/'status.json').read_text());inv=json.loads((d/'forcing_invariance_val.json').read_text());assert inv['status']=='PASS'
  assert s['status']==('PASS' if smoke else 'COMPLETED')
  if not smoke:
   h=json.loads((d/'history.json').read_text());assert len(h)==80 and s['steps']==j['total_updates'] and s['validation_selection_calls']==80
   for epoch,r in enumerate(h,1):
    assert r['actual_epoch']==epoch and r['train_samples']==j['n_train'] and r['epoch_updates']==j['updates_per_epoch'] and r['steps']==epoch*j['updates_per_epoch'] and r['sampling_exact_once']
    assert abs(r['lambda_phys_effective']-j['lambda_physics']*min(epoch/5,1))<1e-12
   assert s['post_warmup_best_epoch']>=5 and s['checkpoint_roundtrip']=='PASS';hs.append((j,h))
  cs.append((j,c))
 for key in {(j['ratio'],j['lead'],j['seed']) for j,c in cs}:
  group=[c for j,c in cs if (j['ratio'],j['lead'],j['seed'])==key]
  for k in ['shared_initial_sha256','stats','split_sha256','tensor_fingerprints','parameters','rhs_scale']:
   assert all(c[k]==group[0][k] for c in group),k
  if not smoke:
   group=[h for j,h in hs if (j['ratio'],j['lead'],j['seed'])==key]
   assert all([r['sample_order_sha256'] for r in h]==[r['sample_order_sha256'] for r in group[0]] for h in group)
 save(STUDY/name,dict(status='PASS',ids=list(ids),same_inputs_initialization_order=True,timestamp=stamp()))

def resource_rows():
 rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()))
 apps=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()))
 return {int(r[0]):dict(uuid=r[1].strip(),used=int(r[2]),total=int(r[3])) for r in rows},apps

def eligible(gpu,running,rows,apps):
 ours=[r for r in running if r['gpu']==gpu];mem={int(x[1]):int(x[2]) for x in apps if x[0].strip()==rows[gpu]['uuid']}
 reserve=sum(max(0,14336-mem.get(r['pid'],0)) for r in ours)
 return len(ours)<2 and rows[gpu]['total']-rows[gpu]['used']-reserve>=14336+4096

def publish(p,stage,running,queue):
 clean=[{k:r[k] for k in ['gpu','job','pid','start_ticks','phase','log']} for r in running]
 save(STUDY/'leases.json',dict(running=clean,timestamp=stamp()))
 states=[dict(job=j['id'],name=j['name'],**job_status(j)) for j in p['jobs']]
 save(STUDY/'run_state.json',dict(stage=stage,status='RUNNING',controller_pid=os.getpid(),running=clean,queue=queue,runs=states,completed=sum(x['status']=='COMPLETED' for x in states),timestamp=stamp()))

def stage(p,name,ids,phase):
 verify_code(p);running=[];queue=[]
 for i in ids:
  j=p['jobs'][i];f=(STUDY/'smoke'/j['name']/'status.json') if phase=='smoke' else Path(j['run_dir'])/'status.json'
  status=json.loads(f.read_text())['status'] if f.exists() else 'PENDING'
  if status==('PASS' if phase=='smoke' else 'COMPLETED'):continue
  assert status=='PENDING',f'{j["name"]}: {status}; manual diagnosis required, no automatic retry'
  queue.append(i)
 try:
  while queue or running:
   for r in list(running):
    rc=r['proc'].poll()
    if rc is not None:
     r['fh'].close();running.remove(r)
     d=STUDY/'smoke'/p['jobs'][r['job']]['name'] if phase=='smoke' else Path(p['jobs'][r['job']]['run_dir'])
     assert rc==0 and json.loads((d/'status.json').read_text())['status']==('PASS' if phase=='smoke' else 'COMPLETED'),f'job {r["job"]} failed rc={rc}; expansion stopped'
   rows,apps=resource_rows()
   for gpu in sorted([7,8,9],key=lambda g:(sum(r['gpu']==g for r in running),g)):
    if queue and eligible(gpu,running,rows,apps):
     i=queue.pop(0);j=p['jobs'][i];lp=STUDY/'logs'/f'{phase}_{j["name"]}_{time.time_ns()}.log';fh=lp.open('x');gatepath=STUDY/'runtime'/f'gate_{phase}_{i}_{time.time_ns()}'
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',ACTUAL_START_GATE=str(gatepath))
     proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'worker.py'),'--job',str(i),'--phase',phase],env=env,stdout=fh,stderr=subprocess.STDOUT)
     x=ident(proc.pid);assert x
     running.append(dict(gpu=gpu,job=i,pid=proc.pid,start_ticks=x['start_ticks'],phase=phase,log=str(lp),proc=proc,fh=fh));publish(p,name,running,queue);gatepath.write_text('ready\n')
     save(STUDY/'launches'/f'{phase}_{i}_{proc.pid}.json',dict(job=i,pid=proc.pid,gpu=gpu,log=str(lp),timestamp=stamp()));log(f'queue launch {j["name"]} phase={phase} GPU{gpu} PID{proc.pid}')
   publish(p,name,running,queue)
   resource=STUDY/'resource_usage.csv';new=not resource.exists()
   with resource.open('a') as f:
    writer=csv.writer(f)
    if new:writer.writerow(['timestamp','stage','gpu','used_mib_all_users','total_mib','our_live_processes'])
    for gpu in [7,8,9]:writer.writerow([stamp(),name,gpu,rows[gpu]['used'],rows[gpu]['total'],sum(r['gpu']==gpu for r in running)])
   time.sleep(5)
 except BaseException:
  for r in running:
   if r['proc'].poll() is None:r['proc'].terminate()
  for r in running:
   try:r['proc'].wait(timeout=20)
   except subprocess.TimeoutExpired:r['proc'].kill();r['proc'].wait()
   r['fh'].close()
  save(STUDY/'leases.json',dict(running=[],timestamp=stamp()));raise
 save(STUDY/'leases.json',dict(running=[],timestamp=stamp()))

def acquire(path):
 path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a');end=time.monotonic()+30
 while True:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
  except BlockingIOError:
   if time.monotonic()>end:raise RuntimeError(f'Controller/GPU lock busy: {path}')
   time.sleep(.2)

def main():
 locks=[]
 def stop(sig,frame):raise KeyboardInterrupt(f'pause requested by signal {sig}')
 signal.signal(signal.SIGTERM,stop)
 try:
  for root in [OLD_STUDY,STUDY]:
   for name in ['controller.lock','resource_controller.lock','shared_controller.lock']:
    locks.append(acquire(root/'runtime'/name))
  for root in [STUDY,OLD_STUDY,ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011']:
   for gpu in [7,8,9]:locks.append(acquire(root/'runtime'/f'gpu{gpu}.lock'))
  p=protocol();verify_code(p);assert json.loads((STUDY/'preflight.json').read_text())['status']=='PASS'
  save(STUDY/'controller_pid.json',dict(pid=os.getpid(),timestamp=stamp()))
  stage(p,'representative_smoke',range(4),'smoke');gate(p,range(4),'smoke_gate.json',True)
  stage(p,'representative_training',range(4),'train');gate(p,range(4),'representative_gate.json')
  subprocess.run([str(PYTHON),'-B',str(HERE/'summarize.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  stage(p,'remaining_92',range(4,96),'train');gate(p,range(96),'all96_gate.json')
  subprocess.run([str(PYTHON),'-B',str(HERE/'summarize.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  save(STUDY/'run_state.json',dict(status='COMPLETED',completed=96,test_evaluation='NOT_RUN',runs=[dict(job=j['id'],name=j['name'],**job_status(j)) for j in p['jobs']],timestamp=stamp()))
  log('Actual epoch 96 training/Validation complete; Test intentionally NOT_RUN; report.md generated')
 except BaseException as e:
  save(STUDY/'run_state.json',dict(status='PAUSED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=traceback.format_exc(),timestamp=stamp()))
  log('Actual epoch controller paused/failed; no automatic expansion/retry');raise
 finally:
  for f in locks:f.close()
if __name__=='__main__':main()
