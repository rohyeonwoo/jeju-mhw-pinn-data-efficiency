from eval_common import *
import controller as C
import signal,traceback
HERE=Path(__file__).parent

def publish(running,queue,status='RUNNING',error=None):
 clean=[{k:r[k] for k in ['job','gpu','pid','start_ticks','log']} for r in running]
 save(EVAL/'leases.json',dict(running=clean,timestamp=stamp()))
 completed=sum(json.loads(f.read_text()).get('status')=='COMPLETED' for f in (EVAL/'runs').glob('*/status.json'))
 save(EVAL/'evaluation_state.json',dict(status=status,completed=completed,total=96,controller_pid=os.getpid(),running=clean,queue=queue,error=error,timestamp=stamp()))

def stage(p,ids):
 queue=[];running=[]
 for i in ids:
  f=EVAL/'runs'/p['jobs'][i]['name']/'status.json'
  state=json.loads(f.read_text())['status'] if f.exists() else 'PENDING'
  if state=='COMPLETED':continue
  assert state=='PENDING',f'{i}: {state}; inspect failure before retry'
  queue.append(i)
 try:
  while queue or running:
   for r in list(running):
    rc=r['proc'].poll()
    if rc is not None:
     r['fh'].close();running.remove(r)
     assert rc==0 and json.loads((EVAL/'runs'/p['jobs'][r['job']]['name']/'status.json').read_text())['status']=='COMPLETED',f'job {r["job"]} failed: {r["log"]}'
   rows,apps=C.resource_rows()
   for gpu in sorted([7,8,9],key=lambda g:(sum(r['gpu']==g for r in running),g)):
    if queue and C.eligible(gpu,running,rows,apps):
     i=queue.pop(0);j=p['jobs'][i];lp=EVAL/'logs'/f'{j["name"]}_{time.time_ns()}.log';fh=lp.open('x');gate=EVAL/'runtime'/f'gate_{i}_{time.time_ns()}'
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',EVAL_START_GATE=str(gate))
     proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'evaluate.py'),'--job',str(i)],env=env,stdout=fh,stderr=subprocess.STDOUT)
     ident=C.ident(proc.pid);assert ident
     running.append(dict(job=i,gpu=gpu,pid=proc.pid,start_ticks=ident['start_ticks'],log=str(lp),proc=proc,fh=fh))
     publish(running,queue);gate.write_text('ready\n')
     save(EVAL/'launches'/f'{i}_{proc.pid}.json',dict(job=i,name=j['name'],pid=proc.pid,gpu=gpu,log=str(lp),timestamp=stamp()))
     log(f'Test inference launch {j["name"]} GPU{gpu} PID{proc.pid}; no training')
   publish(running,queue)
   f=EVAL/'resource_usage.csv';new=not f.exists()
   with f.open('a') as stream:
    w=csv.writer(stream)
    if new:w.writerow(['time','gpu','all_users_used_mib','total_mib','our_processes'])
    for gpu in [7,8,9]:w.writerow([stamp(),gpu,rows[gpu]['used'],rows[gpu]['total'],sum(r['gpu']==gpu for r in running)])
   time.sleep(3)
 except BaseException:
  for r in running:
   if r['proc'].poll() is None:r['proc'].terminate()
  for r in running:
   try:r['proc'].wait(timeout=20)
   except subprocess.TimeoutExpired:r['proc'].kill();r['proc'].wait()
   r['fh'].close()
  raise
 finally:save(EVAL/'leases.json',dict(running=[],timestamp=stamp()))

def main():
 locks=[]
 def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
 signal.signal(signal.SIGTERM,stop)
 try:
  for root in [EVAL,TRAIN,OLD_STUDY]:
   for name in ['controller.lock','resource_controller.lock','shared_controller.lock']:locks.append(C.acquire(root/'runtime'/name))
  for root in [TRAIN,OLD_STUDY,ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011']:
   for gpu in [7,8,9]:locks.append(C.acquire(root/'runtime'/f'gpu{gpu}.lock'))
  for d in ['logs','runs','launches']:(EVAL/d).mkdir(parents=True,exist_ok=True)
  p=evaluation_protocol();verify_evaluation(p);assert json.loads((EVAL/'preflight.json').read_text())['status']=='PASS'
  stage(p,range(4))
  subprocess.run([str(PYTHON),'-B',str(HERE/'summarize_test.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  stage(p,range(4,96))
  subprocess.run([str(PYTHON),'-B',str(HERE/'summarize_test.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  publish([],[],status='COMPLETED');log('Test 96/96 SST/MHW 평가 및 forcing 불변성 검사 완료; 별도 report_ko.md/comparison_summary.csv 생성.')
 except BaseException:
  publish([],[],status='FAILED',error=traceback.format_exc());log('Test evaluation queue stopped; failure evidence preserved, no blind retry');raise
 finally:
  for f in locks:f.close()
if __name__=='__main__':main()
