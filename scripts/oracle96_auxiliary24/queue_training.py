from aux_common import *
import controller as C
import signal,traceback

def status(j,phase):
 d=AUX/'smoke'/j['name'] if phase=='smoke' else Path(j['run_dir']);f=d/'status.json'
 return json.loads(f.read_text()) if f.exists() else dict(status='PENDING')
def publish(p,running,queue,state='RUNNING',error=None):
 clean=[{k:r[k] for k in ['job','gpu','pid','start_ticks','phase','log']} for r in running]
 save(AUX/'leases.json',dict(running=clean,timestamp=stamp()))
 states=[dict(name=j['name'],**status(j,'train')) for j in p['jobs']]
 save(AUX/'run_state.json',dict(status=state,completed=sum(s['status']=='COMPLETED' for s in states),total=24,running=clean,queue=queue,runs=states,controller_pid=os.getpid(),gpus=GPUS,test_evaluation='NOT_RUN',error=error,timestamp=stamp()))
def summarize():subprocess.run([str(PYTHON),'-B',str(HERE/'summarize_training.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
def stage(p,ids,phase):
 running=[];queue=[];attempts={};ready={}
 for i in ids:
  s=status(p['jobs'][i],phase)['status']
  if s==('PASS' if phase=='smoke' else 'COMPLETED'):continue
  assert s in ['PENDING','PAUSED'],f'Inspect existing failure before resume: job{i} {s}'
  queue.append(i)
 try:
  while queue or running:
   changed=False
   for r in list(running):
    rc=r['proc'].poll()
    if rc is not None:
     r['fh'].close();running.remove(r);s=status(p['jobs'][r['job']],phase)
     if rc!=0 or s['status']!=('PASS' if phase=='smoke' else 'COMPLETED'):
      error=s.get('error','');transient=any(x in error.lower() for x in ['out of memory','input/output error'])
      if phase=='train' and transient and attempts.get(r['job'],0)<2:
       queue.append(r['job']);ready[r['job']]=time.monotonic()+60;journal(f'Transient failure job{r["job"]}; one same-config epoch-boundary retry queued. {r["log"]}')
      else:raise RuntimeError(f'{phase} job{r["job"]} failed; see {r["log"]}')
     changed=True
   if changed and phase=='train':summarize()
   rows,apps=C.resource_rows()
   for gpu in sorted(GPUS,key=lambda g:(sum(r['gpu']==g for r in running),g)):
    available=[i for i in queue if time.monotonic()>=ready.get(i,0)]
    if available and C.eligible(gpu,running,rows,apps):
     i=available[0];queue.remove(i);j=p['jobs'][i];attempts[i]=attempts.get(i,0)+1
     lp=AUX/'logs'/f'{phase}_{j["name"]}_{time.time_ns()}.log';fh=lp.open('x');gate=AUX/'runtime'/f'gate_{phase}_{i}_{time.time_ns()}'
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',AUX_START_GATE=str(gate))
     proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'aux_worker.py'),'--job',str(i),'--phase',phase],env=env,stdout=fh,stderr=subprocess.STDOUT)
     ident=C.ident(proc.pid);assert ident
     running.append(dict(job=i,gpu=gpu,pid=proc.pid,start_ticks=ident['start_ticks'],phase=phase,log=str(lp),proc=proc,fh=fh));publish(p,running,queue);gate.write_text('ready\n')
     save(AUX/'launches'/f'{phase}_{i}_{proc.pid}.json',dict(job=i,gpu=gpu,pid=proc.pid,log=str(lp),attempt=attempts[i],timestamp=stamp()));journal(f'launch {phase} {j["name"]} GPU{gpu} PID{proc.pid}')
   publish(p,running,queue)
   with (AUX/'gpu_resource_log.csv').open('a') as f:
    w=csv.writer(f)
    for g in GPUS:w.writerow([stamp(),g,rows[g]['used'],rows[g]['total'],sum(x['gpu']==g for x in running),phase])
   time.sleep(5)
 except BaseException:
  for r in running:
   if r['proc'].poll() is None:r['proc'].terminate()
  for r in running:
   try:r['proc'].wait(timeout=20)
   except subprocess.TimeoutExpired:r['proc'].kill();r['proc'].wait()
   r['fh'].close()
  raise
 finally:save(AUX/'leases.json',dict(running=[],timestamp=stamp()))
def main():
 locks=[]
 def stop(sig,frame):raise KeyboardInterrupt(f'signal{sig}')
 signal.signal(signal.SIGTERM,stop)
 try:
  for root in [AUX,ORACLE,OLD_STUDY]:
   for name in ['controller.lock','resource_controller.lock','shared_controller.lock']:locks.append(C.acquire(root/'runtime'/name))
  for root in [AUX,ORACLE,OLD_STUDY,ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011']:
   for gpu in GPUS:locks.append(C.acquire(root/'runtime'/f'gpu{gpu}.lock'))
  p=plan();verify(p);assert json.loads((AUX/'cpu_preflight.json').read_text())['status']=='PASS'
  stage(p,[0],'smoke')
  evidence=json.loads((AUX/'smoke'/p['jobs'][0]['name']/'diagnostic_noninterference.json').read_text());assert evidence['status']=='PASS'
  save(AUX/'smoke_gate.json',dict(status='PASS',diagnostic_noninterference='PASS',timestamp=stamp()))
  (AUX/'preflight.md').write_text('# Preflight PASS\n\nzero-head 최소해는 사용자 지시로 비차단 한계로 재분류. 기존 PASS 보존. CPU24×80epoch sampling/LR 일치 PASS. GPU 대표실행: 독립 초기tensor/원SHA/통계/분할 일치 PASS, 진단 전후 state/RNG/.grad/mode 보존 PASS, 진단ON/OFF 후 실제1step model/optimizer bitwise 일치 PASS, checkpoint roundtrip PASS. 각 run은 동일 초기값/통계/순서를 runtime에서 다시 검사한다. Test 금지.\n')
  stage(p,range(24),'train');summarize();publish(p,[],[],state='COMPLETED');journal('24/24 Auxiliary-only 80actual epochs 완료, Validation-only 보고서 집계. Test NOT_RUN.')
 except BaseException as e:
  p=plan();publish(p,[],[],state='PAUSED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=traceback.format_exc());journal('Auxiliary24 manager interrupted/failed; evidence preserved');raise
 finally:
  for f in locks:f.close()
if __name__=='__main__':main()
