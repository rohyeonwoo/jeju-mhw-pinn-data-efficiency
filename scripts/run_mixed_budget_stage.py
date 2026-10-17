"""Explicit manual stages, never launches later stages or restarts a failed job."""
from mixed_budget_common import *
import argparse,signal,traceback

def status(job,phase='train'):
 d=STUDY/'smoke'/job['name'] if phase=='smoke' else Path(job['run_dir'])
 p=d/('evaluation_status.json' if phase=='evaluate' else 'status.json')
 return json.loads(p.read_text()).get('status') if p.exists() else 'not_started'

def gate(p,stage):
 pre=json.loads((STUDY/'preflight_report.json').read_text());assert pre['status']=='CPU_PASS_GPU_PENDING' and pre['protocol_sha256']==digest(p)
 if stage in ['representative','remaining','evaluate']:
  assert all(status(j,'smoke')=='PASS' for j in p['jobs'][:4]),'Run smoke stage first'
  assert json.loads((STUDY/'smoke_test_report.json').read_text())['status']=='PASS'
 if stage in ['remaining','evaluate']:
  assert all(status(j)=='trained' for j in p['jobs'][:4]),'All four representative trainings must finish'
  assert json.loads((STUDY/'representative_report.json').read_text())['status']=='PASS'
 if stage=='evaluate':assert all(status(j)=='trained' for j in p['jobs']),'All24 matched trainings must finish before Test'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','representative','remaining','evaluate'],required=True);o=ap.parse_args()
 p=protocol();verify_code(p);gate(p,o.stage)
 guard_hashes('launch_'+o.stage)
 features=json.loads((FEATURES/'manifest.json').read_text())
 assert sha(FEATURES/'manifest.json')==p['feature_manifest_sha256']
 for path,h in features['sha256'].items():assert sha(path)==h,path
 locks=[]
 for gpu in [0,1,2]:
  f=(STUDY/'runtime'/f'gpu{gpu}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f);gpu_idle(gpu)
 # Check old stopped processes by identity; do not revive supervisors or terminate any user's process.
 record=ROOT/'outputs/v4_cpu_physics_audit_20261011/gpu_stop_record.json'
 if record.exists():
  for r in json.loads(record.read_text())['processes']:
   f=Path('/proc')/str(r['pid'])/'stat'
   if f.exists():
    s=f.read_text().rsplit(')',1)[1].split();assert s[19]!=r['start_ticks'] or s[0]=='Z','Prior adaptive process still active'
 save(STUDY/f'resources_{o.stage}_{time.time_ns()}.json',snapshot())
 ids=list(range(4)) if o.stage in ['smoke','representative'] else list(range(4,24)) if o.stage=='remaining' else list(range(24))
 phase='smoke' if o.stage=='smoke' else 'evaluate' if o.stage=='evaluate' else 'train';done='PASS' if phase=='smoke' else 'completed' if phase=='evaluate' else 'trained'
 queue=[i for i in ids if status(p['jobs'][i],phase)!=done]
 # Never blindly resume an active or failed job. A bounded manual retry is a separate explicit worker command after inspection.
 assert all(status(p['jobs'][i],phase) not in ['training','failed'] for i in queue),'Inspect existing failed/interrupted job before retry'
 running={};completed=[]
 try:
  while queue or running:
   for gpu,(proc,i,fh) in list(running.items()):
    rc=proc.poll()
    if rc is not None:
     fh.close();del running[gpu]
     if rc!=0 or status(p['jobs'][i],phase)!=done:raise RuntimeError(f'Job {i} failed; rc={rc}; no automatic retry')
     completed.append(i)
   for gpu in [0,1,2]:
    if not queue or gpu in running:continue
    gpu_idle(gpu);i=queue.pop(0);j=p['jobs'][i];dest=STUDY/'logs';dest.mkdir(exist_ok=True);lp=dest/f'{phase}_{j["name"]}_{time.time_ns()}.log';fh=lp.open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    proc=subprocess.Popen([str(PYTHON),'-B',str(ROOT/'scripts/train_mixed_budget_gpu.py'),'--job',str(i),'--phase',phase],env=env,stdout=fh,stderr=subprocess.STDOUT)
    running[gpu]=(proc,i,fh);save(STUDY/f'launch_{phase}_{i}_{proc.pid}.json',dict(job=i,gpu=gpu,pid=proc.pid,log=str(lp),timestamp=stamp()))
   save(STUDY/'training_status.json',dict(stage=o.stage,status='running',running=[dict(gpu=g,job=i,pid=pr.pid) for g,(pr,i,fh) in running.items()],queued=queue,completed_this_stage=completed,timestamp=stamp()));time.sleep(2)
 except BaseException:
  # Only children started by this manager are stopped; never unrelated GPU work.
  for pr,i,fh in running.values():
   if pr.poll() is None:pr.terminate()
  for pr,i,fh in running.values():
   try:pr.wait(timeout=15)
   except subprocess.TimeoutExpired:pr.kill();pr.wait()
   fh.close()
  save(STUDY/'training_status.json',dict(status='failed_or_interrupted',stage=o.stage,error=traceback.format_exc(),timestamp=stamp()));raise
 finally:
  for f in locks:f.close()
 if phase=='smoke':
  reports=[json.loads((STUDY/'smoke'/j['name']/'status.json').read_text()) for j in p['jobs'][:4]]
  cfg=[json.loads((STUDY/'smoke'/j['name']/'config.json').read_text()) for j in p['jobs'][:4]]
  for k in ['shared_initial_sha256','stats','split_sha256','tensor_fingerprints','parameters','rhs_scale']:
   assert all(c[k]==cfg[0][k] for c in cfg),k
  save(STUDY/'smoke_test_report.json',dict(status='PASS',reports=reports,matched_setup=True,timestamp=stamp()))
 if o.stage=='representative':
  hs=[json.loads((Path(j['run_dir'])/'history.json').read_text()) for j in p['jobs'][:4]]
  for i in range(min(map(len,hs))):assert len({h[i]['batch_order_sha256'] for h in hs})==1
  save(STUDY/'representative_report.json',dict(status='PASS',models=4,batch_order_equal=True,timestamp=stamp()))
 save(STUDY/'training_status.json',dict(status='stage_complete',stage=o.stage,timestamp=stamp()));log('수동 stage 완료 '+o.stage+'; 다음 stage는 자동 시작하지 않음.');print('Stage complete:',o.stage)
if __name__=='__main__':main()
