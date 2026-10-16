"""Resource-policy-only launch adapter; the frozen scientific worker is unchanged."""
from mixed_budget_common import *
import argparse,traceback
import torch
import train_mixed_budget_gpu as w

def main():
 gate=Path(os.environ['ORACLE_START_GATE']);deadline=time.monotonic()+60
 while not gate.exists():
  assert time.monotonic()<deadline,'Missing resource lease';time.sleep(.05)
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);ap.add_argument('--phase',choices=['train','evaluate'],required=True);a=ap.parse_args()
 policy=json.loads((STUDY/'shared_resource_policy.json').read_text())
 for path,h in policy['source_sha256'].items():assert sha(path)==h,path
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in [7,8,9]
 p=protocol();verify_code(p);j=p['jobs'][a.job];d=Path(j['run_dir']);d.mkdir(parents=True,exist_ok=True)
 lease=json.loads((STUDY/'shared_resource_leases.json').read_text());assert any(r['pid']==os.getpid() and r['job']==a.job and r['gpu']==gpu for r in lease['running'])
 rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()));uuid=next(r[1].strip() for r in rows if int(r[0])==gpu)
 own={r['pid'] for r in lease['running'] if r['gpu']==gpu}
 apps=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()))
 # Shared GPU use explicitly authorized; never signal or change another user's process.
 assert sum(r['gpu']==gpu for r in lease['running'])<=2
 save(d/f'{a.phase}_pid.json',dict(pid=os.getpid(),gpu=gpu,phase=a.phase,resource_policy_sha256=sha(STUDY/'shared_resource_policy.json'),timestamp=stamp()))
 torch.set_num_threads(1);w.install();device=torch.device('cuda:0')
 try:
  torch.cuda.set_per_process_memory_fraction(policy['worker_allocator_limit_mib']*2**20/torch.cuda.get_device_properties(device).total_memory,device)
  w.w.gpu_identity(d);identity=json.loads((d/'gpu_identity.json').read_text());assert all(r[0].strip()==uuid for r in identity['nvml'])
  if a.phase=='train':
   assert json.loads((STUDY/'smoke_test_report.json').read_text())['status']=='PASS'
   if a.job>=4:assert json.loads((STUDY/'representative_report.json').read_text())['status']=='PASS'
   if a.job>=12:assert json.loads((STUDY/'seed_gate_report.json').read_text())['status']=='PASS'
  else:assert all(json.loads((Path(x['run_dir'])/'status.json').read_text())['status']=='trained' for x in p['jobs'])
  args,b,model,pa,cfg=w.setup(j,p,device)
  if (d/'config.json').exists():assert json.loads((d/'config.json').read_text())==cfg
  else:save(d/'config.json',cfg)
  log(f'shared GPU adapter {a.phase} 시작 {j["name"]} GPU{gpu} PID{os.getpid()}; 과학 설정·seed·optimizer·학습 예산 불변')
  {'train':w.train,'evaluate':w.evaluate}[a.phase](args,b,model,pa,cfg,d,device);b.ds.close();log(f'shared GPU adapter {a.phase} 완료 {j["name"]}')
 except BaseException:
  save(d/f'failure_{a.phase}_{time.time_ns()}.json',dict(error=traceback.format_exc(),timestamp=stamp()));save(d/'status.json',dict(status='failed',phase=a.phase,timestamp=stamp()));raise
if __name__=='__main__':main()
