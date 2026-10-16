"""GPU0/1/2, maximum three processes per GPU; preserve original protocol. Freeze all48 Validation-best states before Test."""
import os,sys,json,csv,subprocess,time,fcntl,math
from pathlib import Path
import train_v3_study as w
ROOT=w.ROOT;OUT=ROOT/'outputs/v3_physics_ratio_lead_lambda001_20261009'
def query(kind):
 fields='index,uuid,memory.total,memory.free' if kind=='gpu' else 'gpu_uuid,pid,process_name,used_memory'
 return [[v.strip() for v in row] for row in csv.reader(subprocess.check_output(['nvidia-smi','--query-'+('gpu' if kind=='gpu' else 'compute-apps')+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()) if row]
def alive(row):
 try:
  q=Path('/proc')/str(row['pid']);f=(q/'stat').read_text().rsplit(')',1)[1].split()
  return q.stat().st_uid==os.getuid() and f[0]!='Z' and f[19]==row['start_ticks']
 except FileNotFoundError:return False

def run_manifest(jobs,phase,done,active,failed):
 rows=[]
 for j in jobs:
  sf=Path(j['run_dir'])/'status.json';s=json.loads(sf.read_text()) if sf.exists() and sf.stat().st_size else {}
  state=('reused_verified' if j['reuse'] else 'waiting')
  if s.get('status')=='trained':state='trained'
  if j['id'] in active:state=phase+'_running'
  if j['id'] in json.loads((OUT/'resource_policy_gpu012.json').read_text())['blocked_job_ids']:state='blocked_corrupt_resume_checkpoint'
  if any(x['job']==j['id'] for x in failed):state='failed'
  if (Path(j['evaluation_dir'])/'metrics.json').exists():state='evaluated'
  rows.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['arm'],reuse_new='Reuse' if j['reuse'] else 'New',checkpoint=str(Path(j['run_dir'])/'best.pt'),config=str(Path(j['run_dir'])/'config.json'),evidence=j['reuse_evidence']['reason'] if j['reuse'] else 'new native matched-all-head training',status=state,best_update=s.get('best_epoch',0)*478,end_update=s.get('steps'),validation_rmse=s.get('best_val_rmse')))
 tmp=OUT/'run_manifest.csv.tmp'
 with tmp.open('w',newline='') as f:
  c=csv.DictWriter(f,list(rows[0]));c.writeheader();c.writerows(rows)
 tmp.replace(OUT/'run_manifest.csv')

def freeze(jobs):
 rows=[];valrows=[];torch=w.torch;torch.set_num_threads(2)
 for jb,jp in zip(jobs[::2],jobs[1::2]):
  configs=[json.loads((Path(j['run_dir'])/'config.json').read_text()) for j in [jb,jp]]
  for k in ['parameters','trainable_parameters','shared_initial_sha256','stats','manifest_sha256','rhs_scale','applied_rhs_scale','split_sha256','tensor_fingerprints']:assert configs[0][k]==configs[1][k],(jb['name'],k)
  histories=[json.loads((Path(j['run_dir'])/'history.json').read_text()) for j in [jb,jp]]
  assert all(a['batch_order_sha256']==b['batch_order_sha256'] for a,b in zip(*histories))
  for j,c,h in zip([jb,jp],configs,histories):
   d=Path(j['run_dir']);s=json.loads((d/'status.json').read_text());assert s['status']=='trained'
   assert c['settings']['lambda_physics']==j['lambda_physics'] and c['settings']['lambda_rres']==0
   assert c['config_sha256']==w.digest({k:v for k,v in c.items() if k!='config_sha256'})
   assert c['manifest_sha256']==j['manifest_sha256']==w.sha(j['manifest'])
   best=min(h,key=lambda r:r['val_rmse']);assert s['best_epoch']==best['epoch'] and s['steps']==len(h)*478
   assert len(h)==80 or len(h)-best['epoch']==12
   assert all(r['rres_calls']==0 and math.isfinite(r['gradient_norm_max']) for r in h)
   assert all((r['physics_calls']==0) if j['arm']=='baseline' else r['physics_calls']>0 for r in h)
   cp=torch.load(d/'best.pt',map_location='cpu',weights_only=False)
   assert cp['config_sha256']==c['config_sha256'] and cp['best_epoch']==best['epoch'] and cp['best_value']==best['val_rmse']
   assert all(torch.isfinite(v).all() for v in cp['model_state'].values())
   ch=w.sha(d/'best.pt')
   if j['reuse']:assert ch==j['reuse_evidence']['checkpoint_sha256']
   rows.append(dict(id=j['id'],checkpoint=str(d/'best.pt'),checkpoint_sha256=ch,config_sha256=c['config_sha256'],best_epoch=best['epoch'],best_val_rmse=best['val_rmse']))
   valrows.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['arm'],validation_rmse=best['val_rmse'],best_update=best['steps'],end_update=s['steps'],early_stopping=s['early_stopping']))
   del cp
 for b,p in zip(valrows[::2],valrows[1::2]):
  for r in [b,p]:r['paired_improvement_percent']=100*(b['validation_rmse']-r['validation_rmse'])/b['validation_rmse']
 with (OUT/'validation_results.csv').open('w',newline='') as f:
  c=csv.DictWriter(f,list(valrows[0]));c.writeheader();c.writerows(valrows)
 w.save(OUT/'checkpoint_lock.json',dict(status='PASS',rows=rows,timestamp=w.stamp(),selection='Validation only; immutable48 checkpoint hashes before any new Test'))

def main():
 lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prot=json.loads((OUT/'protocol.json').read_text());pre=json.loads((OUT/'preflight.json').read_text());assert pre['status']=='PASS';jobs=prot['jobs'];peak=pre['peak_reserved_mib']
 for n,h in prot['code_sha256'].items():assert w.sha(ROOT/n)==h
 for n,h in json.loads((OUT/'operator_sources.json').read_text()).items():assert w.sha(ROOT/n)==h
 for n,h in json.loads((OUT/'operator_sources_gpu012.json').read_text()).items():assert w.sha(ROOT/n)==h
 policy=json.loads((OUT/'resource_policy_gpu012.json').read_text());assert policy['gpu_ids']==[0,1,2] and policy['max_processes_per_gpu']==3
 assert w.sha(OUT/'protocol.json')==policy['original_protocol_sha256']
 blocked=set(policy['blocked_job_ids'])
 statefile=OUT/'queue_state.json';previous=json.loads(statefile.read_text()) if statefile.exists() else {};phase=previous.get('phase','train');attempts=previous.get('attempts',[]);failed=previous.get('failed',[]);assert not failed,'Diagnose failed jobs before retry'
 active={r['job']:dict(**r,process=None) for r in previous.get('active',[]) if alive(r)}
 while True:
  done=[]
  for j in jobs:
   if j['id'] in blocked:continue
   d=Path(j['run_dir']) if phase=='train' else Path(j['evaluation_dir']);sf=d/('status.json' if phase=='train' else 'metrics.json')
   if sf.exists():
    s=json.loads(sf.read_text())
    if s.get('status')==('trained' if phase=='train' else 'completed'):done.append(j['id'])
    peak=max(peak,s.get('peak_reserved_mib',0))
  for jid,row in list(active.items()):
   proc=row['process'];rc=proc.poll() if proc else (None if alive(row) else 0)
   if rc is not None:
    del active[jid]
    if rc!=0 or jid not in done:failed.append(dict(job=jid,phase=phase,returncode=rc,log=row['log']))
  state=dict(timestamp=w.stamp(),phase=phase,total=48,reused=prot['reuse_count'],new=prot['new_count'],completed=len(done),completed_ids=done,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],waiting=[j['id'] for j in jobs if j['id'] not in done and j['id'] not in active and j['id'] not in blocked],failed=failed,attempts=attempts,peak_reserved_mib=peak,gpu_ids=[0,1,2],max_processes_per_gpu=3,blocked=sorted(blocked))
  w.save(statefile,state);run_manifest(jobs,phase,done,active,failed)
  if failed:raise RuntimeError(f'No further launches; unchanged failed runs require diagnosis: {failed}')
  if len(done)==48 and not active:
   if phase=='train':freeze(jobs);phase='evaluate';w.log('Fixedλ=.01 all48 Validation-best checkpoints frozen; Test inference gate PASS.');continue
   state['phase']='analysis';w.save(statefile,state)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
   with (OUT/'analysis.log').open('a') as log:subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts/analyze_ratio001.py')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
   state['phase']='completed';w.save(statefile,state);w.log('Fixedλ=.01 ratio/lead all48training/evaluation/report complete.');return
  # One admission per20s poll; inventory refreshed before each placement.
  for j in jobs:
   if j['id'] in done or j['id'] in active or j['id'] in blocked:continue
   if phase=='train':assert not j['reuse']
   inventory=[r for r in query('gpu') if r[0] in ['0','1','2']];apps=query('apps');own={str(r['pid']) for r in active.values()};choices=[]
   for g in inventory:
    foreign=[r for r in apps if r[0]==g[1] and r[1] not in own]
    try:safe=all(r[2]=='yeonwoo' and Path('/proc/'+r[1]).stat().st_uid==os.getuid() for r in foreign)
    except FileNotFoundError:continue
    if not safe:continue
    count=sum(r['gpu']==g[0] for r in active.values());pending=sum(r['gpu']==g[0] and not any(a[1]==str(r['pid']) for a in apps) for r in active.values())
    if count+len(foreign)>=3:continue
    if int(g[3])-pending*peak*1.2<4096+peak*1.2:continue
    choices.append((count+len(foreign),-int(g[3]),g[0]))
   if not choices:break
   gpu=min(choices)[2];env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
    d=OUT/'runtime'/str(j['id'])/key;d.mkdir(parents=True,exist_ok=True);env[key]=str(d)
   logfile=OUT/'logs'/f'{phase}_{j["id"]}_{time.time_ns()}.log';script='ratio001_worker_gpu012.py' if phase=='train' else 'evaluate_ratio001_gpu012.py'
   with logfile.open('w') as f:proc=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts'/script),'--study',str(OUT),'--job',str(j['id'])],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   row=dict(job=j['id'],phase=phase,pid=proc.pid,gpu=gpu,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19],log=str(logfile));attempts.append(row);active[j['id']]=dict(**row,process=proc);print(json.dumps(row),flush=True);break
  state.update(attempts=attempts,active=[{k:v for k,v in r.items() if k!='process'} for r in active.values()],waiting=[j['id'] for j in jobs if j['id'] not in done and j['id'] not in active and j['id'] not in blocked]);w.save(statefile,state);time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  w.save(OUT/'supervisor_failure.json',dict(error=traceback.format_exc(),timestamp=w.stamp()));raise
