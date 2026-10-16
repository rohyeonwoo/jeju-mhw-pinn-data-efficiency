"""Shared, Ceph-only v7 experiment configuration. Import never starts CUDA."""
import os,sys,json,hashlib,time,subprocess,csv,fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import setproctitle
setproctitle.setproctitle('yeonwoo');assert setproctitle.getproctitle()=='yeonwoo'
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'outputs/v8_actual_epoch_oracle96_20261011'
OLD_STUDY=ROOT/'outputs/v8_training_only_oracle96_20261011'
PYTHON=Path('/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python')
BUDGET=ROOT/'data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc'
GLORYS=Path('/ceph_data/yeonwoo/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d.nc')
FEATURES=ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011/features'
sys.path.insert(1,str(ROOT/'scripts'))
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 p=STUDY/'runtime'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,default=str).encode()).hexdigest()
def save(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(p)
def log(s):
 with (ROOT/'docs/research_log.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n- '+stamp()+' — v8 actual-epoch oracle96: '+s+'\n');f.flush()
def protocol():return json.loads((STUDY/'experiment_protocol.json').read_text())
def verify_code(p):
 for name,h in p['code_sha256'].items():assert sha(ROOT/name)==h,('code changed',name)
def snapshot():
 def run(cmd):
  r=subprocess.run(cmd,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 return dict(timestamp=stamp(),gpu=run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv']),processes=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),tmux=run(['tmux','list-sessions']),disk=run(['df','-h',str(ROOT)]))
def gpu_idle(gpu):
 assert gpu in [7,8,9]
 rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()));row=next(r for r in rows if int(r[0])==gpu);uuid=row[1].strip()
 apps=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()))
 if any(r[0].strip()==uuid for r in apps) or int(row[2])>256:raise RuntimeError(f'GPU {gpu} is occupied; no process was stopped')
 return uuid
def guard_hashes(phase):
 expected=json.loads((STUDY/'protected_input_hashes.json').read_text());actual={p:sha(p) for p in expected};bad=[p for p in expected if expected[p]!=actual[p]]
 save(STUDY/f'protected_hash_{phase}.json',dict(status='FAIL' if bad else 'PASS',count=len(actual),mismatches=bad,actual=actual,timestamp=stamp()));assert not bad,bad
 return dict(status='PASS',count=len(actual),timestamp=stamp())
