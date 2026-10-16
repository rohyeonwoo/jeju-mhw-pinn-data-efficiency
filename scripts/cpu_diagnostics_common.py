"""Read-only inputs, bounded CPU resources, independent analysis outputs."""
import os,sys,resource,time,json,csv,hashlib,math,fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v3_cpu_diagnostics_20261011'
OLD=ROOT/'outputs/v3_physics_ratio_lead_lambda001_20261009'
ADAPT=ROOT/'outputs/v3_adaptive_physics_pilot_20261011'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only execution requires explicit empty CUDA_VISIBLE_DEVICES'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 path=OUT/'runtime'/key;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
sys.dont_write_bytecode=True
resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
try:os.nice(10)
except PermissionError:pass
import setproctitle
setproctitle.setproctitle('yeonwoo')
import numpy as np
sys.path.insert(0,str(ROOT/'src'))
SEEDS=[42,7,123];RATIOS=[10,25,50,100]
def stamp():return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def save(path,x):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def csvsave(name,rows):
 path=OUT/name;rows=[clean(r) for r in rows];keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def readcsv(path):return list(csv.DictReader(Path(path).open()))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def log(msg):
 with (ROOT/'docs/research_log.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(f'\n- {stamp()} — CPU-only analysis: {msg}\n')
def jobs():return json.loads((OLD/'protocol.json').read_text())['jobs']
def meta(j):return dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'])
def pair_jobs():
 js=jobs()
 return [(next(b for b in js if meta(b)==meta(j) and b['arm']=='baseline'),j) for j in js if j['arm']=='physics_only']
def fields(j,split,keys):
 met=json.loads((Path(j['evaluation_dir'])/'metrics.json').read_text());p=met['fields'][split]['path']
 with np.load(p) as z:return {k:z[k].copy() for k in keys}
def corr(x,y):
 x=np.asarray(x,dtype=float).ravel();y=np.asarray(y,dtype=float).ravel();return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else None
def seasons(dates):return np.array(['DJF','MAM','JJA','SON'])[(dates.astype('datetime64[M]').astype(int)%12+1)%12//3]
def table(rows,keys):return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(f'{r[k]:.6g}' if isinstance(r.get(k),float) else str(r.get(k,'N/A')) for k in keys)+' |' for r in rows])
def finish(phase,**extra):
 import torch
 assert not torch.cuda.is_initialized()
 save(OUT/(phase+'_status.json'),dict(status='PASS',phase=phase,timestamp=stamp(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=False,threads=2,address_space_limit_gib=16,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**extra));log(f'{phase} PASS; outputs={OUT}; CUDA disabled; no model inference/backward/training.');print(phase,'PASS',flush=True)
