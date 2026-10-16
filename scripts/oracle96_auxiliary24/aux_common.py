"""Isolated Auxiliary24 runner; original Oracle96 artifacts are read-only."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/oracle96_actual'))
from mixed_budget_common import *
ORACLE=STUDY
AUX=ROOT/'outputs/v8_actual_epoch_auxiliary24_20261015'
HERE=ROOT/'scripts/oracle96_auxiliary24'
GPUS=[5,6,7]
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 d=AUX/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
def plan():return json.loads((AUX/'training_protocol.json').read_text())
def verify(p):
 verify_code(protocol())
 assert sha(ORACLE/'experiment_protocol.json')==p['oracle_protocol_sha256']
 for f,h in p['code_sha256'].items():assert sha(ROOT/f)==h,('Auxiliary source changed',f)
 assert p['gpus']==GPUS and p['lambda_aux']==.0001

def journal(s):
 with (ROOT/'docs/research_log.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write('\n- '+stamp()+' — Auxiliary24: '+s+'\n');f.flush()
