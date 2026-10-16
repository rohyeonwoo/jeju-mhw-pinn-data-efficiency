"""Read-only training artifacts; separate authorized Test evaluation outputs."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/oracle96_actual'))
from mixed_budget_common import *
TRAIN=STUDY
EVAL=TRAIN/'test_evaluation_20261012'
for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 q=EVAL/'runtime'/key;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
def evaluation_protocol():return json.loads((EVAL/'evaluation_protocol.json').read_text())
def verify_evaluation(p):
 verify_code(protocol())
 assert sha(TRAIN/'experiment_protocol.json')==p['training_protocol_sha256']
 assert sha(TRAIN/'evaluation_reference.npz')==p['reference_sha256']
 for path,h in p['evaluation_code_sha256'].items():assert sha(ROOT/path)==h,path
