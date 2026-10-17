"""Freeze launch metadata after CPU checks; provision unchanged Validation reference."""
from mixed_budget_common import *
import ast,shutil
import controller as C

def main():
 p=protocol();pre=json.loads((STUDY/'preflight.json').read_text());assert pre['status']=='PASS'
 assert not (STUDY/'controller_pid.json').exists(),'Never alter a running/frozen experiment'
 ref=OLD_STUDY/'evaluation_reference.npz';dest=STUDY/ref.name
 if dest.exists():assert sha(dest)==sha(ref)
 else:shutil.copy2(ref,dest)
 p['evaluation_reference_sha256']=sha(dest)
 p['lr']='native epoch-based formula: 5 actual-epoch linear LR warmup then cosine to epoch80; physics warmup independently 5 actual epochs'
 p['protocol_revision']='actual epoch initial launch; legacy virtual fields superseded explicitly'
 p['gpu_allocator_limit_mib']=14336;p['gpu_free_headroom_mib']=4096
 p['code_sha256'].update({str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')})
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
 # Capacity includes other users plus unused reservations for pending CUDA workers.
 rows={7:dict(uuid='g7',used=0,total=49140),8:dict(uuid='g8',used=0,total=49140),9:dict(uuid='g9',used=0,total=49140)}
 assert C.eligible(7,[],rows,[])
 one=[dict(gpu=7,pid=10)];two=one+[dict(gpu=7,pid=11)]
 assert C.eligible(7,one,rows,[]) and not C.eligible(7,two,rows,[])
 occupied={7:dict(uuid='g7',used=33000,total=49140)};assert not C.eligible(7,[],occupied,[])
 busy={7:dict(uuid='g7',used=28000,total=49140)};assert not C.eligible(7,one,busy,[['g7','10','1000']])
 save(STUDY/'resource_policy_tests.json',dict(status='PASS',two_per_gpu=True,others_memory_counted=True,pending_worker_reservation=True,allowed_gpus=[7,8,9],timestamp=stamp()))
 save(STUDY/'experiment_protocol.json',p);verify_code(p)
 pre.update(protocol_sha256=digest(p),launch_metadata_finalized=True,resource_policy='PASS');save(STUDY/'preflight.json',pre)
 # Exact function line references for the new path, plus preserved legacy reference sites.
 refs=[]
 for name in ['worker.py','controller.py','preflight.py','summarize.py']:
  path=Path(__file__).parent/name;tree=ast.parse(path.read_text())
  for node in tree.body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):refs.append(f'- `scripts/oracle96_actual/{name}:{node.lineno}–{node.end_lineno}` — `{node.name}`')
 with (STUDY/'preflight.md').open('a') as f:f.write('\n## 실제 코드 위치\n\n'+'\n'.join(refs)+'\n\nGPU resource policy synthetic checks PASS. Existing Validation/MHW reference copied byte-for-byte (SHA checked), without recomputation. No Test evaluation command exists in the worker CLI.\n')
 log('Actual Epoch CPU 최종 준비: 기존 Validation/MHW reference 바이트 동일 사본, 신규 코드 SHA 동결, GPU7/8/9당2개 및 대기 worker 메모리 예약 검사 PASS. 기존 Oracle96 LR 서술은 실제 epoch 수식에 맞게 수정하고 physics warmup0→5 변경을 분리 명시. Test 자동 실행 없음.')
 print('FINALIZED',digest(p),flush=True)
if __name__=='__main__':main()
