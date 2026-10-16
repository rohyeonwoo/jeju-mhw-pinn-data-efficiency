import os,sys,json,hashlib,subprocess,time,csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';A=R/'outputs/v8_actual_epoch_auxiliary24_20261015';T=R/'outputs/v8_actual_epoch_oracle96_20261011'
sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
p=json.loads((A/'training_protocol.json').read_text());jobs=[]
for j in p['jobs']:
 cp=A/'checkpoints'/j['name']/'best.pt';audit=json.loads((Path(j['run_dir'])/'checkpoint_audit.json').read_text());assert sha(cp)==audit['sha256']
 jobs.append(dict(name=j['name'],checkpoint=str(cp),sha256=audit['sha256'],epoch=audit['epoch'],val_rmse=audit['expected_val_rmse']))
(O/'auxiliary_test').mkdir(exist_ok=True)
(O/'auxiliary_test/protocol.json').write_text(json.dumps(dict(jobs=jobs,reference_sha256=sha(T/'evaluation_reference.npz'),training_protocol_sha256=sha(A/'training_protocol.json'),inference_source_sha256=sha(O/'aux_test.py'),selection='fixed pre-existing Validation-best; no re-selection',gpus=[5,6,7],max_processes_per_gpu=1,optimizer_steps=0),indent=2))
for gpu in [5,6,7]:
 ids=list(range(gpu-5,24,3));script=O/f'aux_gpu{gpu}.sh'
 text='#!/bin/bash\nset -eu\ncd '+str(R)+'\nexport CUDA_VISIBLE_DEVICES='+str(gpu)+'\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
 text+='export TMPDIR='+str(O/'runtime')+'\n'
 for i in ids:text+=f'/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B {O}/aux_test.py {i} > {O}/auxiliary_test/job{i}.log 2>&1\n'
 script.write_text(text)
 subprocess.run(['tmux','-S','/ceph_data/yeonwoo/paper_audit.sock','new-session','-d','-s',f'aux_test_{gpu}','bash '+str(script)],check=True)
with (R/'docs/research_log.md').open('a') as f:f.write('\n- '+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')+' — 최종 논문 감사: outputs/paper_final_verification_20261015에 격리된 감사/추론 코드 생성. 기존 Auxiliary24 Validation-best SHA를 고정하고 GPU 5/6/7 각각 1개씩 Test 추론(미래 forcing 제거/원본/NaN 불변성 포함) 시작. 학습·모델·손실 변경 없음.\n')
print('Launched inference-only GPU 5,6,7, one per GPU')
