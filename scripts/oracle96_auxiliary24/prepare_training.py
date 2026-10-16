"""CPU protocol/sampling audit; preserve original audit, apply user's reclassification."""
from aux_common import *
import worker as W
import torch,numpy as np,math,shutil,ast
from torch.utils.data import Dataset

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 assert not (AUX/'training_protocol.json').exists(),'Frozen training plan exists'
 torch.set_num_threads(1);p=protocol();verify_code(p)
 old=json.loads((AUX/'gradient_audit.json').read_text());assert old['initial_nonzero_shared_gradient'] and old['initial_predictor_tensor_equality'] and old['checkpoint_roundtrip']=='PASS'
 assert all(sha(path)==h for path,h in json.loads((AUX/'protected_audit_inputs.json').read_text()).items())
 archive=AUX/'original_preflight';archive.mkdir(exist_ok=True)
 for f in AUX.iterdir():
  if f.is_file() and f.suffix in ['.md','.json','.csv']:
   dest=archive/f.name
   if not dest.exists():shutil.copy2(f,dest)
 for name in ['logs','launches','smoke','runs','figures','checkpoints']:(AUX/name).mkdir(exist_ok=True)
 class Indices(Dataset):
  def __init__(self,ids):self.ids=ids
  def __len__(self):return len(self.ids)
  def __getitem__(self,i):return dict(start_index=torch.tensor(self.ids[i]))
 jobs=[];tests=[]
 for base in p['jobs']:
  if base['label']!='Baseline':continue
  c=json.loads((Path(base['run_dir'])/'config.json').read_text());h=json.loads((Path(base['run_dir'])/'history.json').read_text());m=json.loads(Path(base['manifest']).read_text())
  assert len(h)==80 and sha(base['manifest'])==base['manifest_sha256']
  a,_=W.arguments(base);a.physics_warmup_epochs=0;a.lambda_rres=.0001
  loader=W.t.make_loader(Indices(m['train_start_indices']),a,True);n=len(loader.dataset);assert n==base['n_train']
  dummy=torch.nn.Linear(1,1);opt=torch.optim.AdamW(dummy.parameters(),lr=a.initial_lr)
  for ep in range(1,81):
   checked=W.CheckedEpoch(loader,m['train_start_indices'])
   for batch in checked:pass
   ev=checked.evidence();assert ev['sample_order_sha256']==h[ep-1]['sample_order_sha256'] and ev['epoch_updates']==base['updates_per_epoch']
   assert W.t.set_epoch_lr(opt,a,ep)==h[ep-1]['lr']
  name=f'Auxiliary_r{base["ratio"]}_lead{base["lead"]}_seed{base["seed"]}'
  jobs.append(dict(id=len(jobs),name=name,model='Auxiliary-only',ratio=base['ratio'],lead=base['lead'],seed=base['seed'],baseline_id=base['id'],baseline_run=base['run_dir'],manifest=base['manifest'],manifest_sha256=base['manifest_sha256'],n_train=n,updates_per_epoch=base['updates_per_epoch'],total_updates=80*base['updates_per_epoch'],lambda_aux=.0001,lambda_physics=0.,run_dir=str(AUX/'runs'/name)))
  tests.append(dict(name=name,status='PASS',epochs=80,train_samples=n,sampling_and_lr_identical=True,manifest_sha256=base['manifest_sha256'],baseline_initial_sha256=c['shared_initial_sha256']))
 assert len(jobs)==24
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 cfg=dict(created_at=stamp(),gpus=GPUS,max_processes_per_gpu=2,epochs=80,early_stopping=False,lambda_aux=.0001,lambda_physics=0.,aux_warmup=False,physics_warmup=False,test_evaluation='FORBIDDEN',oracle_protocol_sha256=sha(ORACLE/'experiment_protocol.json'),jobs=jobs,code_sha256={str(f.relative_to(ROOT)):sha(f) for f in HERE.glob('*.py')})
 save(AUX/'training_protocol.json',cfg);save(AUX/'cpu_preflight.json',dict(status='PASS',sampling_lr_all24=tests,original_gradient_checks_preserved=True,zero_head_reclassified='expected regularization minimum; nonblocking limitation',GPU_smoke='PENDING',timestamp=stamp()))
 save(AUX/'preflight_reclassification.json',dict(timestamp=stamp(),previous_status='BLOCKED_PREFLIGHT',current_decision='zero-head minimum is expected, not an implementation error or blocking criterion',authority='explicit user follow-up',unchanged_loss='data + 0.0001 Rres penalty',retained_checks='original gradient/initialization/sampling/scheduler PASS',next_gate='diagnostic noninterference GPU smoke',gpus=GPUS))
 for name in ['auxiliary_definition_audit.md','gradient_audit.md']:
  text=(archive/name).read_text();(AUX/name).write_text(f'> {stamp()} 판정 변경: zero-head 최소해는 의도된 정규화의 예상 성질로 비차단 한계로 기록한다. 아래 과거 FAIL/중단 문구는 당시 감사 이력이며 현재 실행 조건이 아니다. 원본은 original_preflight/에 보존.\n\n'+text)
 (AUX/'protocol.md').write_text(f'''# Auxiliary24 Actual Epoch protocol
작성 {stamp()}

Ratios10/25/50/100 × lead7/14 × seed42/7/123. 24개 신규 학습. 기존96개 read-only.
Loss = Data MSE + 0.0001 Rres penalty, fixed v3 RHS scale. Physics OFF, entrainment tie OFF, auxiliary warmup 없음, heads/초기화 unchanged.
80actual epochs, early stopping OFF, batch16, AdamW(.9,.95)/weight decay.1, clip norm1, peakLR.0002, initialLR.00002, warmup5actual epochs + native cosine. 전체 Validation SST RMSE로 best 선택. Test 금지.
매 epoch train index 정확히1회, 대응 Oracle96 Baseline의 shuffle order/Scheduler와 런타임 대조. 각 run 시작 시 독립 재구성 모델의 tensor 및 원 Baseline initial SHA, 통계/split/input fingerprint 일치 검사.
GPU {GPUS}, GPU당 우리 작업 최대2, 14GiB/worker 예약+4GiB headroom, 타 사용자 메모리 포함. tmux 유지 및 PID/GPU/log 기록. 동일 config epoch-boundary resume 지원.
매 epoch Rres RMS·loss·가중비·마지막층norm 기록. epoch1/5/10/20/80 및 모든 best 후보에 고정 Train16 batch 별도 autograd.grad 진단; model state/RNG/.grad/mode 보존 검사. 실제 optimizer 1step에 대해 진단ON/OFF bitwise 일치 smoke 후 확장.
zero-head 해는 예상되는 정규화 최소해이며 학습 장애가 아니다. shared regularization이 약하거나 초기에만 작용할 가능성을 진단한다. 성능에 따라 설정을 바꾸지 않는다.
''')
 (AUX/'preflight.md').write_text('# Preflight 진행\n\n기존 zero-head FAIL은 사용자 지시로 비차단 한계로 재분류. 기존 PASS 수치와 원본 보존. CPU: 24개×80epoch sampling 및 LR가 원 Baseline과 일치 PASS. GPU 진단 비간섭 smoke PENDING; PASS 후 자동24개 실행.\n')
 with (AUX/'run_manifest.csv').open('w') as f:
  keys=list(jobs[0]);w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(jobs)
 save(AUX/'run_state.json',dict(status='READY_FOR_SMOKE',completed=0,total=24,running=[],gpus=GPUS,test_evaluation='NOT_RUN',timestamp=stamp()))
 journal('zero-head 최소해를 사용자 지시로 비차단 정규화 한계로 재분류; 원 감사 original_preflight/ 복사 보존. aux_common/aux_worker/prepare_training/queue/summarize 신규 추가. 원 모델·손실 그대로 λaux=.0001, GPU5/6/7. 24×80epoch CPU sampling/LR PASS; GPU 비간섭 smoke 대기. Test 금지.')
 print('CPU all24 sampling/LR PASS; frozen plan saved',flush=True)
if __name__=='__main__':main()
