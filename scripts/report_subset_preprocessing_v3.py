"""Finalize v3 validation state only after all CPU/GPU evidence passes."""
from pathlib import Path
import json,hashlib,ast,csv
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parents[1];O=R/'outputs/v3_preflight_20261008';V=R/'data/subsets/preprocessing_subset_v3'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 index=read(V/'index.json');validation=read(V/'validation_report.json');gpu=read(O/'gpu_smoke.json');arms=read(O/'loss_arms.json');preserve=read(O/'preservation_before.json')
 assert gpu['status']==arms['status']=='PASS' and all(v=='PASS' for v in validation['gates'].values())
 assert '\nOK\n' in (O/'regression_tests.log').read_text()
 for row in index['datasets']:
  path=V/row['manifest'];assert sha(path)==row['sha256'];m=read(path);old=read(Path(m['parent_manifest']))
  assert sha(m['parent_manifest'])==m['parent_manifest_sha256']
  for key in ('statistics','train_start_indices','val_start_indices','test_start_indices','input_time_indices','target_time_indices','climatology_sha256','climatology_rule'):assert m[key]==old[key]
  assert sha(V/m['climatology_file'])==m['climatology_sha256']
  for source,digest in m['source_sha256'].items():assert sha(R/source)==digest,source
 for name,digest in preserve['hashes'].items():assert sha(name)==digest,name
 for name,metadata in preserve['existing_results'].items():f=Path(name);assert [f.stat().st_size,f.stat().st_mtime_ns]==metadata,name
 old=ast.parse((O/'originals/src/train_jeju_hilam.py').read_text());new=ast.parse((R/'src/train_jeju_hilam.py').read_text())
 of={n.name:ast.dump(n) for n in old.body if isinstance(n,ast.FunctionDef)};nf={n.name:ast.dump(n) for n in new.body if isinstance(n,ast.FunctionDef)}
 for name in ('anomaly_acc','add_acc_and_sw_metrics','collect_predictions'):assert of[name]==nf[name],name
 assert sha(R/'src/mhw/models/jeju_hilam.py')==read(R/'outputs/v2_final_preflight_20261008/verdict.json')['source_sha256']['src/mhw/models/jeju_hilam.py']
 assert len(index['datasets'])==24 and all(x['physics_rhs_scale']>0 for x in index['datasets'])
 for model in gpu['models']:
  assert model['status']=='PASS' and model['forward_batches']==model['backward_batches']==2 and model['optimizer_steps']==1
 identity=read(O/'gpu_smoke_attempt_001/gpu_process_identity.json');assert identity['title']==identity['proc_comm']=='yeonwoo' and all(x[2].strip()=='yeonwoo' for x in identity['nvml'])
 validation.update(gpu_smoke='PASS',gpu_smoke_report=str(O/'gpu_smoke.json'),gpu_smoke_optimizer_steps_total=2,formal_training_executed=False,loss_arms_actual_model_cpu='PASS',final_verdict='PASS — 정식 학습 가능')
 # Clarify original zero-step fields describe CPU construction only.
 validation['optimizer_steps']=2;validation['training_executed']=False;validation['training_executed_definition']='formal training; limited GPU smoke separately recorded'
 index['status']='validated';index['gpu_smoke']='PASS';save(V/'index.json',index);save(V/'validation_report.json',validation)
 with (O/'rhs_scales.csv').open('w',newline='') as f:
  fields=['ratio','seed','lead','old_rhs_scale','physics_rhs_scale','ratio_old_new','validation'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:row[k] for k in fields} for row in index['datasets'])
 bt=['| Ratio | Seed | Lead | 기존 RHS scale | 보정 RHS scale | 검증 |','|---|---|---|---:|---:|---|']
 for row in index['datasets']:bt.append(f"| {row['ratio']}% | {row['seed']} | {row['lead']} | {row['old_rhs_scale']:.9f} | {row['physics_rhs_scale']:.9f} | PASS |")
 ct=['| Model | Physics 계산 | Auxiliary 계산 | Gradient 검사 | 판정 |','|---|---|---|---|---|']
 for row in arms['arms']:
  ct.append(f"| {row['arm']} | {row['physics_calls']}회 | {row['rres_calls']}회 | 비활성 head grad=None, 활성 shared gradient 유한·비영 | PASS |")
 dt=['| Model | Forward | Backward | Loss finite | Gradient finite | 판정 |','|---|---|---|---|---|---|']
 for row in gpu['models']:dt.append(f"| {row['model']} | 2 batch PASS | 2 batch PASS | PASS | PASS | PASS |")
 nums=['| 모델 | Data loss (batch1/2) | Physics loss (batch1/2) | Total loss (batch1/2) | 총 gradient norm (clip 전) | peak allocated / reserved MiB |','|---|---|---|---|---|---|']
 for row in gpu['models']:
  vals=lambda key:' / '.join(f"{b[key]:.6f}" for b in row['measurements'])
  clip_text=' / '.join(format(b['before'],'.6f') for b in row['gradient_clipping'])
  nums.append(f"| {row['model']} | {vals('data_loss')} | {vals('physics_loss')} | {vals('total_loss')} | {clip_text} | {row['peak_allocated_mib']:.1f} / {row['peak_reserved_mib']:.1f} |")
 ts=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
 report=f'''# Subset-only v3 해안 차분 수정 및 최종 검증

{ts}. **PASS — 정식 학습 가능. 정식 자료량별 학습·Ablation은 시작하지 않았다.**

이번 PASS는 수정된 v3 경로의 전처리/연산 일관성과 지정된 제한 GPU smoke 통과를 뜻한다. 장기 수렴·예측 성능·미실행 GPU ablation의 결과를 뜻하지 않는다. 기존 raw/v1/v2/climatology/인덱스/checkpoint/실험 결과를 보존했다. 원본 모델 구조와 MHW 평가 기준은 변경하지 않았다.

## A. 공간차분 수정 결과

| 검사 | 수정 전 | 수정 후 | 판정 |
|---|---|---|---|
| 해수 전체20, 육지0 | 해양141개에서 비영 gradient | 661개 해양 gradient 모두0 | PASS |
| 육지값0→100 | 해양 gradient 변화 | 정확히 불변 | PASS |
| 육지 NaN/Inf | 마스크 없는 차분에서 유입 가능 | 연산 전에 육지를0으로 치환; gradient 및 backward 유한 | PASS |
| 기존 검증된 보정과 비교 | 별도 runner에만 적용 | 실제 격자 전체와 무작위 mask에서 정확히 일치 | PASS |
| 양쪽/한쪽/이웃 없음 | 육지값 포함 | 중앙/한쪽/0; 축별 해양 이웃 판정 | PASS |
| Physics support | 661개 | 양 축에 각각 이웃이 있는660개 | PASS |
| P0/P1 loss·RHS·잔차/예산 진단 | legacy 차분 | 같은 함수와 같은 support | PASS |

차분은 기존 서버 백업 `server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py:34-61`을 재사용했다. 새 원본은 `src/mhw/ocean_stencil.py:8-39`이다. 유한한 해양 값에서 기존 보정의 계산과 동일하다. 육지 NaN/Inf가 backward에 유입되지 않게 차분 전 육지를0으로 만드는 처리를 더했다. 해양 관측은 임의 clipping하거나 채우지 않았다.

- 양쪽 해양 이웃: `(T_right-T_left)/(2*dx)` 및 y 방향 대응식.
- 한쪽 해양 이웃: 해당 방향 한쪽 차분.
- 해당 축 해양 이웃 없음: 그 축의 gradient0. x 또는 y축 이웃이 없으면 physics loss/RHS 집계에서 제외한다.
- 온도 data loss·평가 마스크는 기존661개를 유지한다. 원래 ocean mask로 이웃을 판정하고, support mask는 집계에만 쓴다. support로 이웃 mask를 반복 축소하지 않는다.

실제 연결 위치(프로젝트 `/ceph_data/yeonwoo/MHW_subset` 기준):

| 파일:행 | 역할 |
|---|---|
| `src/mhw/ocean_stencil.py:15` | ocean_neighbor_support |
| `src/mhw/ocean_stencil.py:21` | ocean_neighbor_gradients |
| `src/train_hybrid_jeju_forecast.py:590` | spatial_gradients_torch 공통 위임 |
| `src/train_hybrid_jeju_forecast.py:659-735` | P0/P1 daily loss, 동일 support |
| `src/train_hybrid_jeju_forecast.py:738-792` | 선형 경로 resolved RHS |
| `src/train_hybrid_jeju_forecast.py:812-839` | 선택 sample RHS mean-square 집계 |
| `src/train_hybrid_jeju_forecast.py:891` | 별도 budget integration의 공통 1차 차분 호출 |
| `src/train_hybrid_jeju_forecast.py:913-1009` | budget diagnostic 차분·support |
| `src/train_jeju_hilam.py:180-213` | raw physics residual diagnostic |

P1의 수평 advection도 같은 차분으로 검증했다. 이번 v3 manifest는 P0 scale용이므로 P1 학습에 P0 manifest를 그대로 쓰는 것은 허용하지 않는다. 선택하지 않은 별도 horizontal-diffusion Laplacian은 새 물리식을 설계하지 않기 위해 변경하지 않았다.

회귀검사24개(기존12+새12)를 통과했다. 첫 검사에서 Rres 함수 분리 시 device 변수 누락을 발견해 수정했고, 기준 완화 없이 재실행해 모두 통과했다. 로그는 `regression_tests.log`, 실제 격자 결과는 `stencil_real_grid.json`이다. 새 검사는 `tests/test_v3_physics.py`에 있다.

수정 전후 잔차는 `rhs_residual_comparison.json`에24조건 모두 저장했다. 비교 입력은 **각 subset의 첫2개 선택 sample, 관측 SST 선형 경로, we=0**이며 학습 모델 결과가 아니다. 10%/seed42/lead7에서 RHS RMS는3.716934→0.333899 K/day, residual RMS는3.715992→0.294819 K/day였다. 아래 scale 표는 이2개 sample이 아니라 전체 선택 training sample의 mean-square다.

## B. RHS scale

단위는 `(K/day)^2`, 정의는 선택 training sample의 관측 선형 경로·we=0·resolved RHS의 해양 support pooled mean-square다. Loss residual은 기존대로 K/day에서 clip한 뒤 동일 fixed RHS scale로 정규화한다. 단위 변환·clip 한계·손실 가중치는 바꾸지 않았다.

{chr(10).join(bt)}

24조건에서 RHS>0 및 finite. 서로 다른 계산은20개이며, seed에 무관하게 선택 인덱스가 같은100%의 두 lead 계산을 재사용했다. Input/Target mean/std와 subset-only climatology는 공간차분에 의존하지 않으므로 다시 계산하지 않았다. v3는 기존12개 v2 NPZ를 상대경로와 SHA-256으로 참조한다. v2 삭제 없이 함께 보관해야 한다.

실제로 바뀐 manifest 필드는 `schema`, `physics_rhs_scale`, `physics_stencil`, `source_sha256`, `climatology_file`(기존 파일 참조 경로), `physics_scale_definition`(stencil/support 명시), `parent_manifest`, `parent_manifest_sha256`뿐이다. 나머지는 v2와 대조했다. 원본24개 manifest와 index에 hash를 기록했다.

코드 근거: `scripts/build_subset_preprocessing_v3.py`의 main/verify_setup, `src/mhw/subset_preprocessing.py:94-124`의 선택 sample RHS loader, `src/mhw/subset_preprocessing.py:170-211`의 v3 dispatch/scale 적용, `src/mhw/subset_climatology.py:98-133`의 기존 anomaly 로딩. Legacy v1/v2 scale로 새 physics 경로를 실행하면 명시적으로 오류를 내어 잘못된 분모의 재사용을 막는다. v1/v2 파일 자체는 바꾸지 않았다.

## C. 손실 분기 검증

{chr(10).join(ct)}

`src/train_jeju_hilam.py:264-299`의 `compute_batch_losses`가 활성 항만 계산하며, 같은 함수를 실제 `run_epoch`가 사용한다. lambda_eff=0이면 daily physics 함수를 호출하지 않는다. lambda_rres=0이면 Rres penalty를 호출하지 않는다. lambda_we_tie=0이면 entrainment tie도 호출하지 않는다. 단순히0을 곱하는 구현을 제거했다.

`src/train_hybrid_jeju_forecast.py:1081`의 독립 Rres penalty와 `:1120`의 tie penalty는 기존 식을 그대로 분리했다. `:795-809`는 fixed RHS가 있으면 resolved physics 계산을 하지 않고 바로 scale을 반환한다. 따라서 Auxiliary only에 physics gradient 경로가 재유입되지 않는다. Rres의 zero target/동일 RHS 분모, zero_rres_physics=True 설정은 유지한다. 평가용 physics diagnostic은 기존처럼 no_grad 영역에 분리돼 있으며, objective loss 함수 호출과 구분한다.

검증 근거: 비활성 loss 함수를 NaN 반환으로 대체해도 호출 횟수0이며 total loss 유한, 비활성 head 출력에 NaN을 주어도 loss/활성 gradient 유한. 실제 hidden256 모델에서도 네 조건을 CPU1batch로 검사해 비활성 head의 gradient가0 tensor가 아니라 **None**임을 확인했다. `loss_arms.json` 및 `scripts/verify_v3_loss_arms.py` 참조. 따라서 해당 head는 AdamW에서도 gradient 없는 parameter로 처리된다.

| Native 조건 | Shared predictor parameter | 생성된 we head | 생성된 Rres head | requires_grad=True 합계 | Loss에서 사용하는 경로 |
|---|---:|---:|---:|---:|---|
| Baseline | 14,887,681 | 0 | 0 | 14,887,681 | data |
| Physics only | 14,887,681 | 66,049 | 66,049 | 15,019,779 | data + physics(shared/we), Rres grad=None |
| Auxiliary only | 14,887,681 | 66,049 | 66,049 | 15,019,779 | data + Rres(shared/Rres), we grad=None |
| Full PINN | 14,887,681 | 66,049 | 66,049 | 15,019,779 | data + physics + Rres |

보조 head는 원래 구조를 유지해 둘 다 생성한다. Physics only에서 필요한 we head는 실제 물리손실에 들어가며 Rres head는 계산된 출력이 있어도 objective에 연결되지 않는다. 구조 생성 자체를 네 조건마다 새로 설계하지 않았다. 조건별 requires_grad parameter 수와 현재 loss에 gradient가 연결된 수는 다르다. CPU 진단의 gradient 연결 수는 baseline676,353 / physics742,402 / auxiliary742,402 / full808,451로 기록됐다. 기존 모델 구조의 모든 parameter가 gradient를 받는다고 주장하지 않는다. 모델 정의 파일은 변경하지 않았다.

CPU1batch에서 Auxiliary only의 Rres penalty는171.274, lambda_rres10을 적용한 total loss는1714.308로 finite였다. RHS가 작아지면서 동일 lambda_rres의 유효 강도가 커진 실제 관측이다. 이번에는 lambda를 임의로 재조정하지 않았고, GPU smoke는 요청된 Baseline/Physics only만 수행했다. 이 수치를 ablation 성능이나 장기 안정성 결과로 해석하지 않는다.

## D. GPU Smoke Test

{chr(10).join(dt)}

조건: 실제 v3 10%/seed42/lead7, batch16, 기존 hidden256/processor3/fp32 모델. 실제 main 초기화·native run_epoch를 사용했다. **모델별2batch forward/backward, optimizer update는1회만** 수행했다. 추가로 physics gradient를 분리 측정하기 위해 같은 forward graph에서 autograd.grad를 호출했으며 추가 input batch forward는 없었다. 두 번째 batch의 optimizer update는 smoke harness에서 생략했다.

{chr(10).join(nums)}

모든 batch에서 clip 후 norm은1.00001 이하였다. P0 physics의 shared predictor gradient norm은77.221207 / 94.884984로 비영이며 finite였다. 적용 lambda=0.1, lambda_rres=0, RHS scale=0.084930987. Baseline physics loss와 Rres penalty는0이며 해당 objective 함수는 호출되지 않는다.

P0의 두 batch pooled raw residual RMS는0.821498 K/day였다. 이는 모델 초기화와1회update 주변의 수치 진단이며 예측 성능 비교에 사용하지 않았다. 두 모델의 초기 shared predictor SHA와 실제 target-index batch 순서가 같았다.

GPU0 RTX A6000을 점유 검사 후 단독 사용했다. 실행 PID {identity['pid']}의 NVML process_name, process title, /proc/self/comm 모두 `yeonwoo`였다. 다른 사용자 프로세스를 종료하지 않았다. CUDA 오류는 없었다. 각 모델의 checkpoint를 Ceph 신규 smoke 디렉터리에 저장하고 새 모델에 strict load해 모든 state tensor가 정확히 동일함을 검증했다. 이는 checkpoint 상태 복원 검사이며 전체 학습 중간 resume 검사는 아니다.

근거: `gpu_smoke.json`, `gpu_smoke_attempt_001/gpu_process_identity.json`, `gpu_before.json`, `results.json`, `scripts/smoke_subset_v3_gpu.py`. GPU batch 제한을 위해 같은 조건으로 smoke를 반복하지 않았다.

## 최종10개 gate 및 보존

| 항목 | 판정 | 근거 |
|---|---|---|
| 24 manifest 무결성 | PASS | index hash, parent hash, source hash, NPZ hash |
| nested/subset/evaluation split | PASS | v2/원본 subset JSON/make_time_splits 대조 |
| Baseline/PINN 입력 텐서 동일 | PASS | 24개×2모드 main setup, GPU 동일 batch |
| Full statistics 덮어쓰기 방지 | PASS | Full builder/RHS fitting 호출 시 실패하도록 runtime 검사 |
| 육지값 영향 제거 | PASS | 상수장·0/100/NaN/Inf·실제661격자 검사 |
| loss/RHS/diagnostic 일관성 | PASS | P0/P1 same stencil/support 수식 대조 |
| 비활성 loss 계산 생략 | PASS | 함수 호출횟수·NaN poison·실제 gradient 검사 |
| scale 양수/finite | PASS | 24/24 |
| 기존 MHW 기준 유지 | PASS | evaluator SHA 및 평가 함수 AST 보존 |
| 기존 결과 보존 | PASS | 기존 v1/v2 SHA, 결과 파일 size/mtime, 모델 정의 SHA 확인 |

48개 main setup/CPU objective 검사는 학습 없이 수행했고, 실제 public v3 disk loader도 별도 실행했다. GPU smoke는 이 gate가 모두 PASS인 후에만 실행했다. 생성·검증 시 누락된 support의1개 격자는 physics 집계만 제외하며 data/평가 split 또는 마스크를 변경하지 않았다.

## E. 최종 결론

**PASS — 정식 학습 가능.** 수정된 해안 공간차분과 subset-only 통계가 함께 적용된 v3에서 요청한 회귀검사 및 제한 GPU smoke를 통과했다. 정식 자료량별 학습 및 Ablation Study는 실행하지 않았다.

후속 실행에서는 `--subset-manifest data/subsets/preprocessing_subset_v3/ratio_...json`을 명시해야 한다. Physics only는 `--physics-mode p0_original --lambda-rres 0 --lambda-we-tie 0`을 사용한다. 현재 CLI의 Rres 기본값10을 그대로 두면 Full PINN 조건이므로 조건을 명시해야 한다. 기존 v2의72-run 실행 계획이나 _queue snapshot은 이번 학습 설정으로 자동 전환되지 않는다. 이번 범위에서는 정식 run queue를 생성하거나 시작하지 않았다.

새 산출물: v3 index/24manifest/validation_report, `rhs_scales.csv`, `rhs_residual_comparison.json`, `runtime_verification.json`, `loss_arms.json`, GPU smoke 결과·checkpoint, 이 보고서. 수정 전 소스는 `originals/`에 보존했다. 모든 변경과 실제 실행 시각·이유는 `docs/research_log.md` 하나에 누적했다.
'''
 (O/'report.md').write_text(report)
 (R/'docs/subset_preprocessing_v3.md').write_text('# preprocessing_subset_v3\n\n해안 공간차분 수정,24개 RHS scale 갱신 및 제한 GPU smoke를 통과했습니다. 정식 학습은 실행하지 않았습니다.\n\n[최종 A–E 보고서](../outputs/v3_preflight_20261008/report.md) · [v3 index](../data/subsets/preprocessing_subset_v3/index.json) · [검증 요약](../data/subsets/preprocessing_subset_v3/validation_report.json)\n\nv3는 v2의 climatology NPZ를 경로와 해시로 참조합니다. v2를 함께 보존하세요. Input/Target 통계와 선택 인덱스는 동일하며 RHS만 보정된 해안 stencil로 재계산했습니다. 새 physics 코드에서 legacy v1/v2 RHS를 적용하면 오류로 중단합니다.\n\nPhysics only 실행 조건은 `--physics-mode p0_original --lambda-rres 0 --lambda-we-tie 0`으로 명시해야 합니다. 기존72-run v2 실행 계획을 자동 재사용하지 마세요.\n')
 readme=R/'README.md';readme.write_text(readme.read_text()+'\n\n### 해안 차분 보정 v3 (2026-10-08)\n\n[Subset-only v3](docs/subset_preprocessing_v3.md)는 보정된 해양 이웃 차분과24개 subset RHS scale을 사용합니다. 회귀검사와 Baseline/P0 각2batch GPU smoke를 통과했으며 본 학습은 미실행입니다. v1/v2 및 기존 결과는 보존했습니다.\n')
 save(O/'final_verdict.json',dict(verdict='PASS — 정식 학습 가능',timestamp_kst=ts,regression_tests=24,cpu_main_setups=48,actual_model_cpu_arms=4,gpu_smoke_models=2,gpu_forward_batches_total=4,gpu_backward_batches_total=4,optimizer_updates_total=2,formal_training_started=False,all_ten_gates='PASS',existing_artifacts_preserved=True))
 with (R/'docs/research_log.md').open('a') as f:f.write(f'\n\n## {ts} — v3 최종 PASS 보고 및 사용 안내\n\n- scripts/verify_v3_loss_arms.py의 실제CPU4조건 loss/gradient 검사 완료. scripts/report_subset_preprocessing_v3.py, outputs/v3_preflight_20261008/report.md·final_verdict.json·rhs_scales.csv, docs/subset_preprocessing_v3.md 및 README 안내를 저장했다. v3 index/validation 상태를 CPU 대기에서 최종PASS로 갱신했다.\n- 24개 회귀검사/48개main setup/4개CPU손실조건/2개GPU smoke 통과. GPU 각2batch 및각1update만 실행. 모든 v1/v2·기존 결과·MHW평가·모델 정의 보존을 재확인했다.\n- native baseline과aux head 모델의 parameter 수/실제gradient연결 수 차이, Rres강도 변경 관측, GPU smoke 범위와한계도 기록했다. 정식 학습 및ablation 미실행.\n')
 print('Final v3 PASS; all requested artifacts written')

if __name__=='__main__':main()
