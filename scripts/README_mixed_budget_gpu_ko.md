# v7 개선 혼합층 열수지 파일럿 — 사용자가 직접 실행

프로젝트: `/ceph_data/yeonwoo/MHW_subset`.
환경 Python: `/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python`.
출력: `outputs/v7_gpu_mixed_budget_comparison_20261011/`.

GPU는 물리 번호 **0, 1, 2**만 허용하고 GPU당 최대 **1개 worker**를 실행한다. GPU 번호→UUID 일치를 CUDA 초기화 후 확인하며 process title은 CUDA 사용 전에 `yeonwoo`로 설정한다. GPU3 이상은 사용하지 않는다. 기존 adaptive 관리자나 학습은 재개하지 않는다.

## 고정 설계

- 10%·Lead7 / 50%·Lead14 × seeds42,7,123 × B0/B1/A/B = 24회.
- B0: 데이터손실만. B1: 기존 P0(학습 we, 고정1K). A: 투과열+MLD 연행 proxy. B: A+이류.
- 모든 모델에서 동일 native HiLAM predictor/we/rres head를 생성한다. B0/A/B의 보조 head는 objective에서 사용하지 않아 gradient가 없다. B1의 we head만 물리손실에서 학습한다. Rres는 전 조건 비활성.
- 별도 혼합층 예측 head를 추가하지 않는다. **NOAA SST는 혼합층 수온 proxy**이므로 엄밀한 GLORYS 혼합층 예측식이라고 주장하지 않는다.
- GLORYS 미래 H/Tb/유속, ERA5 미래 flux는 **loss-only oracle forcing**이다. x_nodes·tm_norm_nodes·모델 forward에는 미래 자료를 넣지 않는다.
- 원래 v3 subset 인덱스·subset-only anomaly·Input/Target 통계 유지. 학습 비교 B1/A/B는 각 subset의 **동일한 기존 v3 RHS mean-square scale**을 적용한다. 식별 자체 RHS mean-square는 `rhs_scale_comparison.csv` 진단만 하며 나누는 값으로 바꾸지 않는다.
- λphysics=0.1 고정, physics warmup=0, λrres=λwe_tie=0, zero_rres_physics=True. optimizer LR warmup5는 기존대로 유지한다. 과거 λ=.01 실험 결과를 재사용하지 않는다.
- batch16, AdamW(betas .9/.95, weight decay .1), gradient clip1, native LR schedule, 최대80×478=38,240 updates, 매478 updates 전체 Validation SST RMSE, patience12회. 같은 최대 예산이며 early stopping 때문에 실제 종료 update는 모델별로 다를 수 있다.
- Kd=.06/m, rho1025, Cp3990, H_min5m, clip±10 K/day 후 공통 RHS 정규화.
- we_proxy=max(0,max(H[t],5)−max(H[t−1],5))/86400. CPU v6의 forward H[t+1]−H[t]와 시간 배치가 다르다. Tb는 max(H[t],5)보다 깊은 첫 유효 층 중심 thetao. 층 두께는 v6의 중심간 중점 근사.
- A/B는 이류 포함 여부만 다르다. B1→A/B는 투과·연행·유속 평균 등 여러 변경이므로 단일 물리항 효과로 해석하지 않는다.

## CPU 준비 상태

`preflight_report.json`: CPU_PASS_GPU_PENDING이면 CPU만 통과한 상태다. GPU PASS가 아니다.
`cpu_unit_tests.json`: Kd0, 일정 온도, 유속0, 육지 NaN, A/B RHS 차이, future target 미사용, predictor 미분.
`subset_input_audit.json`: 6개 조건의 분할/기존 통계/입력 동일성, 4조건 모델 초기화 동일성 및 기존 NumPy 차분과의 비교.
`rhs_scale_comparison.csv`: A/B 자체 RHS 크기와 공통 적용 스케일을 구분.
`cpu_entrypoint_check.json`: 실제 HiLAM batch2에서 forward·autograd 및 Validation 루프 점검. optimizer step은 0이며 GPU batch16 smoke를 대체하지 않는다.

코드·데이터가 현재 준비된 상태이면 CPU 준비를 반복할 필요 없다. 새 환경에서 재현하려면 GPU를 숨겨 아래 순서로 실행한다. 기존 불완전 캐시는 임의 덮어쓰지 않고 중단한다.

```bash
cd /ceph_data/yeonwoo/MHW_subset
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B scripts/prepare_mixed_budget_features.py
/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B scripts/validate_mixed_budget_training.py
```

## 1. GPU smoke — 먼저 이것만 실행

```bash
cd /ceph_data/yeonwoo/MHW_subset
bash scripts/run_mixed_budget_tmux.sh smoke
```

10%·Lead7·seed42에서 B0/B1/A/B 각각 2개 batch forward/backward 및 optimizer step. 실제 v3 입력+신규 forcing을 사용한다. 검증용 체크포인트 roundtrip과 Validation 첫 batch 출력 재현을 확인한다. 각 모델의 loss 유한성, gradient norm/clip, predictor physics/data gradient norm·cosine, GPU peak memory가 `smoke/<모델>/status.json`에 저장된다.

```bash
cat outputs/v7_gpu_mixed_budget_comparison_20261011/smoke_test_report.json
```

기술적 PASS라도 `weighted_ratio`와 `cosine`을 검토한다. PASS가 λ=.1의 최적성이나 성능 개선을 보장하지 않는다. 실패 시 다음 단계가 차단되며 자동 재시작하지 않는다.

## 2. 대표 학습 4회

smoke 확인 후 **별도로** 실행한다. 대표 조건은 10%·Lead7·seed42, 네 모델 각각 한 번이다.

```bash
bash scripts/run_mixed_budget_tmux.sh representative
```

최대 동시3회이므로 첫3개 완료 자리에 4번째가 실행된다. 네 학습과 checkpoint reload·Validation 재현, 공통 구간 batch 순서 검증 후 `representative_report.json=PASS`가 된다. best epoch/gradient 비율/학습 안정성을 확인한 뒤 나머지를 실행한다.

## 3. 나머지 20회

```bash
bash scripts/run_mixed_budget_tmux.sh remaining
```

정해진20개만 처리하고 25%/100%나 다른 lead로 자동 확장하지 않는다.

## 4. 모든24회 완료 후 평가 및 CPU 요약

```bash
bash scripts/run_mixed_budget_tmux.sh evaluate
```

Validation 재현 후 Test를 평가한다. 학습 종료와 평가 종료는 다르다. `training_status.json`의 stage=evaluate, status=stage_complete를 확인한 뒤 CPU 요약:

```bash
CUDA_VISIBLE_DEVICES="" /ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B scripts/summarize_mixed_budget_gpu.py
```

SST RMSE/MAE/Bias/ACC, 해안·계절, 기존 MHW 기준의 grid-day 및 domain-area event 지표, MHW일 SST 오차, missed/recovered grid-day, 공통 A/B raw residual, seed평균/SD 및 paired 시간블록 구간을 출력한다. MHW event 지표를 grid-day CSI와 혼용하지 않는다. Test는 이미 여러 번 열람한 자료이므로 탐색적 평가로 표기한다.

MHW threshold는 원래 raw SST 첫 round(70%×전체일수) 기간에서 계산한다. 이 기간 끝이 현재 Validation/Test target 시작 전인지 CPU 사전 검증한다. 학습용 subset-only anomaly와 별개이며 이번 작업에서 평가 기준을 바꾸지 않는다.

## 확인·중단·재시도

```bash
cat outputs/v7_gpu_mixed_budget_comparison_20261011/training_status.json
tmux -S /ceph_data/yeonwoo/MHW_subset/outputs/v7_gpu_mixed_budget_comparison_20261011/runtime/tmux.sock list-sessions
```

세션은 `v7_smoke`, `v7_representative`, `v7_remaining`, `v7_evaluate`. 단계별 manager 로그와 모델별 로그/PID는 outputs/.../logs 및 launch_*.json에 보존한다. 각 단계는 다음 단계를 자동 시작하지 않는다. 이미 같은 tmux 세션이 있으면 새로 시작하지 않는다. 다른 사용자 GPU 점유를 발견하면 중단하며 다른 프로세스를 종료하지 않는다. 실패한 경우 상태 파일을 임의로 지우거나 PASS로 고치지 말고 오류·마지막 checkpoint를 확인한다. epoch 경계 last.pt에는 optimizer/RNG/data stream 상태가 포함되어 수동 복구 검토가 가능하다.

원본 src, raw/v1/v2/v3, 기존 checkpoint/결과는 보존한다. 실행·변경 기록은 `docs/research_log.md` 한 파일에 KST로 누적한다.
