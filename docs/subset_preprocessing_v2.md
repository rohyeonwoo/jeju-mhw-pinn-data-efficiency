# preprocessing_subset_v2 구축·검증 보고서

작성: 2026-10-08 14:29 KST. 본 학습·optimizer step·GPU 실행 없음.

## 결과와 범위

기존 raw 및 preprocessing_subset_v1을 보존하고 10/25/50/100% × seed 42/7/123 × lead 7/14의 **24개 manifest**를 별도 생성했다. 동일 ratio/seed의 두 lead는 input-history 날짜가 같으므로 **12개 climatology NPZ**를 공유한다. 기존 subset start index, nested 구조 및 Train/Validation/Test 분할은 유지했다.

v2는 raw NetCDF를 24번 복제하는 방식이 아니다. 원본 raw + subset climatology NPZ + subset 통계 manifest를 로더에서 조합한다. 생성 결과는 `data/subsets/preprocessing_subset_v2/`에 있다. 100%는 기존 block-constrained subset 3,995개이며 원래 Full 학습 샘플 7,644/7,639개와 다르다.

| 비율 | 선택 샘플 수 | 고유 input-history 날짜 수 |
|---|---:|---:|
| 10% | 376 | 608 |
| 25% | 987 | 1,596 |
| 50% | 1,974 | 3,192 |
| 100% | 3,995 | 6,460 |

## Climatology와 통계 정의

- SST 원료: 최종 raw NetCDF의 `sst_mean`. thetao 원료: `/ceph_data/yeonwoo/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d.nc`의 `thetao`, depth index 0/10/15/20/22/24. 기존 anomaly 값을 새 anomaly 생성에 사용하지 않는다.
- 각 manifest의 `input_time_indices`만 climatology 학습에 사용한다. 중복 history 날짜는 한 번만 포함한다. 고정된 전체 연도 구간을 다시 평균하지 않는다.
- 평년·윤년 모두 월/일을 윤년 기준 366 bin에 정렬한다. 3월 1일은 항상 같은 bin이다. 각 해양 격자·bin의 선택된 유한 관측 평균을 구한다.
- 빈 bin은 해당 격자의 관측이 있는 가장 가까운 양쪽 bin 사이를 **366일 주기 선형보간**한다. 연말/연초에도 같은 규칙이다. Full climatology 또는 미선택 관측을 참조하지 않는다.
- 2월 29일은 별도 bin이다. 선택 관측이 있으면 그 평균, 없으면 같은 주기 보간을 적용한다.
- 육지는 fit에서 제외하고 climatology/anomaly를 0으로 저장한다. fit은 NaN을 제외하되 해양 격자당 지원 bin이 2개 미만이면 실패한다. 실제 anomaly 적용 시 raw 해양 NaN/Inf는 조용히 채우지 않고 오류를 낸다. 이번 raw에서 해양 anomaly 비유한 값은 0개였다.
- Input mean/std: 새 anomaly를 포함해 각 변수별 선택된 고유 history 날짜 × 해양 격자에서 계산한다. population std(ddof=0)이며 train/val/test 모두 이 통계를 적용한다.
- Target mean/std: 선택 sample의 실제 target 날짜 × 해양 격자의 raw SST로 재계산했다. anomaly 변경과 독립적이며 **24/24 v1과 정확히 동일**하다.
- Physics RHS scale: 선택 sample의 raw SST 선형 경로와 forcing으로 계산한 resolved RHS의 해양 mean square, 단위 `(K/day)^2`이다. std가 아니다. 정규화는 residual을 sqrt(scale)로 나눈 뒤 제곱하는 것과 같다. **24/24 재계산 결과 v1과 정확히 동일**하다.
- Validation/Test target 및 미선택 관측은 새 학습 통계 fit에 사용하지 않는다. 원래 시간순 분할의 history 중첩은 유지하므로 같은 과거 관측이 선택 training history와 validation history에 나타날 수 있다. 전체 기간 anomaly 분포를 보는 QC는 진단용이며 fit에 되먹이지 않았다.

## 검증 결과

| 항목 | 결과 | 실제 확인 범위 |
|---|---|---|
| Subset 이외 관측 영향 차단 | PASS | 12개 그룹 × 7변수에서 모든 미선택 raw 날짜를 12345로 바꿔 재계산. climatology, 선택 anomaly, 해당 input mean/std가 정확히 동일 |
| Baseline/PINN 입력·통계 일치 | PASS | 24개 manifest × 2모드의 실제 main 초기화 경로, 입력·target·forcing 텐서 비교 및 optimizer 없는 CPU loss 검사 |
| DOY/결측/극단값/분산 검사 | PASS (명시한 수치 기준 내) | 84개 변수-그룹 조합의 해양 유한값, 육지 0, 양의 train 분산, climatology의 선택 raw 범위 보존, z값/분산비 검사 |
| v2 진입점·Full 덮어쓰기 방지 | PASS | Full bundle 생성 및 RHS 재계산 함수가 호출되면 실패하도록 검사. manifest 통계/scale 적용. 실제 public disk loader도 별도 실행 |
| 기존 subset/nested/split 유지 | PASS | 원본 subset JSON 및 make_time_splits와 대조. v1 디렉터리 전후 해시 동일 |
| Target/RHS 통계 유효성 | PASS | 24/24 각각 재계산 후 v1과 정확히 동일 |
| MHW 평가 보존 | PASS | mhw_skill.py 해시 보존, add_acc_and_sw_metrics/anomaly_acc/run_epoch 함수 AST가 수정 전과 동일 |

회귀 검사 12개가 통과했다(기존 5개 + 새 7개). 실제 학습 진입점 검증에서는 모델을 결정론적 Probe로 교체하고 optimizer=None으로 실행했다. 따라서 전처리 전달과 loss 산술의 검증이며 실제 모델 학습·수렴 검증은 아니다. 독립 재검사에서 24개 manifest 해시, 12개 climatology 해시, manifest에 기록된 소스 해시, 원본 인덱스/분할 및 72개 실행 계획을 확인했다.

현재 raw를 고정한 이후의 v2 전처리는 subset-only 조건을 충족한다. 이 검증은 기존 raw 제품의 생산·공간 보간·결측 채움 전체 과정까지 새로 감사했다는 뜻은 아니다.

## 10% 희소 DOY 진단과 한계

| seed | 고유 입력 날짜 | 빈 bin 수 | 관측 1개 bin 수 | Feb29 관측 수 | 7변수 train std 범위(K) | 전체 날짜 최대 abs(z) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 608 | 34 | 104 | 0 | 0.634–0.800 | 27.305 |
| 7 | 608 | 9 | 134 | 1 | 0.539–0.577 | 29.942 |
| 123 | 608 | 52 | 72 | 0 | 0.569–0.761 | 27.541 |

관측 1개인 bin에서는 그 선택 관측이 자기 climatology이므로 학습 anomaly가 0이 된다. 이 영향으로 10%의 v2/v1 train std 비율은 7변수·3seed에서 약 0.453–0.799이다. 전체 기간/train std 비율은 약 1.560–2.353이다. 이 감소를 숨기거나 보정하기 위해 Full 통계, clipping 또는 임의 smoothing을 사용하지 않았다.

사전 선언한 수치 검토 기준은 train max abs(z) ≤ 20, 전체 기간 max abs(z) ≤ 30, 전체/train std 비율 0.1–10이다. 최대 abs(z)=29.942는 상한에 가깝다. **PASS는 유한성 및 이 수치 기준의 통과이지 희소 climatology의 통계적 정확도나 모든 극단값의 물리적 타당성 보증이 아니다.** 608일의 희소성과 단일 관측 bin의 분산 축소를 결과 해석에 함께 명시해야 한다. 모든 366개 관측 수와 변수별 분위수는 `climatology_qc.json`에 있다.

## 구현 근거

프로젝트 루트는 `/ceph_data/yeonwoo/MHW_subset/`이다.

| 파일:행 | 함수/역할 |
|---|---|
| `src/mhw/subset_climatology.py:18-54` | calendar_bins, fit_climatology: 월/일 정렬, 선택 관측 평균, 빈 bin 주기 보간 |
| `src/mhw/subset_climatology.py:57-89` | anomaly, load_raw_anomaly_fields, apply_to_base: raw SST/thetao에서 입력 anomaly만 교체 |
| `src/mhw/subset_climatology.py:97-132` | read_climatology, bundle_from_v2, load_v2_bundle: 저장 해시 확인과 v2 적용 |
| `src/mhw/subset_preprocessing.py:37-83` | selected_time_indices, fit_subset_stats: 고유 history/target 분리와 통계 계산 |
| `src/mhw/subset_preprocessing.py:94-124` | PhysicsScaleDataset, compute_subset_rhs_scale: 선택 sample RHS 집계 |
| `src/train_hybrid_jeju_forecast.py:810-837` | compute_fixed_rhs_scale: RHS 제곱합 / 해양 count |
| `src/mhw/subset_preprocessing.py:135-203` | bundle_from_manifest, load_subset_bundle, apply_manifest_physics: 동일 stats 적용 및 scale 충돌 차단 |
| `src/train_jeju_hilam.py:625-700` | main: --subset-manifest 분기, target 통계 및 physics scale 적용 |
| `scripts/build_subset_preprocessing_v2.py:61-117` | verify_entrypoint, quality: 실제 진입점/CPU loss와 수치 QC |
| `scripts/build_subset_preprocessing_v2.py:148-265` | main: 관측 변조 검사, v1 보존, 24개 생성·검증 |
| `tests/test_subset_climatology.py:8-62` | 달력/윤일/순환보간/누수/육지/NaN/순환 stream/출력 경로 검사 |

## 실행 계획 — 아직 실행하지 않음

`execution_plan.json`에 모든 manifest와 argv를 저장했다.

| 실험 | 구성 | 실행 수 |
|---|---|---:|
| 자료량 비교 | 4비율 × 2lead × Baseline/P0 × 3seed | 48 |
| 10% ablation | Baseline/Physics only/Auxiliary only/Full PINN × 2lead × 3seed | 24 |

자료량 비교에서 paired Baseline/P0는 동일 manifest를 사용한다. Ablation 네 조건은 모두 보조 head를 포함한 동일 구조와 seed를 사용하고 lambda_phys/lambda_rres만 (0,0), (0.1,0), (0,10), (0.1,10)으로 바꾼다. Ablation Baseline은 보조 head가 없는 주 비교 Baseline과 구조가 다르므로 별도 run으로 계획했다. unresolved residual을 resolved physics 식에서 제외하는 설정은 유지한다.

공통 계획: history 30, batch 16, fp32, lr 0.0002(initial 0.00002), weight decay 0.1, virtual epoch당 478 optimizer step, 최대 80 virtual epoch(38,240 step), warmup 5 virtual epoch, 478 step마다 validation, val RMSE 기준 patience 12. 이는 동일 최대 step 예산과 validation 간격이며, early stopping에 따라 실제 총 step은 달라질 수 있다. `--virtual-epoch-steps`가 loader를 순환하며 epoch 사이 남은 위치를 유지한다. 기본 옵션 미지정 동작은 기존 epoch 방식이다.

평가 계획은 SST RMSE/MAE, 기존 MHW 평가 기준, physics residual이다. 이번 입력 climatology 변경은 MHW 기준/임계값 변경과 분리했다. 후속 GPU 실행 시에는 Ceph 출력·TMPDIR·캐시 경로를 지정하고 CUDA 사용 전에 process title을 yeonwoo로 설정하며, 시작 후 nvidia-smi/PID의 이름까지 검증해야 한다. 이번 작업에서는 GPU를 사용하지 않았다.

## 산출물

- `data/subsets/preprocessing_subset_v2/index.json`: 24개 manifest 인덱스
- `data/subsets/preprocessing_subset_v2/validation_report.json`: 네 가지 gate 및 보존 검사 요약
- `data/subsets/preprocessing_subset_v2/climatology_qc.json`: 366 bin 관측 수와 변수별 분포
- `data/subsets/preprocessing_subset_v2/runtime_verification.json`: 학습 진입점 검사 결과
- `data/subsets/preprocessing_subset_v2/execution_plan.json`: 48+24개 실행 계획(미실행)
- `docs/research_log.md`: 실제 KST 수정 시각과 이유를 누적한 단일 변경 이력
