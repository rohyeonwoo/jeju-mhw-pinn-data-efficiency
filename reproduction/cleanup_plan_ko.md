# 재현 필수 파일 보존 및 정리 제안 — 삭제 미실행

- 조사한 원본: 64.374 GiB
- 정리 후보: **1562개 파일, 47.236 GiB**
- 적용 후 원본 예상 잔여량: **17.138 GiB**
- 이 중 완료된 최종 학습의 last/post-warmup checkpoint: 216개, 12.847 GiB

논리 파일 크기 합계이며 Ceph 스냅샷/할당/압축에 따른 물리적 반환량과 다를 수 있다. 새 GitHub export/압축본은 별도 공간을 사용한다.

## 반드시 남김

1. `src/`, `scripts/`, `tests/`, subset 설계·전처리(v1/v2/v3)·원자료 및 config/manifest/학습 history
2. **최종120 Validation-best checkpoint 전부**: `final_checkpoint_inventory.csv`
3. 최종120 저장 예측·MHW 라벨·평가, 최종 검증과 논문 Figure/Table/plot data/통계/보고서
4. 실제 해시 보호 목록이 참조하는 과거 파일과 v7 물리 feature 배열
5. 기존 WSDM/Full 자료와 이전 논문분석: 별도 근거로 유지
6. 외부3D GLORYS 원자료(프로젝트 밖3.19GiB): `external_dependencies.csv`; 이 작업의 정리 범위 밖

## 삭제 대상의 정확한 범위

폴더 전체가 아닌 `cleanup_candidates.csv`의 경로만 대상이다.

- 최종120에 쓰이지 않는 중간·pilot checkpoint와 예측/진단 배열(보호 해시 참조 파일 제외)
- 완료된120개 학습의 `last.pt`, 및 Oracle96의 `post_warmup_best.pt`. 최종 `best.pt`는 항상 남긴다. 삭제 후 중단 지점의 optimizer/RNG 상태 재개와 이 대체 checkpoint로 평가하는 기능은 잃는다. 처음부터 재학습하거나 최종 선택 모델로 평가하는 데는 사용하지 않는다.
- 중간 pilot을 saved predictions/weights에서 다시 분석하는 기능은 해당 후보 삭제 시 잃을 수 있다. 작은 수치/설정/보고서와 실행기 코드는 유지한다.

| 폴더 | 후보 파일 | GiB |
|---|---:|---:|
| `data_efficiency` | 587 | 16.564 |
| `v8_actual_epoch_oracle96_20261011` | 192 | 11.356 |
| `aux_weight_selection_20261009` | 101 | 6.080 |
| `v3_physics_ratio_lead_lambda001_20261009` | 71 | 2.891 |
| `v3_cpu_diagnostics_20261011` | 74 | 2.393 |
| `p0_lambda_v3_20261009` | 31 | 1.858 |
| `p0_dense_validation_20261009` | 26 | 1.541 |
| `v8_actual_epoch_auxiliary24_20261015` | 24 | 1.491 |
| `v3_adaptive_physics_pilot_20261011` | 18 | 1.084 |
| `p0_repair_20261009` | 13 | 0.768 |
| `v3_ablation_lead7_20261009` | 8 | 0.496 |
| `v5_cpu_horizontal_advection_audit_20261011` | 151 | 0.261 |
| `v3_generalization_20261009` | 32 | 0.240 |
| `v3_preflight_20261008` | 4 | 0.123 |
| `v4_cpu_physics_audit_20261011` | 144 | 0.042 |
| `v5_cpu_advection_root_cause_20261011` | 74 | 0.040 |
| `v2_final_preflight_20261008` | 12 | 0.008 |

## 현재 상태

**삭제·이동 및 GitHub 업로드 모두 미실행.** GitHub 준비 사본은 코드/작은 설정·결과·그림만 포함한다. 대용량 데이터와 모델의 대체 백업이 아니다. 최종120 가중치와 데이터까지 Ceph에서 지우면 GitHub만으로 기존 모델을 그대로 복구할 수 없다.

AGENTS.md의 “파일 삭제, 이동, 모델 코드 수정은 반드시 목록을 먼저 보여주고 승인을 받은 뒤 진행한다” 규칙에 따라 이 구체 목록의 최종 승인을 기다린다.
