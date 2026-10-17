# Actual Epoch Oracle96 Validation 보고

2026-10-12 13:48 KST

완료 96/96. 미완료 run은 성능 집계에서 제외한다. Test 평가: 실행하지 않음.

## 모델·자료량·Lead별 Validation

| Ratio | Lead | Model | 완료 seeds | Val RMSE mean ± sample SD | Seed별 개선율 평균 | 개선 seeds |
|---|---|---|---|---|---|---|
| 10 | 7 | Adv | 3 | 1.024691 ± 0.020575 | -1.700% | 1/3 |
| 10 | 7 | Baseline | 3 | 1.008026 ± 0.026757 | 0.000% | 0/3 |
| 10 | 7 | NoAdv | 3 | 0.976983 ± 0.025293 | 3.045% | 3/3 |
| 10 | 7 | P0 | 3 | 1.012521 ± 0.028637 | -0.474% | 1/3 |
| 10 | 14 | Adv | 3 | 1.241691 ± 0.014507 | -0.675% | 1/3 |
| 10 | 14 | Baseline | 3 | 1.233606 ± 0.024532 | 0.000% | 0/3 |
| 10 | 14 | NoAdv | 3 | 1.222871 ± 0.038248 | 0.866% | 1/3 |
| 10 | 14 | P0 | 3 | 1.231175 ± 0.011754 | 0.181% | 2/3 |
| 25 | 7 | Adv | 3 | 0.922328 ± 0.014525 | -4.519% | 0/3 |
| 25 | 7 | Baseline | 3 | 0.882566 ± 0.008307 | 0.000% | 0/3 |
| 25 | 7 | NoAdv | 3 | 0.902334 ± 0.004340 | -2.249% | 0/3 |
| 25 | 7 | P0 | 3 | 0.886279 ± 0.013665 | -0.434% | 1/3 |
| 25 | 14 | Adv | 3 | 1.108001 ± 0.020804 | -1.932% | 0/3 |
| 25 | 14 | Baseline | 3 | 1.086967 ± 0.014083 | 0.000% | 0/3 |
| 25 | 14 | NoAdv | 3 | 1.084479 ± 0.026097 | 0.219% | 2/3 |
| 25 | 14 | P0 | 3 | 1.090634 ± 0.031157 | -0.328% | 1/3 |
| 50 | 7 | Adv | 3 | 0.901196 ± 0.008653 | -6.065% | 0/3 |
| 50 | 7 | Baseline | 3 | 0.849666 ± 0.004697 | 0.000% | 0/3 |
| 50 | 7 | NoAdv | 3 | 0.875248 ± 0.009276 | -3.011% | 0/3 |
| 50 | 7 | P0 | 3 | 0.853487 ± 0.013542 | -0.450% | 1/3 |
| 50 | 14 | Adv | 3 | 1.058979 ± 0.010472 | -3.646% | 0/3 |
| 50 | 14 | Baseline | 3 | 1.021854 ± 0.016427 | 0.000% | 0/3 |
| 50 | 14 | NoAdv | 3 | 1.050701 ± 0.010453 | -2.830% | 0/3 |
| 50 | 14 | P0 | 3 | 1.020917 ± 0.024177 | 0.098% | 1/3 |
| 100 | 7 | Adv | 3 | 0.880486 ± 0.006169 | -7.638% | 0/3 |
| 100 | 7 | Baseline | 3 | 0.818007 ± 0.000783 | 0.000% | 0/3 |
| 100 | 7 | NoAdv | 3 | 0.864399 ± 0.003767 | -5.672% | 0/3 |
| 100 | 7 | P0 | 3 | 0.831603 ± 0.007926 | -1.662% | 0/3 |
| 100 | 14 | Adv | 3 | 1.016693 ± 0.005058 | -4.841% | 0/3 |
| 100 | 14 | Baseline | 3 | 0.969752 ± 0.000990 | 0.000% | 0/3 |
| 100 | 14 | NoAdv | 3 | 1.001807 ± 0.001768 | -3.306% | 0/3 |
| 100 | 14 | P0 | 3 | 0.982486 ± 0.004640 | -1.313% | 0/3 |

## Best 및 warmup 진단

전체 best가 full-weight 시작(epoch5) 전인 run: 0/96. 이 run들의 best로 완전히 활성화된 물리 제약의 효과를 입증했다고 주장하지 않는다.

| Run | Overall best epoch | Post-warmup best epoch | Overall roundtrip | Post roundtrip |
|---|---|---|---|---|
| Baseline_r10_lead7_seed42 | 14 | 14 | PASS | PASS |
| P0_r10_lead7_seed42 | 20 | 20 | PASS | PASS |
| NoAdv_r10_lead7_seed42 | 43 | 43 | PASS | PASS |
| Adv_r10_lead7_seed42 | 44 | 44 | PASS | PASS |
| Baseline_r10_lead7_seed7 | 21 | 21 | PASS | PASS |
| P0_r10_lead7_seed7 | 8 | 8 | PASS | PASS |
| NoAdv_r10_lead7_seed7 | 20 | 20 | PASS | PASS |
| Adv_r10_lead7_seed7 | 20 | 20 | PASS | PASS |
| Baseline_r10_lead7_seed123 | 21 | 21 | PASS | PASS |
| P0_r10_lead7_seed123 | 21 | 21 | PASS | PASS |
| NoAdv_r10_lead7_seed123 | 28 | 28 | PASS | PASS |
| Adv_r10_lead7_seed123 | 34 | 34 | PASS | PASS |
| Baseline_r10_lead14_seed42 | 17 | 17 | PASS | PASS |
| P0_r10_lead14_seed42 | 17 | 17 | PASS | PASS |
| NoAdv_r10_lead14_seed42 | 17 | 17 | PASS | PASS |
| Adv_r10_lead14_seed42 | 26 | 26 | PASS | PASS |
| Baseline_r10_lead14_seed7 | 12 | 12 | PASS | PASS |
| P0_r10_lead14_seed7 | 10 | 10 | PASS | PASS |
| NoAdv_r10_lead14_seed7 | 20 | 20 | PASS | PASS |
| Adv_r10_lead14_seed7 | 11 | 11 | PASS | PASS |
| Baseline_r10_lead14_seed123 | 30 | 30 | PASS | PASS |
| P0_r10_lead14_seed123 | 30 | 30 | PASS | PASS |
| NoAdv_r10_lead14_seed123 | 26 | 26 | PASS | PASS |
| Adv_r10_lead14_seed123 | 26 | 26 | PASS | PASS |
| Baseline_r25_lead7_seed42 | 28 | 28 | PASS | PASS |
| P0_r25_lead7_seed42 | 19 | 19 | PASS | PASS |
| NoAdv_r25_lead7_seed42 | 20 | 20 | PASS | PASS |
| Adv_r25_lead7_seed42 | 19 | 19 | PASS | PASS |
| Baseline_r25_lead7_seed7 | 20 | 20 | PASS | PASS |
| P0_r25_lead7_seed7 | 24 | 24 | PASS | PASS |
| NoAdv_r25_lead7_seed7 | 46 | 46 | PASS | PASS |
| Adv_r25_lead7_seed7 | 35 | 35 | PASS | PASS |
| Baseline_r25_lead7_seed123 | 17 | 17 | PASS | PASS |
| P0_r25_lead7_seed123 | 22 | 22 | PASS | PASS |
| NoAdv_r25_lead7_seed123 | 37 | 37 | PASS | PASS |
| Adv_r25_lead7_seed123 | 26 | 26 | PASS | PASS |
| Baseline_r25_lead14_seed42 | 15 | 15 | PASS | PASS |
| P0_r25_lead14_seed42 | 28 | 28 | PASS | PASS |
| NoAdv_r25_lead14_seed42 | 28 | 28 | PASS | PASS |
| Adv_r25_lead14_seed42 | 28 | 28 | PASS | PASS |
| Baseline_r25_lead14_seed7 | 12 | 12 | PASS | PASS |
| P0_r25_lead14_seed7 | 24 | 24 | PASS | PASS |
| NoAdv_r25_lead14_seed7 | 41 | 41 | PASS | PASS |
| Adv_r25_lead14_seed7 | 26 | 26 | PASS | PASS |
| Baseline_r25_lead14_seed123 | 12 | 12 | PASS | PASS |
| P0_r25_lead14_seed123 | 34 | 34 | PASS | PASS |
| NoAdv_r25_lead14_seed123 | 16 | 16 | PASS | PASS |
| Adv_r25_lead14_seed123 | 16 | 16 | PASS | PASS |
| Baseline_r50_lead7_seed42 | 22 | 22 | PASS | PASS |
| P0_r50_lead7_seed42 | 22 | 22 | PASS | PASS |
| NoAdv_r50_lead7_seed42 | 22 | 22 | PASS | PASS |
| Adv_r50_lead7_seed42 | 22 | 22 | PASS | PASS |
| Baseline_r50_lead7_seed7 | 23 | 23 | PASS | PASS |
| P0_r50_lead7_seed7 | 39 | 39 | PASS | PASS |
| NoAdv_r50_lead7_seed7 | 37 | 37 | PASS | PASS |
| Adv_r50_lead7_seed7 | 25 | 25 | PASS | PASS |
| Baseline_r50_lead7_seed123 | 32 | 32 | PASS | PASS |
| P0_r50_lead7_seed123 | 11 | 11 | PASS | PASS |
| NoAdv_r50_lead7_seed123 | 39 | 39 | PASS | PASS |
| Adv_r50_lead7_seed123 | 34 | 34 | PASS | PASS |
| Baseline_r50_lead14_seed42 | 10 | 10 | PASS | PASS |
| P0_r50_lead14_seed42 | 11 | 11 | PASS | PASS |
| NoAdv_r50_lead14_seed42 | 18 | 18 | PASS | PASS |
| Adv_r50_lead14_seed42 | 29 | 29 | PASS | PASS |
| Baseline_r50_lead14_seed7 | 27 | 27 | PASS | PASS |
| P0_r50_lead14_seed7 | 18 | 18 | PASS | PASS |
| NoAdv_r50_lead14_seed7 | 24 | 24 | PASS | PASS |
| Adv_r50_lead14_seed7 | 28 | 28 | PASS | PASS |
| Baseline_r50_lead14_seed123 | 31 | 31 | PASS | PASS |
| P0_r50_lead14_seed123 | 33 | 33 | PASS | PASS |
| NoAdv_r50_lead14_seed123 | 15 | 15 | PASS | PASS |
| Adv_r50_lead14_seed123 | 33 | 33 | PASS | PASS |
| Baseline_r100_lead7_seed42 | 19 | 19 | PASS | PASS |
| P0_r100_lead7_seed42 | 29 | 29 | PASS | PASS |
| NoAdv_r100_lead7_seed42 | 21 | 21 | PASS | PASS |
| Adv_r100_lead7_seed42 | 23 | 23 | PASS | PASS |
| Baseline_r100_lead7_seed7 | 23 | 23 | PASS | PASS |
| P0_r100_lead7_seed7 | 23 | 23 | PASS | PASS |
| NoAdv_r100_lead7_seed7 | 46 | 46 | PASS | PASS |
| Adv_r100_lead7_seed7 | 54 | 54 | PASS | PASS |
| Baseline_r100_lead7_seed123 | 20 | 20 | PASS | PASS |
| P0_r100_lead7_seed123 | 25 | 25 | PASS | PASS |
| NoAdv_r100_lead7_seed123 | 40 | 40 | PASS | PASS |
| Adv_r100_lead7_seed123 | 50 | 50 | PASS | PASS |
| Baseline_r100_lead14_seed42 | 19 | 19 | PASS | PASS |
| P0_r100_lead14_seed42 | 14 | 14 | PASS | PASS |
| NoAdv_r100_lead14_seed42 | 19 | 19 | PASS | PASS |
| Adv_r100_lead14_seed42 | 19 | 19 | PASS | PASS |
| Baseline_r100_lead14_seed7 | 18 | 18 | PASS | PASS |
| P0_r100_lead14_seed7 | 14 | 14 | PASS | PASS |
| NoAdv_r100_lead14_seed7 | 18 | 18 | PASS | PASS |
| Adv_r100_lead14_seed7 | 18 | 18 | PASS | PASS |
| Baseline_r100_lead14_seed123 | 7 | 7 | PASS | PASS |
| P0_r100_lead14_seed123 | 19 | 19 | PASS | PASS |
| NoAdv_r100_lead14_seed123 | 35 | 35 | PASS | PASS |
| Adv_r100_lead14_seed123 | 19 | 19 | PASS | PASS |

## 기존 Virtual Epoch와 비교

| Run | Old updates | Actual updates | Old best virtual epoch | New best actual epoch | Old Val | New Val | New−Old K |
|---|---|---|---|---|---|---|---|
| Baseline_r10_lead7_seed42 | 38240 | 1920 | 4 | 14 | 1.065011 | 1.031431 | -0.033580 |
| P0_r10_lead7_seed42 | 38240 | 1920 | 1 | 20 | 1.048509 | 1.001946 | -0.046563 |
| NoAdv_r10_lead7_seed42 | 38240 | 1920 | 2 | 43 | 0.977975 | 0.963589 | -0.014385 |
| Adv_r10_lead7_seed42 | 38240 | 1920 | 3 | 44 | 1.040387 | 1.009734 | -0.030652 |

epoch 숫자는 서로 동일한 학습량이 아니다. 기존 10% 1 virtual epoch≈19.92순회; 신규 actual epoch=1순회. 학습 예산, Validation 빈도, LR의 update 기준 진행률, physics warmup(기존0→신규5)이 함께 바뀌었다. 변화 전체를 과적합 감소만의 인과효과로 설명하지 않는다.

## 해석 및 연구 질문

- 10% Lead7: Adv 1.024691 ± 0.020575, 개선 -1.700%, 개선 seeds 1/3; Baseline 1.008026 ± 0.026757, 개선 0.000%, 개선 seeds 0/3; NoAdv 0.976983 ± 0.025293, 개선 3.045%, 개선 seeds 3/3; P0 1.012521 ± 0.028637, 개선 -0.474%, 개선 seeds 1/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 0/3.

- 10% Lead14: Adv 1.241691 ± 0.014507, 개선 -0.675%, 개선 seeds 1/3; Baseline 1.233606 ± 0.024532, 개선 0.000%, 개선 seeds 0/3; NoAdv 1.222871 ± 0.038248, 개선 0.866%, 개선 seeds 1/3; P0 1.231175 ± 0.011754, 개선 0.181%, 개선 seeds 2/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 1/3.

- 25% Lead7: Adv 0.922328 ± 0.014525, 개선 -4.519%, 개선 seeds 0/3; Baseline 0.882566 ± 0.008307, 개선 0.000%, 개선 seeds 0/3; NoAdv 0.902334 ± 0.004340, 개선 -2.249%, 개선 seeds 0/3; P0 0.886279 ± 0.013665, 개선 -0.434%, 개선 seeds 1/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 0/3.

- 25% Lead14: Adv 1.108001 ± 0.020804, 개선 -1.932%, 개선 seeds 0/3; Baseline 1.086967 ± 0.014083, 개선 0.000%, 개선 seeds 0/3; NoAdv 1.084479 ± 0.026097, 개선 0.219%, 개선 seeds 2/3; P0 1.090634 ± 0.031157, 개선 -0.328%, 개선 seeds 1/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 0/3.

- 50% Lead7: Adv 0.901196 ± 0.008653, 개선 -6.065%, 개선 seeds 0/3; Baseline 0.849666 ± 0.004697, 개선 0.000%, 개선 seeds 0/3; NoAdv 0.875248 ± 0.009276, 개선 -3.011%, 개선 seeds 0/3; P0 0.853487 ± 0.013542, 개선 -0.450%, 개선 seeds 1/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 0/3.

- 50% Lead14: Adv 1.058979 ± 0.010472, 개선 -3.646%, 개선 seeds 0/3; Baseline 1.021854 ± 0.016427, 개선 0.000%, 개선 seeds 0/3; NoAdv 1.050701 ± 0.010453, 개선 -2.830%, 개선 seeds 0/3; P0 1.020917 ± 0.024177, 개선 0.098%, 개선 seeds 1/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 1/3.

- 100% Lead7: Adv 0.880486 ± 0.006169, 개선 -7.638%, 개선 seeds 0/3; Baseline 0.818007 ± 0.000783, 개선 0.000%, 개선 seeds 0/3; NoAdv 0.864399 ± 0.003767, 개선 -5.672%, 개선 seeds 0/3; P0 0.831603 ± 0.007926, 개선 -1.662%, 개선 seeds 0/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 0/3.

- 100% Lead14: Adv 1.016693 ± 0.005058, 개선 -4.841%, 개선 seeds 0/3; Baseline 0.969752 ± 0.000990, 개선 0.000%, 개선 seeds 0/3; NoAdv 1.001807 ± 0.001768, 개선 -3.306%, 개선 seeds 0/3; P0 0.982486 ± 0.004640, 개선 -1.313%, 개선 seeds 0/3

  Adv가 NoAdv보다 낮은 Val RMSE인 seed: 0/3.

## 일반화 곡선

각 조건의 80-epoch Train/Validation·LR·physics λ 곡선은 figures/에 저장했다. Training RMSE는 optimizer update 중 배치별 online 집계이다. 모델을 고정해 Train 전체를 재평가한 지표가 아니다.

## 한계와 Test 계획

3 seeds의 SD와 개선 seed 수를 함께 제시하며 통계적 확증을 주장하지 않는다. Training-only oracle은 SST-label efficiency 비교이며 total-information efficiency 비교가 아니다. NOAA SST는 혼합층 수온 대리변수, 투과식은 전체 단파복사의 단일지수 근사, Kd=.06은 파일럿 상수다. 기존 Test 결과를 확인한 이후 프로토콜을 변경했으므로 향후 같은 Test 평가도 탐색적 분석이다.

96개 COMPLETED 및 양 checkpoint 검증 PASS 후, 전체 Validation-best만 고정하고 별도 승인된 Test 평가를 수행한다. Post-warmup best는 진단용이며 Test로 둘 중 하나를 선택하지 않는다. Test 실행 명령은 본 관리자가 호출하지 않는다. SST/MHW·해안/계절별·forcing 독립성 평가 계획을 별도 확정한다.
