# 최종 분석 요약

Oracle120 기존 결과만 사용했다. 120개 조건의 입력 파일 존재 및 날짜·관측 SST·Mask·MHW 라벨 대응을 확인했다. 모델 학습/추론/감사 전체 재실행 없이, 기존 지표와 90일 bootstrap을 재사용하고 저장 예측으로 공간·계절 요약을 생성했다.

## 1. 고정 물리 제약이 Baseline보다 일관되게 우수한가?

- P0: seed별 RMSE 개선 8/24, 조건별 평균 RMSE 개선 1/8. seed 개선율 범위 -6.074~5.701%.
- NoAdv: seed별 RMSE 개선 12/24, 조건별 평균 RMSE 개선 6/8. seed 개선율 범위 -5.890~13.457%.
- Adv: seed별 RMSE 개선 5/24, 조건별 평균 RMSE 개선 1/8. seed 개선율 범위 -8.089~5.817%.

보편적 우위로 기술하지 않는다. 모델별·조건별 효과와 실패를 함께 제시한다.

## 2. Auxiliary-only의 독립 효과

25%·Lead14에서 seed별 개선율 평균 2.466%, 3/3 seed 개선이 관찰됐다. 반면 10%·Lead7은 0/3 개선이며 평균 RMSE는 악화됐다. 그 외 다수 조건의 변화는 작다. 고정 진단 batch의 gradient 비 중앙값은 epoch1 1.7588% → epoch20 0.1985% → epoch80 0.04071%로 감소했고, Rres RMS는 0.17484 → 0.05187 → 0.01888 K/day다. 감소 자체와 원인 해석을 구분한다.

Test RMSE 개선 15/24. Validation 개선 16/24.

| ratio | lead | delta_rmse_mean | paired_improvement_pct_mean | candidate_rmse_wins | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 7 | 0.0128295 | -1.34503 | 0 | -0.00200707 | 0.0285666 |
| 10 | 14 | -0.00448843 | 0.240796 | 1 | -0.0264405 | 0.0176088 |
| 25 | 7 | -0.000803319 | 0.0940856 | 3 | -0.00264819 | 0.000963563 |
| 25 | 14 | -0.0282698 | 2.46551 | 3 | -0.0453411 | -0.0117517 |
| 50 | 7 | -0.000198432 | 0.0224975 | 2 | -0.000513617 | 0.0001047 |
| 50 | 14 | -0.00136422 | 0.134456 | 2 | -0.0142357 | 0.0121705 |
| 100 | 7 | -0.000214278 | 0.0267882 | 3 | -0.000361995 | -7.59948e-05 |
| 100 | 14 | -0.000200908 | 0.0199201 | 1 | -0.000508032 | 7.39296e-05 |

Epoch 1/20/80의 원본 진단 batch 중앙값은 tables/auxiliary_epoch1_20_80.csv에 수록했다. 감소 관찰과 원인 해석을 구분하며, Rres가 완전히 0이라고 하지 않는다.

## 3. NoAdv와 Adv 차이가 반복되는 조건

수평 이류항 포함 여부의 효과를 분리한 비교는 동일 구현의 NoAdv–Adv뿐이다. P0와 NoAdv·Adv의 비교는 활성 head 및 물리항 정의가 함께 달라지는 복합 비교다. P0에서만 we head가 활성화되며, 세 모델 모두 Rres head는 비활성이다. 단파 투과, 학습 연행과 MLD-deepening proxy, 산술 평균과 두께 가중 혼합층 유속 정의도 다르다. 동일한 네트워크 구조가 동일한 활성 목적함수를 뜻하지 않는다. 따라서 P0 비교를 수평 이류항만의 독립 효과로 해석하지 않으며, NoAdv–Adv 결과도 구현된 학습 제약의 예측 효용에 한정한다.

8개 Ratio×Lead 조건 모두 평균 RMSE는 NoAdv가 낮다. 차이는 0.01252–0.06156 K다. 기존 90일 조건부 신뢰구간은 6/8 조건에서 0을 제외하지만, 50%·Lead14와 100%·Lead14는 0을 포함한다. 이는 모든 조건에서 확증된 효과라는 뜻이 아니다. 본문 표 `tables/table5_paired.tex`는 NoAdv–Adv 8개 조건을 6개 열로 제시한다. 전체 48개 대응 비교는 Supplementary Table S1 (`tables/tableS_paired_all.tex`)에 있으며, CSI 신뢰구간을 포함한 확장표는 `tables/tableS_noadv_adv.csv`다.

NoAdv 우위: RMSE 22/24, CSI 19/24 seed 조건. 아래 ΔRMSE는 Adv−NoAdv로 양수가 NoAdv 우위다.

| ratio | lead | delta_rmse_mean | delta_rmse_sd | Adv_RMSE_wins_out_of_3 | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 7 | 0.0615648 | 0.0495219 | 0 | 0.0461477 | 0.0793242 |
| 25 | 7 | 0.0370797 | 0.0269344 | 0 | 0.025476 | 0.0500821 |
| 50 | 7 | 0.0416721 | 0.0179955 | 0 | 0.0260133 | 0.0586334 |
| 100 | 7 | 0.0229738 | 0.0127409 | 0 | 0.0112596 | 0.0337652 |
| 10 | 14 | 0.0312799 | 0.041067 | 1 | 0.00467274 | 0.0590669 |
| 25 | 14 | 0.0504384 | 0.0467617 | 0 | 0.0262628 | 0.0773887 |
| 50 | 14 | 0.0125216 | 0.0266915 | 1 | -0.011005 | 0.0371425 |
| 100 | 14 | 0.0164069 | 0.00193316 | 0 | -0.00194777 | 0.0350426 |

## 4. 자료 감소가 제약의 유용성을 증가시키는가?

보편적·단조적 증가로 결론낼 수 없다. NoAdv는 Lead7의 10/25/50%에서 평균 seed 개선율이 양수지만 100%는 음수이며, Lead14는 50%에서 악화되는 등 비단조적이다. 작은 자료량에서 유리한 일부 조건을 물리 제약 전체의 자료 효율 향상으로 일반화하지 않는다.

비율별 대응 개선율은 아래와 같다. 값은 동일 조건 내 seed 개선율의 평균이며 자료 부족에 따른 단조 개선을 가정하지 않는다. 100%는 3,995개 window다.

| candidate | lead | 10 | 25 | 50 | 100 |
| --- | --- | --- | --- | --- | --- |
| Adv | 7 | -2.86865 | -3.65982 | -4.46973 | -6.97094 |
| Adv | 14 | -1.29058 | 0.352132 | -5.10848 | -1.36649 |
| Auxiliary | 7 | -1.34503 | 0.0940856 | 0.0224975 | 0.0267882 |
| Auxiliary | 14 | 0.240796 | 2.46551 | 0.134456 | 0.0199201 |
| NoAdv | 7 | 3.158 | 0.652493 | 0.499938 | -4.07768 |
| NoAdv | 14 | 1.06808 | 4.76349 | -3.89732 | 0.283759 |
| P0 | 7 | -3.36361 | -1.64306 | -0.67871 | -5.44227 |
| P0 | 14 | 2.47796 | -0.912313 | -4.25204 | -0.871955 |

## 5. SST 차이와 MHW 차이의 연결

NoAdv는 Adv보다 RMSE가 낮은 22/24 조건 중 18개에서 CSI도 높다. 두 지표가 함께 좋아지는 경향은 있지만 항상 일치하지 않으며, POD/FAR trade-off와 함께 해석한다. 아래 일반 비교표의 candidate가 Adv일 때는 Adv 우위 횟수이므로 NoAdv 우위 횟수와 혼동하지 않는다.

| reference | candidate | n | sst_and_csi_better | sst_better_csi_not_better | descriptive_delta_correlation |
| --- | --- | --- | --- | --- | --- |
| Auxiliary | NoAdv | 24 | 7 | 4 | -0.0141465 |
| Baseline | Adv | 24 | 4 | 1 | -0.4067 |
| Baseline | Auxiliary | 24 | 14 | 1 | -0.691297 |
| Baseline | NoAdv | 24 | 10 | 2 | -0.371559 |
| Baseline | P0 | 24 | 4 | 4 | -0.642123 |
| NoAdv | Adv | 24 | 1 | 1 | -0.607755 |

RMSE 개선이 CSI 개선을 보장하지 않는다. POD/FAR와 TP/FP/FN을 함께 판단한다.

## 6. 공간·계절 차이

NoAdv−Adv 차이의 반대부호인 Adv−NoAdv RMSE를 보면, 겨울의 조건평균 차이는 Lead7 약 0.0779 K, Lead14 약 0.0735 K이며 여름은 각각 0.0173 K, 0.0018 K다. 일부 여름 조건은 Adv가 더 낮은 RMSE를 보인다. 이 요약은 ratio×seed의 기술적 평균이며 계절 효과의 독립 통계검정이 아니다. 새 SW-Jeju 영역은 40개 해양 격자, coast 141개, offshore 520개다.

New exploratory SW-Jeju box: 32.0–33.25°N, 125.0–126.5°E, inclusive, intersected with the saved ocean mask. Chosen geographically south/west of Jeju (approx.33.4°N,126.5°E), before computing error summaries for this new box; not independently validated as a coastal dynamical region. Historical SW-Korea box:33.5–35°N,125.5–127.5°E. Coast: ocean cells with a cardinal land neighbor; domain exterior is not land. Case rule: for each lead choose the earliest June–August day in the final complete Test year (2022); fixed ratio100/seed42 for all models. No skill-based selection.

| lead | season | mean | min | max |
| --- | --- | --- | --- | --- |
| 7 | DJF | 0.077883 | 0.0164874 | 0.248155 |
| 7 | JJA | 0.0172757 | -0.0521205 | 0.162601 |
| 7 | MAM | 0.0479539 | -0.0241525 | 0.0941178 |
| 7 | SON | 0.0312003 | 0.00819291 | 0.0610229 |
| 14 | DJF | 0.0734948 | -0.0374496 | 0.163189 |
| 14 | JJA | 0.00178853 | -0.131837 | 0.224089 |
| 14 | MAM | 0.0493184 | -0.0645349 | 0.149864 |
| 14 | SON | 0.00404619 | -0.0712483 | 0.0592142 |

위 mean/min/max는 ratio×seed 조건별 대응차의 기술적 요약이다. Figure5와 지역·계절 원본 CSV에서 조건별 차이를 확인한다. 관측장 기반 이류항 크기와 오차차의 공간 상관은 인과 근거가 아니다.

## 7. 논문에서 허용되는 결론

고정 열수지 제약의 예측 효용은 항 구성, 자료량 및 lead에 따라 달라진다. 현재 실험에서 수평 이류를 제외한 제약의 우위가 반복되지만, 이는 이산화된 학습 제약의 효용에 관한 결과이며 실제 해양 이류의 중요성을 부정하지 않는다. Auxiliary-only 효과는 조건부이며 큰 보편적 개선으로 서술하지 않는다. 단일 Test 기간과 3개 seed에 조건부인 탐색 결과라는 한계를 명시한다.

## 산출물

Main Figure 1–5, Supplementary S1–S6; Table1–6 및 별도 Validation/seed 표; 모두 영어 Figure·캡션, 350dpi PNG·PDF·SVG 및 plot CSV. 수치 검증 보고서는 기존 증거를 인용하며 새 PASS로 포장하지 않는다. WSDM 확정 기존 표는 [기존 별도 표 디렉터리](/ceph_data/yeonwoo/MHW_subset/paper_analysis_20260929/results/paper_tables)에 그대로 두었다.
