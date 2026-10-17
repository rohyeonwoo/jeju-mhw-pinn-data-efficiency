# Actual Epoch Oracle96 최종 Validation 결과 검토

2026-10-12 14:23 KST

2026-10-12 13:48 KST에 96/96 완료. 새 GPU 실행·학습·Test 평가 없이 기존 개별 로그 7,680행을 재검증했다.

## 결론

가장 일관된 개선은 10%·Lead7 NoAdv: paired 개선율 평균 +3.045%, 3/3 seeds 개선이다. P0의 전반적인 우위는 확인되지 않았고, Adv는 모든 ratio×lead의 평균에서 Baseline보다 불리했다. 수평 이류 추가가 예측을 개선한다는 근거는 현재 Validation에서 약하다. 이는 물리식 자체의 진실성 또는 실제 해양에서 이류의 중요성을 부정하는 결과가 아니다.

## 학습 및 무결성

모든96개80actual epochs, 각epoch정확한sample1회/lastbatch포함, 자료량별1920/4960/9920/20000 optimizer calls, LR/physics warmup5 PASS. 모든 run 전체/post-warmup checkpoint 저장·재로드 기록192개 PASS. 같은ratio/lead/seed 입력통계·초기화·shuffle순서 일치. full Validation forcing 독립성96개 PASS. GPU7/8/9, 각최대2개, 남은worker lease0. Checkpoint 바이너리 재추론은 이번 검토에서 수행하지 않았다.

## Validation RMSE: 3 seeds 평균 ± 표본 표준편차

| Lead | Ratio | Baseline | P0 | NoAdv | Adv |
|---|---|---|---|---|---|
| 7 | 10% | 1.0080 ± 0.0268 | 1.0125 ± 0.0286 | 0.9770 ± 0.0253 | 1.0247 ± 0.0206 |
| 7 | 25% | 0.8826 ± 0.0083 | 0.8863 ± 0.0137 | 0.9023 ± 0.0043 | 0.9223 ± 0.0145 |
| 7 | 50% | 0.8497 ± 0.0047 | 0.8535 ± 0.0135 | 0.8752 ± 0.0093 | 0.9012 ± 0.0087 |
| 7 | 100% | 0.8180 ± 0.0008 | 0.8316 ± 0.0079 | 0.8644 ± 0.0038 | 0.8805 ± 0.0062 |
| 14 | 10% | 1.2336 ± 0.0245 | 1.2312 ± 0.0118 | 1.2229 ± 0.0382 | 1.2417 ± 0.0145 |
| 14 | 25% | 1.0870 ± 0.0141 | 1.0906 ± 0.0312 | 1.0845 ± 0.0261 | 1.1080 ± 0.0208 |
| 14 | 50% | 1.0219 ± 0.0164 | 1.0209 ± 0.0242 | 1.0507 ± 0.0105 | 1.0590 ± 0.0105 |
| 14 | 100% | 0.9698 ± 0.0010 | 0.9825 ± 0.0046 | 1.0018 ± 0.0018 | 1.0167 ± 0.0051 |

## Baseline 대비 개선율: 양수가 개선

각 seed에서 100×(Baseline−Model)/Baseline을 계산한 뒤 평균했다. 괄호는 개선 seed 수이다. 평균 RMSE끼리 나눈 개선율과는 구분한다.

| Lead | Ratio | P0 | NoAdv | Adv |
|---|---|---|---|---|
| 7 | 10% | -0.474% (1/3) | +3.045% (3/3) | -1.700% (1/3) |
| 7 | 25% | -0.434% (1/3) | -2.249% (0/3) | -4.519% (0/3) |
| 7 | 50% | -0.450% (1/3) | -3.011% (0/3) | -6.065% (0/3) |
| 7 | 100% | -1.662% (0/3) | -5.672% (0/3) | -7.638% (0/3) |
| 14 | 10% | +0.181% (2/3) | +0.866% (1/3) | -0.675% (1/3) |
| 14 | 25% | -0.328% (1/3) | +0.219% (2/3) | -1.932% (0/3) |
| 14 | 50% | +0.098% (1/3) | -2.830% (0/3) | -3.646% (0/3) |
| 14 | 100% | -1.313% (0/3) | -3.306% (0/3) | -4.841% (0/3) |

## 핵심 10%·Lead7 seed별 결과

| Seed | Baseline | P0 | NoAdv | Adv | NoAdv 개선율 |
|---|---|---|---|---|---|
| 42 | 1.031431 | 1.001946 | 0.963589 | 1.009734 | +6.577% |
| 7 | 1.013790 | 1.044942 | 1.006156 | 1.048155 | +0.753% |
| 123 | 0.978857 | 0.990676 | 0.961203 | 1.016185 | +1.804% |

NoAdv는 세 seed에서 개선됐으나 효과 크기는 +0.753%~+6.577%로 다르다. Seed3개뿐이고 Validation은 모델 선택에 사용했으므로 확증적 통계 우위나 Test 일반화를 주장하지 않는다. 10%·Lead14의 NoAdv 평균 개선 +0.866%는 1/3 seeds만 개선한 결과여서 일관되지 않다. 25%·Lead14는 +0.219%, 2/3 seeds로 작은 효과다. 50/100%에서는 NoAdv가 양Lead 모두 세seed 전부 악화했다.

## Best epoch와 과적합

| Model | Best epoch 중앙값 | 범위 | Epoch80 악화율 평균(best 대비) | Train RMSE가 best시점보다 낮은 run |
|---|---|---|---|---|
| Baseline | 19.5 | 7–32 | 5.49% | 24/24 |
| P0 | 21.5 | 8–39 | 6.23% | 24/24 |
| NoAdv | 25.0 | 15–46 | 3.70% | 24/24 |
| Adv | 26.0 | 11–54 | 3.67% | 24/24 |

Best 전체범위7–54, 중앙값22, epoch5 이전0/96. 모든run에서 전체best와 post-warmup best의 Val RMSE가 같았다. 따라서 warmup 완료 전에만 좋은 checkpoint가 선택되는 문제는 이번에 관찰되지 않았다. 다만 후반 Training 오차 감소와 Validation 악화는 여전히 있으므로 과적합이 완전히 해결됐다고 표현하지 않는다. Training RMSE는 online 집계이다.

## 기존 Virtual Epoch 대비: 동일 10%·Lead7·seed42만

| Model | Old Val | Actual Val | 자기 기존결과 대비 개선 | Old best virtual epoch | Actual best epoch |
|---|---|---|---|---|---|
| Baseline | 1.065011 | 1.031431 | 3.15% | 4 | 14 |
| P0 | 1.048509 | 1.001946 | 4.44% | 1 | 20 |
| NoAdv | 0.977975 | 0.963589 | 1.47% | 2 | 43 |
| Adv | 1.040387 | 1.009734 | 2.95% | 3 | 44 |

10% optimizer calls는38240→1920(94.98% 감소). old NoAdv의 Baseline 대비8.17% 개선은 seed42 단일조건이었다. 같은seed의 신규개선은6.58%; 신규3-seed 평균3.05%와 단순일대일 비교하면 안 된다. 네 모델 모두 자기 기존seed42 Val보다 개선했으나 epoch정의·학습량·Validation빈도·LR시간축·physics warmup0→5가 함께 바뀌어 변화 전체를 과적합 완화 하나로 귀속할 수 없다.

## 연구 해석과 다음 단계

자료량 증가에 따라 Baseline RMSE는 두Lead 모두 낮아졌다. NoAdv의 일관된 장점은 현재10%·Lead7에 한정되며 저자료량 전반 또는 긴lead의 우위로 일반화하지 않는다. P0의 평균 개선은Lead14의10/50%에서0.181/0.098%에 그치고 seed재현성이 제한적이다. Adv는NoAdv와의paired24조건 중2개에서만 더 낮은RMSE(22개는높음). NoAdv/Adv는이류 유무만 다른 통제비교이므로 현재구성에서이류제약추가의예측상 이점이 없다는 근거다. 물리적 열수지 적합성은 별개 평가로 유지한다.

현재 설정과 전체Validation-best checkpoint를 고정해 보존한다. 이번에는 Test를 실행하지 않았다. 이후 별도 승인된 Test에서 SST/MHW 및 해안/계절 오차를 평가하되, Test를 보고 λ·Kd·checkpoint 종류를 다시 선택하지 않는다. 기존Test를 확인한 뒤 변경한 프로토콜이므로 같은Test의 결과는탐색적분석이다. Oracle 추가감독을사용하므로SST-label efficiency이며total-information efficiency는입증하지않는다.

## 자료

![Validation overview](validation_overview.png)

[32개 조건 요약 CSV](comparison_summary.csv) · [96개 실행 감사 CSV](per_run_review.csv) · [검증 JSON](verification.json)

원본 결과: ../report.md, ../validation_results.csv, ../seed_improvements.csv, ../all96_gate.json, 각runs/<name>/history.json·status.json·checkpoint_audit.json·forcing_invariance_val.json. 집계코드 scripts/oracle96_actual/summarize.py; 새 독립검토 코드 analyze.py. 기존 원본 결과는 덮어쓰지 않았다.
