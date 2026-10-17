# Auxiliary-only 기여 및 고정 Test 평가

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**PASS WITH LIMITATIONS. 학습 오류는 확인되지 않았으며 효과는 조건별로 작거나 불균일하다.**

Data+0.0001×Rres penalty, physics OFF, we-tie OFF, 동일 모델 구조/초기화/v3 RHS/80 actual epochs 유지. 비활성 physics 함수 호출은 24개×80epoch 모두0이고, Rres 호출은 각 epoch의 모든 training batch와 일치했다. [outputs/paper_final_verification_20261015/protocol120.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/protocol120.csv:1); [class Monitor(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_auxiliary24/aux_worker.py:32), [def compute_batch_losses(](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:264).

Rres=1e−4 tanh(head), U=mean[(86400Rres)²]/S. 마지막 affine weight/bias를0으로 두면 Rres=0, U=0, 공유 Predictor로 가는 U gradient도0이 될 수 있다. 이는0-target 정규화의 예상 최소해이지 학습 실패가 아니다. 실제 마지막층 norm이0이 되었다는 주장과 구분한다. 기존 사전검사 FAIL을 수학적 성질로 재분류한 기록을 유지했다: [outputs/v8_actual_epoch_auxiliary24_20261015/preflight_reclassification.json](/ceph_data/yeonwoo/MHW_subset/outputs/v8_actual_epoch_auxiliary24_20261015/preflight_reclassification.json:1).

247개 별도 train-batch autograd 진단의 model state·기존 grad·CPU/CUDA/Python/NumPy/loader RNG·mode 보존 flag가 모두 참이다. optimizer update는0이며 fixed first16 selected Train samples를 사용했다. Epoch1은 초기화 직후가 아니라 1epoch 학습 후이다. raw auxiliary gradient와 λ를 곱한 gradient를 구분한다. [def diagnostic(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_auxiliary24/aux_worker.py:64), [outputs/paper_final_verification_20261015/aux_diagnostics_rechecked.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_diagnostics_rechecked.csv:1).

| epoch | rres_rms_kday | auxiliary_loss | weighted_auxiliary_loss | data_loss | weighted_aux_data_loss_ratio | shared_aux_gradient_norm | shared_data_gradient_norm | weighted_gradient_percent | rres_last_layer_parameter_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1748439 | 0.3439141 | 3.439141e-05 | 0.02791489 | 0.001240116 | 189.1758 | 1.153789 | 1.758776 | 0.5897906 |
| 5 | 0.1090555 | 0.1404222 | 1.404222e-05 | 0.02228996 | 0.0004513782 | 95.24875 | 1.687945 | 0.5135662 | 0.5687251 |
| 10 | 0.07701458 | 0.07707977 | 7.707977e-06 | 0.02166316 | 0.0003787051 | 52.99599 | 0.853481 | 0.5338465 | 0.5513025 |
| 20 | 0.05186549 | 0.0347911 | 3.47911e-06 | 0.01979085 | 0.0001197378 | 20.92383 | 0.7099745 | 0.1985071 | 0.5181911 |
| 80 | 0.01887642 | 0.00422016 | 4.22016e-07 | 0.01243431 | 2.986169e-05 | 0.8338922 | 0.1767041 | 0.04071002 | 0.461234 |

기존 1.759%→0.199%→0.041%는 각각1.758776%,0.198507%,0.040710%로 재현됐다. Rres RMS 및 auxiliary gradient가24개 모두 감소한다. 최종 gradient가 정확히0이라는 증거는 없으며, 고정 진단 batch에서의 감소이지 모든 배치/시점에서 사라진다는 증명은 아니다.

**Validation 검산: 16/24개 seed 조건에서 개선.**

| ratio | lead | Baseline | Auxiliary | delta_K | wins |
| --- | --- | --- | --- | --- | --- |
| 10 | 7 | 1.008026 | 1.00838 | 0.0003540471 | 1 |
| 10 | 14 | 1.233606 | 1.224155 | -0.009450701 | 3 |
| 25 | 7 | 0.8825661 | 0.8840851 | 0.001518999 | 1 |
| 25 | 14 | 1.086967 | 1.082426 | -0.004540749 | 3 |
| 50 | 7 | 0.8496662 | 0.849602 | -6.422654e-05 | 2 |
| 50 | 14 | 1.021854 | 1.02184 | -1.412742e-05 | 2 |
| 100 | 7 | 0.8180066 | 0.8179817 | -2.492831e-05 | 2 |
| 100 | 14 | 0.9697523 | 0.9696406 | -0.0001117015 | 2 |

**이번 고정 Test 추론: 24/24 완료.** 기존 Validation-best의SHA/epoch/ValRMSE를 실행 전 protocol.json에 고정했으며, 재학습·reselection·optimizer step 없이 GPU5/6/7 각1개로만 추론했다. 원래 Oracle96과 같은 score()·고정 MHW threshold를 사용하고 기존96결과를 덮어쓰지 않았다. 모두 forcing 제거/원본/NaN에서 prediction bitwise 동일, state unchanged, 대응 Baseline의 target/mask/date 동일. 근거 [outputs/paper_final_verification_20261015/auxiliary_test/protocol.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/auxiliary_test/protocol.json:1), [outputs/paper_final_verification_20261015/aux_test.py](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/aux_test.py:1), 각 `auxiliary_test/runs/*/forcing_invariance_test.json`.

Test 개선 **15/24**, Validation16/24와 혼동하지 않는다. Δ=Aux−Baseline, 음수가 RMSE 개선; 개선율은 seed별 비율의 평균. 아래 CI는 고정3seed에 조건부인90일블록 구간이다.

| ratio | lead | Baseline_RMSE | Auxiliary_RMSE | delta_K | improvement_pct | wins | CI95 | Baseline_CSI | Auxiliary_CSI | delta_POD | delta_FAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 7 | 1.009889 | 1.022719 | 0.01282954 | -1.345035 | 0 | [-0.002007, 0.028567] | 0.282848 | 0.308017 | 0.04285851 | 0.01907439 |
| 25 | 7 | 0.8613391 | 0.8605358 | -0.0008033191 | 0.09408558 | 3 | [-0.002648, 0.000964] | 0.2868958 | 0.2908518 | 0.008575783 | 0.00733597 |
| 50 | 7 | 0.8431505 | 0.8429521 | -0.0001984318 | 0.02249749 | 2 | [-0.000514, 0.000105] | 0.2436255 | 0.2441659 | 0.0006952913 | 0.0002750103 |
| 100 | 7 | 0.7960509 | 0.7958366 | -0.000214278 | 0.02678824 | 3 | [-0.000362, -0.000076] | 0.3016763 | 0.3023791 | 0.001086098 | 0.0004351679 |
| 10 | 14 | 1.276356 | 1.271867 | -0.004488426 | 0.2407964 | 1 | [-0.026440, 0.017609] | 0.1833107 | 0.1886165 | 0.01064596 | -0.01619009 |
| 25 | 14 | 1.127465 | 1.099195 | -0.02826982 | 2.465512 | 3 | [-0.045341, -0.011752] | 0.1716559 | 0.213569 | 0.06963429 | -0.02290564 |
| 50 | 14 | 1.018989 | 1.017625 | -0.001364217 | 0.1344563 | 2 | [-0.014236, 0.012170] | 0.214292 | 0.2024429 | -0.0313556 | -0.01615866 |
| 100 | 14 | 0.9936103 | 0.9934094 | -0.0002009077 | 0.0199201 | 1 | [-0.000508, 0.000074] | 0.1559541 | 0.1557033 | -0.0003704035 | 0.0005207416 |

25%·Lead14의 평균 개선약2.47%,3/3seed는 실질적인 차이가 관찰된 조건이다. 반면10%·Lead7은0/3개선이며,50/100%의 절대변화는 대체로 작다.100%·Lead7의 매우 작은 감소가 조건부 시간CI에서0을 제외하더라도 큰 실용적 효과나 seed 모집단 재현성을 의미하지 않는다. 별도 모델/손실 변경으로 좋은 수치를 선택하지 않았다. 기존 Test를 반복 열람한 연구이므로 이번 평가는 누락된 ablation Test의 탐색적 보완이지 독립 확증이 아니다.
