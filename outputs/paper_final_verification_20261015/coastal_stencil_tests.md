# 해안 stencil CPU 수치 시험

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**88/88 PASS; 운영 코드 변경 없음.** [outputs/paper_final_verification_20261015/audit_cpu.py](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/audit_cpu.py:1), [outputs/paper_final_verification_20261015/stencil_tests.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/stencil_tests.json:1).

실제 구현 [src/mhw/ocean_stencil.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/ocean_stencil.py:8), [src/mhw/ocean_stencil.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/ocean_stencil.py:15), [src/mhw/ocean_stencil.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/ocean_stencil.py:21):

| 이웃 | 미분 규칙 | 물리 손실 support |
|---|---|---|
| 양쪽 바다 | (T_hi−T_lo)/(2Δ) | 각 축에 최소1개 있으면 포함 |
| 한쪽 바다 | (T_hi−T)/Δ 또는(T−T_lo)/Δ | 반대 축도 최소1개 있으면 포함 |
| 해당 축 이웃 없음 | 해당 축 미분0 | 해당 셀 제외 |
| 고립 바다 | 두 축0 | 제외 |
| 바깥 경계 | 영역 밖을 이웃으로 보지 않음; 내부 쪽 편측 | 동일 support 규칙 |
| 육지 | arithmetic 전에 T를0으로 대체; 출력0 | 제외 |

실제 해양661개, physics support660개. 육지0/−100/1000/NaN/Inf를 바꿔도 해양 미분 차이0. 유효한 바다 자체의 NaN을 숨기는 규칙은 아니다. Native/current P0, scale, NoAdv/Adv는 같은 함수를 공유한다. NoAdv도 Adv와 support를 동일하게 유지하므로 'gradient가 필요 없어서 해안 support를 늘린 모델'이 아니다.

FP32 허용 오차는 사전 식 `86400 × eps(dtype) × 32 × max|T| / 23000`으로 정했다. 온도 차의 취소오차·분모·곱셈을 포함한 보수적 연산 bound이며, 실제 오차보다 느슨한 별도 성능 기준을 탐색하지 않았다. FP64도 같은 식. 지리 spacing이float32로 반환되는 테스트에는 rtol1e−7/atol1e−12 K/m. 상수장·부호반전·육지불변·과거 연산자 비교에는 exact equality.

| test | expected_min | expected_max | calculated_min | calculated_max | max_abs_error | max_rel_error | atol | rtol | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch.float32_constant | 0 | 0 | -0 | -0 | 0 | 0 | 0 | 0 | PASS |
| torch.float32_linear | -0.432 | -0.432 | -0.432002 | -0.4319991 | 2.037764e-06 | 4.717045e-06 | 0.0003393342 | 0 | PASS |
| torch.float64_constant | 0 | 0 | -0 | -0 | 0 | 0 | 0 | 0 | PASS |
| torch.float64_linear | -0.432 | -0.432 | -0.432 | -0.432 | 4.385381e-15 | 1.015134e-14 | 6.320593e-13 | 0 | PASS |
| geographic_dx_descFalse | 4.153789e-06 | 4.504291e-06 | 4.153789e-06 | 4.504291e-06 | 1.617457e-13 | 3.760215e-08 | 1e-12 | 1e-07 | PASS |
| geographic_dy_descFalse | 1.798643e-06 | 1.798643e-06 | 1.798643e-06 | 1.798643e-06 | 4.922146e-14 | 2.736588e-08 | 1e-12 | 1e-07 | PASS |
| geographic_dx_descTrue | 4.153789e-06 | 4.504291e-06 | 4.153789e-06 | 4.504291e-06 | 1.617457e-13 | 3.760215e-08 | 1e-12 | 1e-07 | PASS |
| geographic_dy_descTrue | 1.798643e-06 | 1.798643e-06 | 1.798643e-06 | 1.798643e-06 | 4.922146e-14 | 2.736588e-08 | 1e-12 | 1e-07 | PASS |
| nonuniform_one_sided_documented_rule | 1.388106e-05 | 1.388106e-05 | 1.388106e-05 | 1.388106e-05 | 1.245819e-13 | 8.97495e-09 | 1e-12 | 1e-07 | PASS |
| archived_operator_equivalence | -0.0001557331 | 0.000175268 | -0.0001557331 | 0.000175268 | 0 | 0 | 0 | 0 | PASS |
| actual_mask_constant | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS |

전체16개 해양 이웃 조합×float32/64의 계산 기대값과 support는 CSV에 각각 기록했다. 상대오차는 기대값0 부근에서 해석하지 않고 절대오차를 사용한다.

과거 보정 구현 [physics_stencil.py](/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py:46)과 실제 mask의 유한장 결과는 bitwise 동일. current는 NaN/Inf 육지 arithmetic도 사전에 차단한다. 과거30회 pilot는 runtime patch로 loss와 scale 모두 보정됐다는 별도 hash/checkpoint 감사가 있다: [outputs/historical_coastal_audit_20261008/report.md](/ceph_data/yeonwoo/MHW_subset/outputs/historical_coastal_audit_20261008/report.md:1).

반면 v2 원본 native 미분은 mask가 없었다: [outputs/v3_preflight_20261008/originals/src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/outputs/v3_preflight_20261008/originals/src/train_hybrid_jeju_forecast.py:588). 이 오류를 이미 보정된 Oracle120에 다시 귀속하지 않는다. 비균일 격자의 F 규칙 재현 PASS는 비균일 격자 정확도 보장이 아니며 horizontal_advection_audit.md에 구체적인33.333% 편차를 남겼다.
