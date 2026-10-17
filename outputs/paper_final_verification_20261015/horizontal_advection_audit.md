# 수평 이류 수식·단위·좌표 검증

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**판정: 현재 균일 0.25° 격자의 실행 계산 PASS. 원격자→병합 최초 처리 및 불균일 격자 일반화는 한계.**

1. budget2d `u_ml`, `v_ml`: m s⁻¹, 각각 ML-averaged eastward/northward velocity. Adv의 3D `uo`, `vo`도 각각 eastward/northward m s⁻¹. 원래 파일의 실제 attrs를 저장했다: [outputs/paper_final_verification_20261015/dataset_metadata.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/dataset_metadata.json:1), [outputs/paper_final_verification_20261015/physics_numerical_checks.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/physics_numerical_checks.json:1).
2. tensor의 마지막 축=x=longitude, 뒤에서 두 번째=y=latitude. 실제 latitude30→37°, longitude123→131°, 간격 모두+0.25°. `stored_direction=decreasing` 속성은 실제 배열과 모순된다. 계산은 이 속성이 아니라 좌표값의 부호를 사용한다. 이 불일치를 실제 v 부호 오류로 바꾸어 해석하지 않는다.
3. dx=R cos(lat) gradient(lon_radians,axis=1), dy=R gradient(lat_radians,axis=0), R=6,371,000m. 부호 있는 간격을 유지하므로 좌표 감소 순서에서도 테스트가 통과한다. [def compute_grid_spacing_m(](/ceph_data/yeonwoo/MHW_subset/src/mhw/datasets.py:101). degree를 그대로 미분 분모로 사용하는 활성 경로 없음.
4. D_xT,D_yT는 K/m. A_h=−u D_xT−v D_yT는 K/s. P0는 최종 residual/RHS에86400을 곱하고, Adv는 항 내부에서−86400(u D_xT+v D_yT)를 만든다. 서로 중복 변환하지 않는다. [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:690), [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:725), [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:28).
5. 최초 재격자화/일평균 코드·가중치·원격자 u/v 파일이 없어 그 단계의 벡터 정합성까지 PASS로 확대하지 않는다. 현재 3D/budget2d time/lat/lon 배열은 정확히 일치하고, feature builder가 순서를 검사한 뒤 uo/vo를 동일 depth 가중으로 투영한다. [scripts/prepare_mixed_budget_features.py](/ceph_data/yeonwoo/MHW_subset/scripts/prepare_mixed_budget_features.py:16), [scripts/prepare_mixed_budget_features.py](/ceph_data/yeonwoo/MHW_subset/scripts/prepare_mixed_budget_features.py:22), [outputs/glorys_source_inventory_20261011/report_ko.md](/ceph_data/yeonwoo/MHW_subset/outputs/glorys_source_inventory_20261011/report_ko.md:1).

수치 검증: FP32 선형장 T=20+2e−5 x−1e−5 y, u=.2,v=−.1에서 기대−0.432K/day, 최대 오차2.038e−6. FP64 최대4.385e−15. 상수장, u/v 반전, 실제 해안 mask, NaN/Inf 육지, 위도 보정 및 좌표 역순, 16개 이웃 조합 포함88/88 통과. [outputs/paper_final_verification_20261015/stencil_tests.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/stencil_tests.csv:1).

비균일 좌표의 편측차분은 중심 좌표의 평균 간격을 사용한다. 0.25°/0.5° 양쪽 간격에서 한 이웃을 제거하면 진짜 선형장 미분보다33.333% 큰 사례가 재현된다. 이는 **일반 불균일 격자 차분으로서의 제한**이며, 모든 실제 좌표 간격이0.25°인120개 모델의 오류로 판정하지 않는다. F 테스트의 PASS는 명시된 구현 규칙 재현이고, 임의 격자에서의 수학적 정확성 PASS가 아니다.

논문 문장: “Horizontal temperature gradients are computed in metres using ocean-neighbour finite differences on the regular latitude–longitude grid, with the longitude spacing corrected by cos(latitude); the resulting advective tendency is expressed in K day⁻¹.”
