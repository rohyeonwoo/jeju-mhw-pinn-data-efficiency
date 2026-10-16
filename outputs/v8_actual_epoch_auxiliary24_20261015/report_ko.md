# Auxiliary24 Validation 보고 (24/24 완료)
작성 2026-10-15 06:10 KST

Zero-head 최소해는 의도된 출력 축소 정규화의 수학적 성질로 재분류했다. 이는 구현 오류나 실행 실패가 아니다. 기존 λaux=0.0001, Physics/tie OFF, v3 RHS scale, native architecture 유지. Test 실행 없음. 원 Oracle96 96개는 저장된 Validation 결과만 읽었다.
학습 설정: 80 actual epochs, early stopping OFF, 매 epoch Validation-best 선택. 대응 Baseline initial SHA/실제 tensor, 입력/통계, 매 epoch sample order/LR를 검사한다. 진단은 고정 첫16 Train samples의 train-mode gradient (cuDNN GRU backward)로, 학습 batch 평균과 구분한다. RNG/model/.grad/mode 보존 및 smoke 진단ON/OFF 1step 동일성 검증.
개선율은 각 seed (Baseline−Auxiliary)/Baseline×100 후 평균. 양수가 개선이다. 작은 차이를 통계적 유의성으로 해석하지 않으며 3seed의 방향 일관성을 함께 본다. 기존 Test 열람 이력이 있으나 이번 작업에서는 Test를 추가 사용하지 않는다.

|Ratio|Lead|완료 seed|Baseline RMSE 평균|Aux RMSE 평균±SD|대응 개선율 평균 (%)|개선 seed|Best epoch 중앙값|
|---|---|---|---|---|---|---|---|
|10|7|3/3|1.00803|1.00838 ± 0.02892|-0.035|1/3|26.0|
|10|14|3/3|1.23361|1.22415 ± 0.01841|+0.757|3/3|14.0|
|25|7|3/3|0.88257|0.88409 ± 0.00758|-0.173|1/3|20.0|
|25|14|3/3|1.08697|1.08243 ± 0.01277|+0.417|3/3|14.0|
|50|7|3/3|0.84967|0.84960 ± 0.00504|+0.008|2/3|23.0|
|50|14|3/3|1.02185|1.02184 ± 0.01594|+0.001|2/3|31.0|
|100|7|3/3|0.81801|0.81798 ± 0.00082|+0.003|2/3|20.0|
|100|14|3/3|0.96975|0.96964 ± 0.00137|+0.012|2/3|18.0|

현재 완료 seed 조건 중 RMSE 개선 16/24. 개선율 범위 -0.817% ~ +1.962%.
고정 진단 batch의 Rres RMS가 epoch1보다 epoch80에 감소한 조건 24/24. 최종/초기 RMS 비 중앙값 0.0712251.
Auxiliary gradient norm 최종/초기 비 중앙값 0.00344304; 최종 gradient가 정확히0인 조건 0/24. 작은 양수를 0이라고 표시하지 않았다.
가중 Aux/Data gradient 비 중앙값 epoch1=0.0175878, epoch80=0.0004071. 초기만 작용하는지는 감소율과 최종 크기를 함께 해석해야 한다. 정규화 출력 감소 자체를 SST 성능 향상 근거로 쓰지 않는다.

전체120개 비교는 all_models_validation_by_seed.csv/all_models_validation_summary.csv에 저장한다(미완료 신규 모델 제외). epoch_metrics.csv는 실제 학습 epoch별 기록, diagnostics.csv는 별도 고정 batch 진단, seed_improvements.csv는 seed 대응 개선율과 초기/최종 진단을 제공한다.

기존 경로: /ceph_data/yeonwoo/MHW_subset/outputs/v8_actual_epoch_oracle96_20261011
신규 경로: /ceph_data/yeonwoo/MHW_subset/outputs/v8_actual_epoch_auxiliary24_20261015
