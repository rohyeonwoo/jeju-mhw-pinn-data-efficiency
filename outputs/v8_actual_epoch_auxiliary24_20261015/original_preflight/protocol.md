# Auxiliary-only 24개 계획 — 실행 보류

작성: 2026-10-15 01:44 KST

기존 Oracle96 원본: `/ceph_data/yeonwoo/MHW_subset/outputs/v8_actual_epoch_oracle96_20261011`. 신규 경로: `/ceph_data/yeonwoo/MHW_subset/outputs/v8_actual_epoch_auxiliary24_20261015`.
10/25/50/100% × lead7/14 × seed42/7/123. 기존 96개 재학습 없음. 계획 조합과 Baseline 대응 경로는 run_manifest.csv.

유지할 프로토콜: 80 actual epochs, epoch당 각 선택 Train window 정확히1회, batch16, 기존 v3 manifest/통계/climatology, AdamW betas(.9,.95)/weight decay0.1, gradient clipping norm1, peakLR0.0002/initialLR0.00002, LR warmup5actual epochs 후 native cosine, 매 epoch 전체 Validation SST RMSE, 전체80epoch 중 최저 checkpoint, early stopping OFF. Physics weight0/warmup 비활성, Aux weight0.0001/warmup 없음. Test 자동 실행 금지. 새 hyperparameter tuning 없음.

GPU: 첨부 마지막 지시의 **5·6·7**을 적용(본문7·8·9 대체), GPU당 우리 학습 작업 최대2. 다른 사용자 프로세스를 종료하지 않는다. 감사는GPU5 1개만 사용했고 본 학습 큐는 시작하지 않았다.

차단 조건: 기존 Auxiliary 정의에 predictor와 무관하게 head만0으로 보내는 정확한 퇴화해가 확인됨. 사용자가 요구한 사전검사 조건에 따라 자동24개 확장을 수행하지 않는다. 손실을 임의로 변경하지 않았다.
