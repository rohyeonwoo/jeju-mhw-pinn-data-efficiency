# Auxiliary24 Actual Epoch protocol
작성 2026-10-15 01:53 KST

Ratios10/25/50/100 × lead7/14 × seed42/7/123. 24개 신규 학습. 기존96개 read-only.
Loss = Data MSE + 0.0001 Rres penalty, fixed v3 RHS scale. Physics OFF, entrainment tie OFF, auxiliary warmup 없음, heads/초기화 unchanged.
80actual epochs, early stopping OFF, batch16, AdamW(.9,.95)/weight decay.1, clip norm1, peakLR.0002, initialLR.00002, warmup5actual epochs + native cosine. 전체 Validation SST RMSE로 best 선택. Test 금지.
매 epoch train index 정확히1회, 대응 Oracle96 Baseline의 shuffle order/Scheduler와 런타임 대조. 각 run 시작 시 독립 재구성 모델의 tensor 및 원 Baseline initial SHA, 통계/split/input fingerprint 일치 검사.
GPU [5, 6, 7], GPU당 우리 작업 최대2, 14GiB/worker 예약+4GiB headroom, 타 사용자 메모리 포함. tmux 유지 및 PID/GPU/log 기록. 동일 config epoch-boundary resume 지원.
매 epoch Rres RMS·loss·가중비·마지막층norm 기록. epoch1/5/10/20/80 및 모든 best 후보에 고정 Train16 batch 별도 autograd.grad 진단; model state/RNG/.grad/mode 보존 검사. 실제 optimizer 1step에 대해 진단ON/OFF bitwise 일치 smoke 후 확장.
zero-head 해는 예상되는 정규화 최소해이며 학습 장애가 아니다. shared regularization이 약하거나 초기에만 작용할 가능성을 진단한다. 성능에 따라 설정을 바꾸지 않는다.

진단 mode: cuDNN GRU backward를 위해 Train mode로 별도 forward하며 원래 module modes를 복원한다. 매번 RNG/state/기존 .grad 동일성 검사. Smoke의 진단ON/OFF 비교는 버리는 모델에서 각각1회 optimizer step(총2회); 본학습 초기값이나 optimizer에 이어 붙이지 않는다.
