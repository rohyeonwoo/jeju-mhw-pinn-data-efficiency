# 실제 batch Gradient 감사
시각: 2026-10-15 01:44 KST

3 actual Train batches, ratio10 lead7 seed42, batch16; no optimizer updates

|Batch|Data loss|Aux loss|공유 Data gradient norm|공유 Aux gradient norm|가중 Aux/Data 비|Rres head gradient norm|cosine|
|---|---|---|---|---|---|---|---|
|0|0.95301729|142.93689|10.196846|7559.2662|0.074133378|5451.1309|-0.11852536|
|1|1.0019132|120.40386|12.775825|6679.1561|0.052279645|4737.6586|-0.1003818|
|2|1.4043624|117.68012|24.07779|6961.2364|0.028911442|4851.4842|-0.098419677|

Physics 함수 호출: 0, Auxiliary 호출: 6 (초기3 + 반사실3). 초기 gradient는 finite/nonzero이며 **초기 신호 존재 검사 PASS**.

|반사실 Batch|Aux loss|공유 Aux gradient|Rres head gradient|SST 예측 동일|Data loss 동일|
|---|---|---|---|---|---|
|0|0.0|0.0|0.0|True|True|
|1|0.0|0.0|0.0|True|True|
|2|0.0|0.0|0.0|True|True|

마지막 Rres 층의 257개 weight/bias만0으로 하면 정확한 zero-loss 해가 존재한다. **퇴화해 배제 검사 FAIL**. optimizer step은0회. 검사 후 원래 모델 전체 state hash 복구 확인.
GPU5 프로세스 이름 yeonwoo 확인; 최대 reserved 9890.0 MiB.
