# Auxiliary24 최종 사전검사 보고

2026-10-15 01:44 KST

**FAIL — 요청된 사전검사 중 zero-head 퇴화해 배제 조건을 통과하지 못해 본 학습0/24, Test 미실행.**

정의는 명확하고 초기 학습 신호도 있다. 이전 Validation에서 선택한 weight0.0001을 그대로 적용했으며, 실제 Train3batch에서 shared predictor와 Rres head의 보조 gradient는 finite/nonzero였다. Physics loss 함수는 호출되지 않았다.

그러나 Rres 마지막 층257개 parameter만0으로 설정하면 SST 예측과 Data loss는 bitwise 그대로이고 보조손실 및 보조 gradient는 모두 정확히0이었다. 이는 새로운 관측 target이나 예측 정확도 개선 없이 보조 목적을 해결할 수 있는 구조다. 첨부의 “보조손실이 상수이거나 의미 없는0으로 수렴하도록만 구성되어 있지는 않은가” 검사를 충족했다고 표시할 수 없어 자동 확대를 중단했다.

초기 gradient가 없거나 상수 loss라는 판정은 아니다. 실제 과거 학습이 항상 zero-head로 수렴했다는 결론도 아니다. 이 보조 정규화가 초기 optimization에 영향을 줄 가능성은 남는다. 따라서 과거 성능 결과를 무효라고 단정하지 않는다. 다만 이를 독립적인 물리 감독이나 유의미한 auxiliary target 학습의 증거로 해석할 수 없다.

현재 Oracle96 Baseline에도 두 head가 이미 포함되어 있으므로 구조 추가 parameter0개이며 비교 대상은 주로 기존 head 정규화 활성화 효과다. 공유 parameter14,887,681, Rres head66,049, we head66,049. 본 학습 설정을 임의로 수정하거나 다른 보조 task를 만들지 않았다.

## 기존 보조손실 감소 (참고만)

|Seed|기존 종료 epoch|첫 epoch Aux loss|마지막 Aux loss|
|---|---|---|---|
|42|16|1.672098|0.012253|
|7|13|0.622585|0.023374|
|123|15|0.798481|0.009172|

위 과거 실험은 478updates/virtual epoch와 early stopping을 사용했으므로 이번80actual epoch 결과로 대체하거나 직접 같은 예산이라고 해석하지 않는다. 손실 감소 자체는 shared representation 개선과 head 축소를 구별하지 못한다.

## 산출물

- auxiliary_definition_audit.md: 정확한 기존 정의/가중치/코드 위치
- gradient_audit.md 및 gradient_audit.json: 실제3batch 초기 및 반사실 수치
- protocol.md, preflight.md: 유지할 설정과 PASS/FAIL/NOT_RUN 범위
- run_manifest.csv: 24개 계획 조합, 모두 BLOCKED_PREFLIGHT
- run_state.json: 전체 본 학습 미시작 상태
- checkpoint_audit.md: 감사용 초기 모델 roundtrip
- epoch_metrics.csv / validation_summary.csv / seed_improvements.csv: 본 학습이 없어 header만 존재
- gpu_identity.json / gpu_resource_log.csv: 감사 자원 및 process name 확인

현재 결정은 요청된 중단 규칙 준수다. 기존 zero-target Rres를 의도적인 출력 축소 정규화로 평가하는 실험과, 의미 있는 보조 예측 task를 학습하는 실험은 연구 목적이 다르다. 후자를 원할 경우 보조 target/연결 구조를 별도로 정해야 하며 이번 작업에서 임의 변경하지 않았다.
