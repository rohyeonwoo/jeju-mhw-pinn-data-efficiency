# Checkpoint audit

PASS (감사 전용): checkpoints/audit_only/initial_roundtrip.pt 저장/재로드 state tensor 일치. 반사실 head 변경을 복원한 초기 모델만 저장했다. optimizer update0. 정식 best/last 학습 checkpoint는 생성하지 않았다. 기존 Oracle96 및 과거 Auxiliary checkpoint는 수정하지 않았다.
