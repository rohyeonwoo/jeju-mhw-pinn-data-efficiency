# Preflight: FAIL — 24개 본 학습 보류

|검사|판정|범위/근거|
|---|---|---|
|Auxiliary 정의/가중치|PASS|기존 선택 λaux=0.0001, fixed RHS-normalized zero-target Rres|
|Physics 비활성|PASS|실제6호출에서 physics 함수0회; 호출 시 예외를 내도록 검사|
|초기 shared/head gradient|PASS|10% lead7 seed42, 실제 Train3batch finite/nonzero|
|head-only zero 퇴화해 배제|FAIL|마지막257개 parameter만0 → auxloss/sharedgrad0, SST 동일|
|초기 Predictor 동일|PASS (대표1조건)|독립 재구성 Baseline과 state tensors torch.equal; 저장 Oracle initial SHA와 일치|
|v3 통계/분할/입력 fingerprint|PASS (대표1조건)|원 Oracle96 config와 비교; 다른23조건 실제 bundle 검사는 NOT_RUN|
|Actual Epoch sampling|PASS (대표1epoch)|376samples, 24updates, 원 Baseline epoch1 순서SHA 일치|
|80epoch LR schedule|PASS (대표1조건)|원 Baseline history 80개 LR와 대조|
|Early stopping OFF|계획 고정 / 학습 NOT_RUN|Oracle96 원 설정 유지; 정식 루프는 시작하지 않음|
|Checkpoint save/load|PASS (감사 전용)|initial_roundtrip.pt state tensor 비교; 학습 best checkpoint 아님|
|GPU 이름/자원|PASS (감사GPU5)|NVML yeonwoo 및 지정 UUID; 본24개 배정 NOT_RUN|
|보호 입력 무결성|PASS|protected_audit_inputs.json 원문SHA 전후 일치|

검증한 대표조건을 전체24조건 PASS로 확대하지 않았다. 퇴화해 차단 조건 때문에 나머지 실행 준비/전체 검사는 진행하지 않는다.
