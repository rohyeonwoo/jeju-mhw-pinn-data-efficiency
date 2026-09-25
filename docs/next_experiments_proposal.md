# 후속 실험 제안 — 범위를 줄인 수정안

> **최신 상태 (2026-09-25):** 초기 제안 중 lambda 파일럿18회는 완료됐고, 해양 이웃 차분을 적용한 P0/P1 후속30회가 GPU0/1 각4슬롯으로 진행 중이다. 아래 초안과 날짜별 추가 기록은 설계 이력이다. 현재 확정 조건과 해석은 [오늘 작업 정리](daily/2026-09-25.md)와 [4+4 프로토콜](server_preflight/physics_pilot_4plus4_protocol.json)을 우선한다.

2026-09-25. 사용자 피드백과 실제 코드·결과 점검을 반영한 제안이다. 실험 실행 조건으로 확정된 것은 아니다. 기존 모델 코드·시간 분할·val/test 인덱스·Full 결과를 보존한다.

## 현재 실행 범위 (2026-09-25 승인)

사용자가 먼저 파일럿을 실행하도록 승인했다. 직전 논의에 따라 **lead7 × 10%·Full × 3seed × 물리 lambda0/0.01/0.3 = 18회**를 완료했다. lambda_rres=10을 유지해 보조 손실 대조와 물리 lambda 민감도를 먼저 확인한다. 아래 24회는 tendency-only까지 포함한 확장 제안이며 이번 큐에는 포함하지 않았다. 기존 baseline과 lambda0.1 결과는 보존·재사용한다. 확정된 이번 분석 규칙은 [프로토콜](server_preflight/lambda_pilot_protocol.json), 검증 증거는 [검증 보고서](server_preflight/lambda_pilot_validation.json)에 있다.

## 파일럿 완료 후 판단 (2026-09-25)

18회와 자동 분석을 모두 완료했다. 10%에서 보조 손실만 남긴 조건이 기존 P0(λ0.1)보다 세 seed 모두 좋았고, 검증 선택 λ0.3과도 평균 RMSE가 거의 같았다. Full 평균 악화도 해소되지 않았다. 따라서 아래 광범위한 확대보다 **초기화·shuffle을 P0와 맞춘 모든 추가 손실 off baseline 6회**를 우선 검토한다. 이는 새 실행 승인이 아니라 결과에 따른 권고다. [상세 검토](lambda_pilot_review.md)에 수치와 해석을 기록했다.

## 목적과 우선순위

교수님이 제안한 자료량별 PINN 효과 비교를 중심에 둔다. 핵심 질문은 (1) 고정 물리 가중치가 Full 악화의 원인인가, (2) P0의 개선은 경향 감쇠만으로 설명되는가이다.

- 이번 단계: 구현 기준점 검증 → lambda 스윕과 tendency-only 대조.
- 선택적 다음 단계: 기후값 forcing 대조. forcing 섞기는 이후 진단으로 미룬다.
- 그다음: subset-only 통계·정보 접근 통제.
- 후속 연구: 별도 해역·기간, 물리 오차별 세부 분석, 적응형 방법, seed 교차 확장. 이번 필수 실행 목록에 넣지 않는다.

## 0. 실행 전에 반드시 분리할 기준점

실제 코드 확인 결과, 현재 P0에서 lambda_physics=0만 설정하면 기존 baseline과 동일하지 않다.

- train_jeju_hilam.py의 전체 loss는 data + lambda_eff*physics + lambda_rres*rres_l2 + lambda_we_tie*we_tie이다.
- 기존 P0 설정은 lambda_rres=10, lambda_we_tie=0. zero_rres_physics=True는 물리 잔차 식에서 Rres를 제외할 뿐 rres_l2 보조 손실까지 끄지 않는다.
- P0는 we/rres head를 추가하며, 그 head는 SST 예측과 공유하는 특징에서 계산된다. rres 억제 항이 공유 backbone에 미치는 영향을 별도 효과로 보아야 한다. 저장된 P0 test_rres_l2도 0이 아니다.
- 현재 실행기에는 lambda override CLI가 없다. 기존 실행기와 결과를 보존하는 별도 후속 경로를 준비해야 한다.

따라서 다음 세 기준점을 혼동하지 않는다.

1. 기존 baseline: 보조 head/보조 손실이 없는 실행.
2. P0 구조 유지, 모든 추가 손실 0: baseline과 동일한 SST 초기 가중치·batch 순서·optimizer 상태로 짧은 CPU 또는 승인된 GPU 수치 검증. 예측·공유 gradient·몇 update 뒤 파라미터의 일치를 확인한다. 추가 head 초기화의 RNG 소비가 loader 순서를 바꿀 수 있어 RNG도 독립적으로 맞춘다.
3. P0 구조와 lambda_rres=10 유지, lambda_physics=0: 보조 손실만 있는 실제 ablation. 이것은 baseline이라고 부르지 않는다. 물리 loss만의 추가 효과를 분리하려면 본 실행 대조군으로 필요하다.

현재 60개 결과는 각 설정 전체의 비교로 유효하다. 다만 개선을 물리 잔차 단독 효과로 귀속하는 설명은 보류한다. 아직 lambda=0 수치 일치 테스트를 실행한 것은 아니다.

## 1. lambda 스윕과 시간 경향 대조

### 가설

- H1: 검증셋에서 선택되는 물리 제약 강도는 자료량에 따라 달라지며, Full에서는 현재 0.1보다 작은 값이 유리할 수 있다.
- H2: 실제 P0는 단순한 SST 경향 감쇠를 넘어서는 추가 이득을 제공할 수 있다.

lambda 최적값이 자료량과 함께 감소해도 “반비례(lambda ∝ 1/N)”가 입증되는 것은 아니다. 세 자료 조건과 성긴 lambda grid에서는 “자료량 증가에 따라 약한 제약이 선택되는 경향”까지만 주장한다. lambda 최적값이 경계에 위치하면 범위 내 최적값이라고 표기한다.

### 설계

- 대표 자료 조건: 10%, 25%, Full; lead7/14; 기존 seed42/7/123.
- 기존 lambda=0.1 결과 보존. 새 lambda_physics=0.01,0.3에서 다른 P0 설정은 유지한다.
- lambda_physics=0, lambda_rres=10 대조로 보조 손실 효과를 따로 확인한다.
- 새 일반 정규화는 공간 평활화보다 tendency-only를 우선한다. 예: ocean mean[((T_pred−Tm)/lead_days)^2]를 공통 train-fixed RHS scale로 정규화. 입력·정답·마스크·단위·warmup·clipping·보조 손실 등은 비교 목적에 맞춰 명시적으로 맞춘다.
- 현재 daily_rollout은 양 끝 SST 사이를 선형 보간한다. 따라서 여기서 tendency는 (T_pred−Tm)/lead_days이며, 일별 독립 출력을 가정한 2차 시간 차분 평활화와 혼동하지 않는다.
- 현재 P0 RHS에는 예측 SST의 공간 기울기에 의존하는 이류와 예측 we가 포함된다. P0를 예측에 무관한 외부 forcing에 대한 회귀로 단순화하지 않는다.
- 공정한 비교에는 손실뿐 아니라 공유 backbone에 대한 가중 gradient 크기, clipping 빈도, 실제 step을 기록한다. 우선 고정 강도 비교와 lambda 민감도를 구분한다. 최적 P0와 최적 tendency-only의 우열을 주장하려면 대조군에도 동일한 검증 기반 강도 탐색 예산이 필요하며 아래 본 실행 수에 튜닝 비용이 추가된다.

P0와 tendency-only의 점추정이 비슷하거나 차이가 유의하지 않다고 “사실상 경향 감쇠였다”고 확정하지 않는다. 먼저 “실제 물리 정보의 추가 이득을 확인하지 못했다”라고 해석한다. 동등성을 주장하려면 사전에 정한 허용 차이와 동등성 분석이 필요하다.

## 2. 기후값 forcing은 선택적 다음 단계

역할 분리에 추가 가치가 있을 때, 물리 loss 경로의 qnet/hm/u/v를 train-only DOY·격자 기후값으로 바꾼다. 입력과 감독 target은 그대로 유지한다. 계절·공간 구조는 유지하되 일별 외부 forcing의 추가 정보가 필요한지 본다. 학습된 we와 예측 공간 기울기를 유지하면 이는 RHS 전체의 기후값 치환이 아니라 외부 forcing 채널의 기후값 치환이다.

- 기후값은 validation/test를 제외한 미리 정한 train 기간에서만 구한다. 현재 Full 통계 공유 프로토콜을 유지한다면 Full-train 기후값이라는 정보 접근 범위를 명시한다. subset-only 통계 실험과 섞지 않는다.
- 기후값 치환도 분산과 gradient를 바꾸므로 동일 lambda가 동일 정규화 강도를 뜻하지 않는다. 공통 단위·분모와 loss/gradient 기록, 대조군 강도 민감도가 필요하다.
- forcing 섞기는 더 후순위로 둔다. 큰 잔차 수치 자체가 곧 큰 gradient를 의미하지 않으며, 특히 현재 RHS는 예측에 의존한다. 단순히 섞은 P0가 나쁘다는 결과만으로 물리 기전을 입증하지 않는다.

## 3. 실행 전 고정할 평가 규칙 — 초안

- 주 지표: 해양 격자 전체 test SST RMSE. MHW CSI/POD/FAR와 사건 onset/duration/intensity는 보조 지표.
- 같은 subset·training seed의 paired 차이 Δ=RMSE(P0)−RMSE(control)를 구하고 3개 차이를 모두 보고한다. 평균±SD와 상대 개선율을 함께 제시한다.
- 대표 주 비교는 10%/lead7에서 P0(0.1) vs tendency-only로 사전 지정하는 안을 제안한다. 이 조건은 기존 결과를 보고 선택했으므로 후속 분석을 독립 확증 연구라고 부르지 않는다. 25%·Full·lead14는 적용 범위 확인으로 모두 보고한다.
- 보조 불확실성: 90일 circular time-block bootstrap 4000회, 동일 날짜 재표집을 모든 모델과 seed에 공동 적용. 30/180일 블록을 사전 지정한 민감도 분석으로 보고한다. 격자를 독립 표본으로 세지 않는다. 이 구간은 학습된 세 모델 쌍에 조건부인 테스트 기간 불확실성이다.
- 주 비교의 탐색적 지지 기준안: 평균 개선과 3seed 개선 방향, 90일 조건부 ΔRMSE 95% 구간의 상한<0을 함께 확인한다. 하나라도 불충족하면 혼재/불확실로 보고하고 다른 지표를 사후 주 지표로 올리지 않는다. seed 전반의 통계적 유의성이라는 표현은 사용하지 않는다. 3쌍 exact two-sided sign-flip의 최소 p는 0.25다.
- 검증 최적 lambda는 조건별 3seed 평균 best validation RMSE로 선정하고 해당 lambda를 세 seed 모두에 공통 적용한다. 동률 처리(작은 lambda 우선), 허용 차이 및 후보군을 실행 전에 확정한다. test로 lambda·epoch·seed를 고르지 않는다. 보조 손실만 있는 lambda=0과 기존 baseline을 구분한다.
- “Full에서 손해 없음”은 차이 CI에 0이 포함되는 것만으로 주장하지 않는다. 실용적 허용 악화 폭 δ를 먼저 정하고 조건부 ΔRMSE 상한<δ 등 비열등성 기준으로 평가해야 한다. δ와 실용적 최소 개선폭은 아직 미정이며 실제 용도에 맞게 확정해야 한다.
- 여러 조건·lambda의 유의성 주장을 동시에 한다면 주 비교군과 다중비교 보정을 먼저 고정한다. 기본안은 단일 주 비교와 나머지 탐색 결과를 구분한다.

## 4. MHW 사건 수 — 현재 출력으로 확인한 값

원본 mhw_skill 평가기와 5개 자료 조건의 결과를 대조했다.

| 정의 | lead7 | lead14 |
|---|---:|---:|
| 지역 관측 MHW 사건 | 30 | 30 |
| MHW 양성 격자×날짜 셀 | 212381 | 212381 |

지역 사건은 해양 격자 중 MHW인 비율이 15%를 초과하는 지역 지수가 최소 5일 지속된 경우다. 두 lead의 같은 사건을 독립 60건으로 합치지 않는다. 30건도 완전히 독립이라고 보장되지 않는다. 격자×날짜 양성 셀은 사건 수나 독립 표본 수가 아니다. fixed_mhw_obs_event_count라는 출력 이름의 212381도 양성 셀 수이므로 21만 사건이라고 해석하지 않는다.

현재 headline mhw_grid_csi/pod/far는 사건별이 아니라 해양 격자×시간 pooled counting이다. 지역 event POD/FAR와 구분한다. 모델별 매칭된 지역 사건 수는 전체 실험에서 8~20개다. 단계별 사건 분할 평가는 이번 필수 분석에서 제외하고, RMSE 주 지표·MHW 보조 지표로 제한한다.

## 5. 비용과 단계별 실행안

실제 환경은 Colab이 아니라 RTX A6000 48GB 서버 GPU0/1, 최대 5+5 동시 실행이다. 아래는 GPU를 공유한 실제 완료 P0의 평균 wall time이며 단독 GPU 시간이나 Colab 시간으로 환산할 수 없다.

| 자료 | lead7 P0 평균 | lead14 P0 평균 |
|---|---:|---:|
| 10% | 1.38시간 | 1.73시간 |
| 25% | 1.49시간 | 1.56시간 |
| Full | 2.67시간 | 2.29시간 |

해당 18개 P0의 process wall time 합계는 33.35시간이다. 서로 겹쳐 실행됐으므로 전용 GPU-hours가 아니다. 같은 종료 step 분포를 가정하면 두 새 lambda 36개는 약 66.7 process-hours, 10슬롯을 완벽히 채운 단순 환산은 약 6.7시간이다. startup/평가/큐 꼬리/서로 다른 early stopping을 제외한 이상적 처리량 참고값이며 완료 ETA로 보장하지 않는다. 새 제약의 step 시간이 비슷해도 early stopping 횟수는 달라질 수 있다.

권장 순서:

1. 본 학습 전에 lambda=0 수치 일치 점검과 짧은 시간·gradient 측정.
2. 먼저 10%·Full × lead7 × 3seed의 6조건에서 새 lambda 0.01/0.3(12회), 보조 손실만 유지한 physics lambda=0(6회), tendency-only 고정 강도(6회): 기본 24회 파일럿. 원래 lambda=0.1/baseline은 보존해 활용한다. 강도 튜닝은 별도다.
3. 설계가 확인되면 25%와 lead14로 확장. 전체 3자료×2lead×3seed에서 새 lambda 두 값 36회 + 보조 손실 대조18회 + tendency-only18회 = 기본 72회. 기후값 forcing은 필요 시 추가18회이며 forcing 섞기는 우선 제외한다.

24회 파일럿을 모두 기존 해당 P0 비용으로 가정하면 약 48.5 process-hours, 10슬롯 이상적 환산 약4.85시간이다. 이는 설계 비교용 참고값이다. 후속 큐는 기존 비율별 고정 슬롯 대신 동적 배정 여부를 검토하고, 실측 전 확정 시간으로 제시하지 않는다. Colab으로 옮기면 GPU 종류·메모리·동시 실행 수·step 시간을 새로 재야 한다.

## 관련 문헌과 해석 범위

- [HW-OPINN](https://www.mdpi.com/2072-4292/18/5/723): MHW 예측에 혼합층 열수지 제약, 가중치·collocation 적응.
- [MARINA](https://www.nature.com/articles/s41612-026-01367-y): physics-aware MHW와 자료 희소 관측 지점 검증.
- [SSTODE](https://arxiv.org/abs/2511.05629): SST 예측에 수송·열수지 정보 결합.
- [SeaCast](https://www.nature.com/articles/s41598-025-31177-w): 지역 해양 GNN과 forcing 민감도. 현재 구현과 원논문의 차이를 정확히 표기한다.

이번 연구의 현실적인 기여 후보는 “자료량에 따른 물리 제약 강도의 효과와 단순 경향 감쇠를 넘어서는 이득의 유무를 구분한다”이다. 양성 결과를 전제로 하지 않으며, 추가 이득을 확인하지 못하거나 강도 조절만으로 설명되는 결과도 보고한다.


## 최신 결정: 해양 이웃 차분 방식으로 30회 진행 승인

사용자가 해양 이웃 중앙/한쪽 차분으로 보정한 30회 실행을 확정했다. 물리손실660개, 온도학습·평가661개, 공통Full-train RHS스케일0.08996321398을 사용한다. 앞선 선택대기 상태는 해소됐다. 실행 전 고정 프로토콜은 `server_preflight/physics_pilot_protocol.json`이다.


**실행 시작 확인:** GPU사전검증4조건을통과했고30회큐를시작했다. GPU0/1각5개로10개실행/20개대기이며50step이상진행을확인했다. tmux `mhw_physics_pilot`, 상태는 Ceph `outputs/data_efficiency/physics_pilot_lead7_20260925/_queue/batch_status.json`, 상세증거는 `server_preflight/physics_pilot_launch_check.json`이다.


**최신 실행 변경:** 사용자요청으로5+5를중단하고동일30회를4+4로새로시작했다. 이전부분결과는보존하며학습조건변경은없다. 현재tmux `mhw_physics_pilot_4plus4`, Ceph `outputs/data_efficiency/physics_pilot_lead7_20260925_4plus4/_queue/batch_status.json`, 8개실행/22개대기다.
