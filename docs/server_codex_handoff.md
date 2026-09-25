# 서버 Codex 인계: SeaCast-PINN 데이터 효율 추가 실험

작성일: 2026-09-24. 이 문서는 준비 상태와 실험 조건을 전달한다. 학습 완료 보고가 아니다.

## 1. 작업 위치와 먼저 읽을 파일

- 서버: `yeonwoo@node1`, 외부 SSH `203.253.207.61:20001`.
- 프로젝트: `/ceph_data/yeonwoo/MHW_subset`.
- 사용자에게 데이터·코드 전송과 해시 확인 절차를 안내했고, 사용자는 “다 오케이임”이라고 답했다. 서버 상태를 로컬 Codex가 직접 검사한 것은 아니다.
- 먼저 `AGENTS.md`, `README.md`, `docs/research_log.md`를 읽고, 원래 실행 조건은 `src/run_main_experiment_final.ps1` 및 `outputs/main_experiment_final/experiment_manifest_final.json`과 해당 HiLAM 실행별 설정 JSON을 대조한다.
- 이 문서는 기존 AGENTS.md의 제약을 완화하지 않는다.

## 2. 연구 질문과 비교 범위

학습 데이터가 줄어들 때 물리 제약(PINN)이 inductive bias 또는 정규화 역할을 하여 SeaCast baseline보다 성능을 더 잘 유지하는지 검증한다. PINN이 반드시 좋아진다는 결론을 전제하지 않는다.

- 비교 모델: SeaCast baseline vs SeaCast + PINN(P0).
- 저장소에서는 SeaCast-inspired HiLAM 구현을 사용한다. 주 학습 진입점은 `src/train_jeju_hilam.py`이다.
- 기존 Full 실험에 ConvLSTM 등 다른 계열과 lead 1·3 결과가 있어도 이번 추가 실험 범위로 확장하지 않는다.
- baseline은 `--physics-mode none`, P0는 `--physics-mode p0_original`이며 기존 최종 조건의 `--no-residual-feature`를 유지한다.
- P0 수식·계산법은 기존 구현을 그대로 사용한다. 이름만 보고 별도 물리식을 재구현하지 않는다.

## 3. 반드시 유지할 조건

- 모델 구조와 기존 모델 코드를 변경하지 않는다. 원본 `src/`의 구조·내용을 보존한다.
- lead는 7일과 14일, history는 30일. subset 설계의 공통 WINDOW는 44일이다. lead7의 실제 타깃을 14일로 바꾸는 의미가 아니다.
- `src/mhw/datasets.py`의 `make_time_splits`를 직접 사용한다. 시간순 샘플 수 기준 70/15/15 분할을 유지한다.
- 각 lead의 원래 검증·테스트 인덱스를 그대로 두고 학습 인덱스에만 subset을 적용한다. subset 이후 다시 70/15/15로 나누지 않는다.
- 동일 subset seed·비율에서 두 모델과 두 lead는 동일한 시작 인덱스를 사용한다.
- 데이터 양 이외의 비교 조건은 동일하게 유지한다. 새 데이터 전처리, feature 변경, 아키텍처 변경, 비율별 PINN 가중치 튜닝을 추가하지 않는다.
- 파일 삭제·이동·모델 코드 수정은 목록을 먼저 보여주고 승인받아야 한다. 이번 인계 자체는 그러한 변경에 대한 승인이 아니다.
- 주요 결정과 이유는 `docs/research_log.md`에 날짜별로 기록한다.

## 4. 데이터와 원래 분할

데이터: `data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc`

- 실제 날짜: 1993-01-01 ~ 2022-12-31, 10,957일.
- 크기: 1,300,083,528 bytes.
- SHA-256: `fff22a44aeeb35fdd26f29b0d0ade7cf9549c18f5ae29aef2a415a5826708ec2`.
- 시작일은 입력 첫날이다. 타깃 날짜는 시작일 + history - 1 + lead이다.

| lead | 분할 | 샘플 수 | 시작일 범위 | 타깃 날짜 범위 |
|---|---|---:|---|---|
| 7 | train | 7,644 | 1993-01-01 ~ 2013-12-05 | 1993-02-06 ~ 2014-01-10 |
| 7 | val | 1,638 | 2013-12-06 ~ 2018-05-31 | 2014-01-11 ~ 2018-07-06 |
| 7 | test | 1,639 | 2018-06-01 ~ 2022-11-25 | 2018-07-07 ~ 2022-12-31 |
| 14 | train | 7,639 | 1993-01-01 ~ 2013-11-30 | 1993-02-13 ~ 2014-01-12 |
| 14 | val | 1,637 | 2013-12-01 ~ 2018-05-25 | 2014-01-13 ~ 2018-07-07 |
| 14 | test | 1,638 | 2018-05-26 ~ 2022-11-18 | 2018-07-08 ~ 2022-12-31 |

## 5. 확정된 subset 설계

이미 생성·로컬 검증을 마쳤으며 새로 추출할 필요가 없다.

- 파일: `data/subsets/train_subsets_v2_win44_seed{42,7,123}.json` 3개.
- 생성·검증 코드: `scripts/make_train_subsets.py`.
- JSON 비율 키는 문자열 `100`, `50`, `25`, `10`. 각 비율에 `selected_blocks`, `start_dates`, `start_indices`, `summary`가 있고 `_metadata`에 원래 분할·범위·순위·원본 해시가 있다.
- 공통 후보는 두 lead의 train 시작 인덱스 교집합 0~7,638, 총 7,639개다. 시작일은 1993-01-01 ~ 2013-11-30이다.
- 블록을 나눌 공통 학습 날짜 범위는 1993-01-01 ~ 2014-01-10, 7,680일이다. 후보 시작일 범위와 구분한다.
- 첫날부터 비중첩 90일 블록 85개를 만들고 마지막 30일은 제외한다. 각 블록에서 44일 전체가 포함되는 시작 인덱스 47개만 사용한다.
- 시간순 P1/P2/P3/P4 구간은 22/21/21/21개 블록이다. 정확한 날짜는 AGENTS.md를 참조한다.
- 계절은 블록 시작일 + 44일인 중심 날짜로 판단한다. 겨울 12~2월, 봄 3~5월, 여름 6~8월, 가을 9~11월이다.
- seed마다 기본 계절 순서를 한 번 섞고 P1~P4에서 각각 0/1/2/3칸 왼쪽 회전한다. 구간별 계절 내부 블록 순서도 무작위로 섞는다. 계절 round-robin 후, 한 번 섞은 구간 순서로 구간 round-robin을 하여 전역 순위를 한 번만 만든다.
- Python `round(비율 * 85)`개를 전역 순위의 앞부분에서 선택한다. 10% ⊂ 25% ⊂ 50% ⊂ 100%의 nested 구조이며 구간별 블록 수 차이는 최대 1이다.
- subset seed는 42, 7, 123이다. 10%는 네 구간에 각각 2개, 네 계절에도 각각 2개 블록이 포함되는지 assert한다.

| 비율 | 블록 수 | 시작 인덱스 수 | 44일 창의 고유 날짜 수 | 7,680일 대비 커버리지 |
|---|---:|---:|---:|---:|
| 100% | 85 | 3,995 | 7,650 | 99.6094% |
| 50% | 42 | 1,974 | 3,780 | 49.2188% |
| 25% | 21 | 987 | 1,890 | 24.6094% |
| 10% | 8 | 376 | 720 | 9.3750% |

세 seed 모두 위 수치는 같다. 유지된 7,650일 기준 커버리지도 JSON에 별도로 기록했다.

**100% subset과 Full은 다르다.** 100%는 블록 내 3,995개 샘플이다. 기존 Full은 lead7 7,644개, lead14 7,639개이며 별도 기준점이다. Full의 lead7을 7,639개로 줄이지 않는다.

## 6. 완료 사항과 미구현 부분

- 완료: subset 생성, nested·범위·계절 균형 등의 assert, 로컬 재생성 바이트 일치 확인, 원본 SHA-256 기록, 전송 준비.
- **미구현: v2 subset을 학습 데이터로더에 연결하는 처리와 추가 실험용 실행기.** 현재 학습 코드를 그대로 실행하면 Full 학습 분할을 사용한다.
- 원본 모델·학습 코드·데이터로더는 수정하지 않았다.
- 전송 묶음에는 Full JSON/CSV/Markdown을 포함했다. 체크포인트·예측 배열·실행 로그는 제외했다. 저장 지표로 비교할 수 있으나 Full 재추론에는 별도 파일이 필요하다.
- 원래 PowerShell 실행기는 설정 참조용으로 사용할 수 있다. Ubuntu용 추가 실험 실행 방법은 미준비 상태다.

## 7. 확정된 학습 방침 (2026-09-24)

### 학습 일정
- batch size=8을 유지한다. lead별 S=ceil(Full train 샘플 수/8)로 정의한다.
- lead7: S=956, 최대 76,480 step, physics warmup 4,780 step.
- lead14: S=955, 최대 76,400 step, physics warmup 4,775 step.
- 해당 lead 내 모든 비율·모델·seed에 최대 80*S step과 같은 종료 규칙을 적용한다.
- S step마다 전체 검증 분할을 평가하고, val_rmse가 개선되지 않은 검증이 연속 12회면 종료한다. best checkpoint도 val_rmse로 선택한다.
- 이는 동일한 최대 step 예산과 early stopping 규칙이며, 실제 종료 step은 실행마다 다를 수 있다.
- 다음 batch를 학습하기 직전 virtual_epoch=floor(완료한 학습 step/S)+1로 정의한다.
- 기존 LR 함수와 physics warmup 함수에 virtual_epoch을 전달한다. 구간 내부에서 값을 유지하며 매 step 연속 보간으로 바꾸지 않는다.
- LR warmup과 physics warmup은 각각 기존 5 epoch를 5 Full-epoch 구간으로 유지한다. physics 계수는 1~5 구간에서 기존처럼 0.2/0.4/0.6/0.8/1.0이다.
- 검증에는 직전에 학습한 구간의 virtual_epoch을 사용한다. optimizer 상태는 구간마다 초기화하지 않는다.
- subset 순회를 마치면 다시 섞어 계속 학습한다. 검증 경계에서 진행 중인 subset 순회를 재시작하지 않는다. 마지막 작은 batch도 기존처럼 유지한다(drop_last=False).
- 기존 AdamW 및 LR 함수·warmup·학습 루프와의 구현 호환성을 먼저 확인한다. 구조 충돌이 있으면 구현 전에 대상과 이유를 보고한다.

### 정규화와 물리 손실
- 입력·타깃 정규화와 관련 train 통계는 각 lead의 원래 Full 학습 분할 기준으로 계산하여 모든 비율·seed·모델에 공통 적용한다. 100% subset 기준이 아니다.
- PINN RHS scale도 해당 lead의 Full 학습 분할에서 기존 계산법으로 구해 모든 비율·seed에 고정한다. lead 간 값을 강제로 같게 하지 않는다.
- lambda_physics=0.1, lambda_rres=10, zero_rres_physics=true, rhs_normalization=train-fixed, physics_time_mode=daily_rollout을 유지한다. 대상 HiLAM 저장 설정과 대조한다.
- 선택되지 않은 날짜도 통계에 포함되며, RHS scale은 관측 타깃을 사용한다. 입력뿐 아니라 타깃 정보도 Full 통계에 반영된다는 점을 연구 로그와 결과 해석에 명시한다.
- 연구 해석은 'Full 기준 통계를 유지하면서 gradient 학습 샘플을 줄인 실험'이다. 전체 정보 접근량을 해당 비율로 제한했다고 주장하지 않는다.

### seed와 평가
- 학습 seed=subset seed로 대응한다: 42↔42, 7↔7, 123↔123.
- 100%도 학습 seed가 다르므로 3회 모두 실행한다.
- 총 실행 수는 비율 4 × seed 3 × 모델 2 × lead 2 = 48회다. 기존 Full 기준점과 동작 확인 실행은 여기에 포함하지 않는다.
- 같은 seed의 baseline/P0 차이를 짝지어 비교한다. 반복 간 변동에는 subset 선택과 학습 난수가 함께 반영된다.
- ACC climatology, MHW 임계값 및 평가 정의는 기존 Full 평가 코드의 실제 날짜 마스크와 계산법을 그대로 유지한다. subset으로 재계산하지 않는다.
- 기존 Full은 별도 기준점이며 100% subset과 구분한다. 모델, 원래 분할, 검증·테스트 인덱스와 Full 결과를 보존한다.

## 8. 서버에서 다음에 할 작업

1. AGENTS.md, README.md, 연구 로그, 기존 실행기·manifest·HiLAM 실행별 설정을 대조하고 환경·GPU 할당·저장 공간을 확인한다.
2. 의존성을 갖춘 환경에서 `python -B scripts/make_train_subsets.py --check`를 실행한다. 실패 시 원인을 조사하며 JSON을 임의로 덮어쓰지 않는다.
3. 원본 src를 보존하는 추가 실행기로 subset 연결과 위 step 일정을 구현할 수 있는지 확인한다. 모델 코드를 복제하거나 수정하지 않는다. 구조 충돌이 있으면 구현 전에 대상과 이유를 보고한다.
4. Full 통계와 RHS scale을 확보한 뒤 train에만 subset을 적용한다. 날짜를 잘라 이어 붙여 윈도우를 만들지 않는다.
5. train 시작 인덱스, 원래 val/test 유지, 두 모델·lead의 대응 subset, 일정 경계와 통계 공통 적용을 검증한다. 짧은 동작 확인 출력은 본 실험과 분리한다.
6. 새 결과는 `outputs/data_efficiency/` 등 별도 경로에 저장한다. 실행 설정, subset·학습 seed, 비율, 샘플 수, 해시, 통계와 기준 범위, S, 최대·실제·best step, 검증 횟수, LR·warmup, 환경과 지표를 기록한다.
7. 48회 구성과 검증 결과를 보고하고 본 실험을 진행한다. 주요 결정과 이유는 docs/research_log.md에 기록한다.

기존 `D:/MHW` 경로는 원본 기록으로 보존하고 실행 시 실제 경로를 지정한다. requirements.txt는 Full 환경 lockfile이 아니며 Linux 실행 환경은 별도로 확인한다.
파일 삭제·이동·모델 코드 수정은 기존 작업 원칙대로 목록을 먼저 제시하고 승인받는다.
이번 갱신은 문서 방침 확정이며 학습 연결 구현이나 서버 검증 완료를 의미하지 않는다.

## 9. 서버 Codex에 전달할 요청

AGENTS.md와 docs/server_codex_handoff.md의 확정 방침을 읽고 추가 실험 준비를 진행해 주세요. 환경·subset 재현성을 확인한 뒤 Full 통계 공통 적용과 virtual_epoch 기반 일정의 구현 가능성을 보고해 주세요. 기존 로직과 충돌하면 구현 전에 대상과 이유를 알려 주세요. 원본 모델·분할·Full 결과를 보존하고, 연결 검증과 동작 확인 후 48회 실험을 진행해 주세요.
