# SeaCast-PINN Data Efficiency 실험

## 실험 목적
학습 데이터 양이 줄어들 때 physics constraint(PINN)가 inductive bias 또는 정규화 역할을 하여
SeaCast baseline보다 더 나은 성능을 보이는지 확인한다.

## 비교 모델
- SeaCast baseline
- SeaCast + PINN (물리식은 기존 실험에서 선정한 P0로 고정)

## 고정 조건 (절대 변경 금지)
- 모델 구조 및 코드는 기존 실험과 동일하게 유지한다. 모델 코드를 수정하지 말 것.
- lead time: 7일, 14일만 비교
- history: 30일, 최대 lead: 14일, WINDOW: 44일
- 데이터 분할은 기존 최종 실험의 `mhw.datasets.make_time_splits` (시간순 샘플 수 기준 70/15/15)를 그대로 사용한다.
- 각 lead의 검증·테스트 인덱스를 유지하며 학습 인덱스 안에서만 subset을 적용한다.
- 학습 데이터 양 외의 모든 조건은 동일하게 유지

## 기준 코드·데이터와 실제 기간
- 주 코드 폴더(C): `src/` (기존 코드의 내부 구조와 파일 내용 유지)
- 모델 정의: `src/mhw/models/`; 데이터로더: `src/mhw/datasets.py`, `src/mhw/jeju_hilam_dataset.py`
- 기준 설정: `src/run_main_experiment_final.ps1`, `outputs/main_experiment_final/experiment_manifest_final.json`
- 데이터: `data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc`
- 기존 manifest의 `D:/MHW/` 경로는 현재 작업공간 루트에 대응시킨다. 생성 JSON에 실제 파일 SHA-256을 기록한다.
- 최상위 폴더 역할: `src/` 코드, `data/` 데이터·subset, `outputs/` 기존 Full 결과, `scripts/` 실험 준비 도구, `docs/` 연구·작업 기록. 안내는 `README.md`에 있다.
- 기존 학습·평가 코드와 Full 결과 JSON의 D:/MHW 절대경로는 원본 기록 그대로 유지했다. 실행 시 현재 데이터·출력 경로를 명시해야 하며, 저장 결과 재평가 시 내부 경로 참조도 별도로 확인한다.
- 실제 데이터 날짜: 1993-01-01 ~ 2022-12-31 (10,957일)
- 아래 시작일은 입력 윈도우의 첫 날짜이며, 타깃 날짜 = 시작일 + history - 1 + lead이다.

| lead | 분할 | 원래 샘플 수 | 샘플 시작일 범위 | 예측 타깃 날짜 범위 |
|------|------|-------------|-----------------|-------------------|
| 7 | 학습 | 7,644 | 1993-01-01 ~ 2013-12-05 | 1993-02-06 ~ 2014-01-10 |
| 7 | 검증 | 1,638 | 2013-12-06 ~ 2018-05-31 | 2014-01-11 ~ 2018-07-06 |
| 7 | 테스트 | 1,639 | 2018-06-01 ~ 2022-11-25 | 2018-07-07 ~ 2022-12-31 |
| 14 | 학습 | 7,639 | 1993-01-01 ~ 2013-11-30 | 1993-02-13 ~ 2014-01-12 |
| 14 | 검증 | 1,637 | 2013-12-01 ~ 2018-05-25 | 2014-01-13 ~ 2018-07-07 |
| 14 | 테스트 | 1,638 | 2018-05-26 ~ 2022-11-18 | 2018-07-08 ~ 2022-12-31 |

## 학습 데이터 subset 설계
- 기존 `make_time_splits`를 직접 import하여 lead 7·14의 학습 시작 인덱스 교집합을 구한다.
- 공통 후보: 시작 인덱스 0~7,638, 시작일 1993-01-01 ~ 2013-11-30, 총 7,639개
- WINDOW = 44일 (history 30 + 최대 lead 14). 두 lead는 같은 시작일·시작 인덱스를 사용한다.
- 블록을 나눌 공통 학습 날짜 범위: 1993-01-01 ~ 2014-01-10 (7,680일). 각 lead의 학습 샘플이 사용하는 전체 날짜 범위(입력 시작~최종 타깃)의 교집합이며, 후보 시작일 범위와 구분한다.
- 첫 날짜부터 90일 비중첩 연속 블록을 구성한다. 완전한 블록은 85개, 1993-01-01 ~ 2013-12-11이다.
- 끝의 30일(2013-12-12 ~ 2014-01-10)은 자투리로 제외한다.
- 각 블록에서 44일 전체가 포함되는 시작일 47개만 사용하며, 모두 공통 후보에 속해야 한다.
- 시간순 블록을 4개 구간(22/21/21/21개)으로 나눈다.
- 계절은 블록 중심 날짜(시작일 + 44일) 기준: 겨울 12~2월, 봄 3~5월, 여름 6~8월, 가을 9~11월.
- seed별로 기본 계절 순서를 한 번만 섞고, 구간 P1/P2/P3/P4는 이를 각각 0/1/2/3칸 왼쪽 회전시켜 계절 round-robin을 시작한다. 각 구간의 계절 내부 블록 순서는 seed별로 무작위로 섞는다. 이후 한 번 섞은 구간 순서로 구간 round-robin을 수행하여 전역 순위를 한 번만 만든다.
- 비율별 목표는 Python `round(비율 × 85)` (정확히 .5인 경우 짝수 쪽 반올림)이며, 전역 순위의 앞부분을 선택한다. 구간별 블록 수 차이는 최대 1개이다.
- nested 구조: 10% ⊂ 25% ⊂ 50% ⊂ 100%
- 서브샘플링 seed: 42, 7, 123

| 구간 | 날짜 범위 | 블록 수 |
|------|-----------|---------|
| P1 | 1993-01-01 ~ 1998-06-03 | 22 |
| P2 | 1998-06-04 ~ 2003-08-06 | 21 |
| P3 | 2003-08-07 ~ 2008-10-08 | 21 |
| P4 | 2008-10-09 ~ 2013-12-11 | 21 |

| 비율 | 블록 수 | 시작일 수 | 고유 날짜 수 | 공통 학습 7,680일 대비 커버리지 |
|------|---------|-----------|--------------|-------------------------------|
| 100% | 85 | 3,995 | 7,650 | 99.6094% |
| 50% | 42 | 1,974 | 3,780 | 49.2188% |
| 25% | 21 | 987 | 1,890 | 24.6094% |
| 10% | 8 | 376 | 720 | 9.3750% |

- 위 수치는 세 seed 모두 동일하다. 10%에서도 네 구간이 각각 2개 블록씩 포함되며, 겨울/봄/여름/가을도 각각 2개 블록씩 포함되는지 assert한다.
- 100%는 전체 원래 학습 샘플이 아니라, 완전한 90일 블록 안에 들어가는 44일 샘플 전체를 뜻한다.
- 이 100% subset(3,995개)은 공통 학습 후보 7,639개(lead14의 원래 전체 학습 샘플 수)보다 적다. 기존 본 실험 결과를 별도의 Full 기준점으로 사용한다. 기존 Full의 lead7 학습 샘플은 7,644개이며 이를 7,639개로 바꾸지 않는다.
- 고유 날짜 커버리지는 44일 윈도우들의 날짜 합집합 기준이다. 유지된 7,650일 대비 비율도 JSON에 별도로 기록한다.

## 사용 파일
- 생성 코드: `scripts/make_train_subsets.py`
- subset: `data/subsets/train_subsets_v2_win44_seed{42,7,123}.json`
  (비율 키 `100`, `50`, `25`, `10` 아래 `selected_blocks`, `start_dates`, `start_indices`, `summary`; `_metadata`에는 원래 분할, 후보·블록 범위, 순위와 원본 해시 포함)
- 실행: `python -B scripts/make_train_subsets.py`
- 재현성 검증: `python -B scripts/make_train_subsets.py --check` (파일을 쓰지 않고 재생성 결과와 바이트 단위 비교)
- 설계 변경 후 기존 v2 결과 재생성: `python -B scripts/make_train_subsets.py --overwrite` (검증 assert 통과 후 갱신)
- 실행 환경: numpy, xarray, torch, NetCDF 읽기 backend 필요. 원래 C의 함수를 직접 import한다.
- 구형 subset, 구형 사본과 층화 없는 초기 노트북은 사용자 백업 및 삭제 승인 후 2026-09-24 정리 과정에서 삭제했다. 이번 실험에서는 위 v2 파일만 사용한다.
- GitHub 준비: `.gitignore`는 대용량 데이터·체크포인트·예측 배열과 로컬 환경 파일을 제외한다. subset JSON, Full manifest, 설정·지표·요약은 포함한다.
- 설치 의존성: `requirements.txt`. 설치·데이터 준비·실행 안내는 `README.md`를 따른다. 원래 Full 학습 환경의 lockfile은 아니다.
- `.gitattributes`는 원본 소스·Full manifest·subset JSON의 줄바꿈 변환을 막아 원본 해시와 subset 재현성을 보존한다.
- 과거 정리·이동 기록(`docs/cleanup_*.json`, `docs/cleanup_plan.md`, `docs/layout_migration.json`)은 현재 작업공간에 없으며 GitHub 포함 대상도 아니다. 당시 변경 내용은 `docs/research_log.md`에 남아 있다.
- 이번 단계에서는 subset 생성만 완료했다. 기존 데이터로더·학습 코드·실행기는 변경하지 않았으며, v2 연결은 다음 단계이다.

## 아직 정하지 않은 학습 설정
- 학습량 기준: 전체 iteration 고정 vs early stopping
- 정규화 통계: 100% 기준 공통 적용 vs subset별 계산
- PINN 손실 가중치: 모든 비율에서 고정 예정

## 작업 원칙
- 파일 삭제, 이동, 모델 코드 수정은 반드시 목록을 먼저 보여주고 승인을 받은 뒤 진행한다.
- 주요 결정과 이유는 docs/research_log.md에 날짜별로 기록한다.


## 서버 GPU 프로세스 이름 규칙 (2026-09-24 사용자 지시)

- 모든 GPU 점검·학습 프로세스는 `yeonwoo` 계정으로 실행하고, nvidia-smi의 Process name에도 정확히 `yeonwoo`가 표시되어야 한다. PID 칸은 숫자다.
- `mhw_subset` 환경의 setproctitle은 필수다. `scripts/requirements-server.txt`에 서버 추가 의존성을 기록했다.
- 기존 원본 src를 수정하지 않고 `python -B scripts/run_as_yeonwoo.py <실행할 Python 파일> [인자...]`로 실행한다. 이 래퍼는 GPU 사용 전에 이름을 지정한다. 향후 별도 학습 실행기와 subprocess 실행도 이 규칙을 적용한다.
- 새 GPU 진입점을 직접 실행해야 한다면 CUDA 사용 전에 setproctitle.setproctitle("yeonwoo")를 호출하고 라이브 nvidia-smi 표시를 검증한다. MHW_PROCESS_NAME 환경변수만 지정해도 미호출 코드에 자동 적용되는 것은 아니다.
- 검증 명령: `python -B scripts/run_as_yeonwoo.py --check-gpu-name`. 모델 학습 없이 GPU를 잠깐 사용해 자신의 PID/Process name을 검사하고 종료한다.


## 로컬 디스크와 Ceph 저장 규칙 (2026-09-24 사용자 정정, 최우선)

- `~/mhw_subset`에는 코드·문서·작은 설정/지표/subset JSON만 보관한다. 대용량 NetCDF, 체크포인트, 예측 배열, 대용량 로그·캐시를 로컬에 복사하거나 생성하지 않는다.
- 데이터 실체: `/ceph_data/yeonwoo/MHW_subset/data/`. 로컬 `data/`의 NetCDF는 Ceph 원본을 가리키는 심볼릭 링크다. 링크를 실체 파일로 바꾸는 copy/dereference 작업을 하지 않는다.
- 새 실험 출력 실체: `/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/`. 로컬 `outputs/data_efficiency`는 이 폴더를 가리키는 링크다. 향후 학습 실행기는 가능하면 Ceph 절대경로를 output-root로 지정한다. smoke 출력도 이 Ceph 경로 아래에 둔다.
- conda 환경은 이미 `/ceph_data/yeonwoo/conda_envs/mhw_subset`에 있으므로 로컬로 복사하지 않는다. 기존 Full 결과는 보존한다.
- subset 생성기는 데이터 경로를 resolve하여 메타데이터에 기록하므로 로컬 symlink 도입 후 로컬 생성기의 --check는 경로 메타데이터가 달라진다. 원본 JSON을 덮어쓰지 않고 아래 Ceph 원본 생성기로 바이트 재현성을 검사한다. 로컬 subset/분할 소스/실행기/manifest는 Ceph 원본과 별도 바이트 비교해 동일성을 확인한다.

```bash
cd ~/mhw_subset
conda activate mhw_subset
python -B scripts/run_as_yeonwoo.py /ceph_data/yeonwoo/MHW_subset/scripts/make_train_subsets.py --check
```


## 현재 본 실험 큐 (2026-09-24 사용자 재배치)

- 이전 outputs/data_efficiency/_queue/batch_status.json은 stopped_by_user인 과거 시도다. 현재 큐는 Ceph `outputs/data_efficiency/allocation_5plus4/_queue/batch_status.json`, 포인터는 `outputs/data_efficiency/current_attempt.json`이다. 기존 큐를 재시작하거나 중복 실행하지 않는다.
- tmux: `mhw_subset_5plus4`. GPU 0에 10% seed42/7/123 + 25% seed42/7 (5개), GPU 1에 25% seed123 + 50% seed42/7/123 (4개). GPU 2는 사용하지 않는다.
- 각 비율/seed 슬롯의 baseline/P0 × lead7/14 네 실행 후, 먼저 빈 슬롯부터 100% 공용 큐를 처리한다. 100% 포함 전체 48개이며 batch8, fp32, Full 통계·S-step 일정은 유지한다.
- 관리자: scripts/launch_fixed_allocation.py. 학습 실행기: scripts/train_data_efficiency.py. 모든 GPU 이름은 yeonwoo, 모든 대용량 출력은 Ceph에 둔다.


## 최신 실행 조건: batch16 / 5+5 (2026-09-24 사용자 변경, 위 이전 배치보다 우선)

- 사용자가 batch16으로 변경하고 GPU 0/1에 각각 5개 동시 실행을 확정했다. 이전 batch8 5+4 시도는 stopped_by_user이며 9+9 증설은 적용되지 않았다.
- 현재 큐는 Ceph outputs/data_efficiency/allocation_batch16_5plus5/_queue/batch_status.json, tmux mhw_subset_b16_5plus5. current_attempt.json 포인터를 먼저 확인하고 과거 큐를 재시작하지 않는다.
- GPU0: 10% seed42/7/123 + 25% seed42/7. GPU1: 25% seed123 + 50% seed42/7/123 + 100% seed42. 각 슬롯의 두 모델×두 lead 처리 후 나머지 100% seed7/123을 공용 큐에서 처리한다. 총 48개/동시 최대 10개.
- 두 lead 모두 S=478, max_steps=38240, warmup_steps=2390, validation 간격 478. batch 외 기존 모델·분할·통계·손실·LR 조건은 유지한다. 기존 Full은 batch8 결과로 보존·별도 표기한다.
- 학습: scripts/train_data_efficiency.py --batch-size 16. 큐: scripts/launch_batch16_5plus5.py. 모든 GPU 이름 yeonwoo 및 Ceph 저장 규칙 유지.


## Full batch16 후속 기준점 (2026-09-24 사용자 추가 요청)

- 현재 subset 48개 종료 후 batch16 Full 12개(3seed×2lead×2모델)를 추가한다. 총 본 학습 60개다. 기존 Full batch8 결과는 보존한다.
- Full은 lead7 7644/lead14 7639 샘플이며 100% block subset 3995와 구분한다. 동일 batch16/S478/max38240/warmup2390/통계/val-test/평가 조건을 사용한다.
- 후속 큐: scripts/run_full_batch16_after_subset.py, tmux mhw_subset_full_b16_followup. Ceph full_batch16_matched/_queue/batch_status.json은 현재 waiting_for_subset이며 subset 성공 완료 뒤 smoke→Full 순서로 실행한다. 중단된 이전 시도를 dependency로 바꾸지 않는다.
- 학습 실행기의 --full과 --ratio는 상호 배타적이다. 후속 계획과 상태는 Ceph batch16_study_plan.json 및 full_followup.json을 참조한다. 예약 후 runner를 바꾸면 후속 큐의 해시 검토도 갱신해야 한다.


## 최신 진행: lambda 파일럿 18회 (2026-09-25)

- 기존 batch16 subset48+Full12는 완료·보존됐다. 현재 큐는 Ceph outputs/data_efficiency/lambda_pilot_lead7_20260925/_queue/batch_status.json, tmux mhw_lambda_pilot. current_attempt.json 또는 lambda_pilot_current.json을 먼저 확인한다.
- lead7, 10% subset376개와 원래 Full7644개, 3seed, lambda_physics=0/0.01/0.3의 18개. lambda_rres=10 유지이므로 lambda_phys=0은 baseline이 아니라 보조 손실 대조다. 기존 lambda0.1과 baseline을 비교점으로 재사용한다.
- GPU0/1 각각 최대5개. process name yeonwoo, conda mhw_subset, 코드 ~/mhw_subset, 큰 출력 Ceph 규칙 유지.
- scripts/train_lambda_pilot.py와 launch_lambda_pilot.py가 현재 파일럿 실행기다. 원본 src 및 이전 train_data_efficiency.py는 보존한다. 범위·검증·통계 규칙은 docs/server_preflight/lambda_pilot_protocol.json, 검증 결과는 lambda_pilot_validation.json 참조.
- 큐 완료 후 기존 MHW 평가와 summarize_lambda_pilot.py가 _analysis/report.md를 만든다. 25%·lead14·tendency-only·기후값 등 확대 실행은 이번 파일럿에 포함되지 않았다.


## 최신 진행: 해양 이웃 차분 P0/P1 파일럿 30회 (2026-09-25)

- 사용자 명시 승인으로 해양 이웃 중앙/한쪽 차분 보정을 적용한다. 원본src와 기존결과는 보존하고 새 scripts/train_physics_pilot.py와 physics_stencil.py, physics_interval.py를 사용한다.
- 큐: Ceph outputs/data_efficiency/physics_pilot_lead7_20260925/_queue/batch_status.json. tmux mhw_physics_pilot. current_attempt.json 및 physics_pilot_current.json이 이 큐를 가리킨다. 이전 완료 큐를 재시작하지 않는다.
- 10%/lead7/3seed: 맞춘baseline3 + 순수P0일별9 + 순수P0기간평균9 + 순수P1일별9 =30회. 각물리arm lambda.01/.1/.3, lambda_rres0, lambda_we_tie0, zero_rres_physics=True. baseline은추가손실모두0이고동일head초기화/RNG를유지한다.
- 물리차분은바다이웃만사용하고660개격자에적용한다. 온도학습/평가는661개전체다. 새공통Full-train P0 RHS스케일0.08996321398271197을사용하며입력/타깃통계와분할은기존그대로다. 기존RHS스케일과Full/부분자료결과를덮어쓰지않는다.
- batch16, S478, 최대38240step, 기존검증·조기종료조건, GPU0/1각5슬롯. GPU이름yeonwoo와Ceph저장규칙유지. 실제10개학습이시작되어50step이상진행한것을확인했다.
- 실행전프로토콜: docs/server_preflight/physics_pilot_protocol.json. GPU4조건smoke, 동일초기화/배치순서, 평가격자검증통과: physics_pilot_validation.json. 실제시작증거: physics_pilot_launch_check.json.
- 종료후원본MHW평가기와 summarize_physics_pilot.py로모든결과를보고하고검증평균으로lambda를선택한다. 추가비율/lead/Full로자동확장하지않는다. 경계진단과한계는docs/physics_boundary_review.md참조.


## 최신 실행: 해안 보정 파일럿 4+4 재시작 (2026-09-25, 이전5+5보다우선)

- 사용자가메모리여유를위해현재5+5를중단하고4+4로줄이라고명시했다. 이전physics_pilot_lead7_20260925는stopped_by_user이며부분결과를보존했다. 재시작하지않는다.
- 현재큐는Ceph outputs/data_efficiency/physics_pilot_lead7_20260925_4plus4/_queue/batch_status.json, tmux mhw_physics_pilot_4plus4. current_attempt.json/physics_pilot_current.json을먼저확인한다.
- 관리자scripts/launch_physics_pilot_4plus4.py가GPU0/1각4개,총8슬롯을사용한다. 동일30회를동일seed에서처음부터실행한다. 기존checkpoint에는optimizer/stream/RNG전체재개상태가없어서부분재개하지않는다.
- 학습/손실코드는변경하지않았으며기존4조건GPU검증의해시와일치한다. batch16/fp32, P0/P1조건·lambda·660개물리/661개온도격자·통계·분할은모두같다. GPU이름yeonwoo/Ceph규칙유지.
- 확정조건: docs/server_preflight/physics_pilot_4plus4_protocol.json. 시작확인: physics_pilot_4plus4_launch_check.json.
