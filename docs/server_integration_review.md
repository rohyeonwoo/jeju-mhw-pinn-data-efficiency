# 서버 환경 및 학습 연결 조사 (2026-09-24)

## 작업 위치와 확정 방침

- 실행 작업공간: `/home/yeonwoo/mhw_subset` (`~/mhw_subset`). 사용자 최신 지정 위치다.
- 전송 원본: `/ceph_data/yeonwoo/MHW_subset`. 실행용 사본을 만들었으며 이동·삭제하지 않았다.
- 복사 시점 원본 242개 파일의 SHA-256이 사본과 모두 일치했다. 기준은 `server_preflight/original_snapshot_sha256.json`이다.
- 사용자에게 요청 메시지의 '미결정'과 인수인계 문서 7절의 '확정' 불일치를 확인했고, 사용자가 **문서 7절 방침으로 확정**했다. AGENTS.md의 과거 미결정 표기보다 이번 사용자 결정을 우선한다.
- 원본 `src/`, 분할 함수, 모델, subset JSON, 기존 Full 결과를 보존한다. 이번 단계는 환경 및 연결 지점 조사이며 본 학습 실행은 포함하지 않는다.

## 원본 코드의 연결 지점

| 항목 | 위치 | 조사 결과 및 연결 방법 |
|---|---|---|
| 시간 분할 | `src/mhw/datasets.py:120` | `make_time_splits`를 그대로 사용. subset 적용 후 재분할하지 않는다. |
| Full 통계 | `src/mhw/jeju_hilam_dataset.py:152` | `make_jeju_hilam_bundle`이 Full train/val/test와 통계를 만든다. 먼저 원본 bundle 전체를 만든다. |
| 실제 통계 범위 | `src/mhw/datasets.py:132` | 입력·타깃 평균/표준편차는 첫 날짜부터 마지막 Full train target까지의 연속 날짜 범위. observed tendency 통계는 Full train 시작 인덱스에 대응한 관측값으로 계산한다. |
| 학습 subset | `bundle.train` / `.base.indices` | JSON의 시작 인덱스를 Full train의 **dataset 위치**로 대응시킨 뒤 `torch.utils.data.Subset`으로 감싼다. 시작 인덱스와 위치가 현재 같더라도 직접 동일하다고 가정하지 않는다. |
| 불변 bundle | `src/mhw/jeju_hilam_dataset.py:40` | frozen dataclass이므로 `bundle.train = ...` 대입을 하지 않는다. 별도 `train_dataset` 변수를 사용하고 feature 이름은 원본 bundle에서 읽는다. |
| RHS scale | `src/train_hybrid_jeju_forecast.py:810` | 원본 `compute_fixed_rhs_scale`에 shuffle=False Full train loader를 전달. 관측 target을 선형 경로 끝점으로 쓰므로 Full 타깃 정보도 포함한다. |
| 학습·물리 계산 | `src/train_jeju_hilam.py:257` | 원본 `run_epoch`, 모델 클래스, 물리 함수를 import하여 재사용할 수 있다. 모델 코드를 복제하거나 물리식을 재구현할 필요가 없다. |
| LR/physics warmup | `src/train_jeju_hilam.py:223`, `src/train_hybrid_jeju_forecast.py:524` | 원본 함수에 virtual_epoch을 전달하면 확정된 구간별 LR와 물리 가중치를 유지할 수 있다. |
| 기존 전체 루프 | `src/train_jeju_hilam.py:702` | 원본 main은 subset 한 순회마다 검증하므로 그대로 호출하면 S-step 검증 일정과 맞지 않는다. 별도 실행기의 바깥 루프가 필요하다. |
| 평가 | `src/train_jeju_hilam.py:488`, `src/summarize_jeju_hilam_mhw_skill.py:393` | 실제 기준은 전체 날짜 수의 앞 round(0.70*T)일이다. 각 lead의 train target 끝과 같다고 가정하지 않고 기존 날짜 마스크·임계값 함수를 유지한다. |

## 일정 구현에서 주의할 실제 충돌

`run_epoch`은 일반 iterable을 순회하므로, 원본 함수를 수정하지 않고 정확히 S개 batch를 제공하는 어댑터를 추가 실행기에 둘 수 있다. 어댑터가 실제 subset DataLoader iterator를 계속 보유하고, subset 소진 때만 새 iterator를 만들어 다시 shuffle한다. 검증 경계에서는 iterator를 초기화하지 않는다.

원본 `max_batches`는 batch를 먼저 꺼낸 뒤 제한을 검사한다. 무한 iterator에 `max_batches=S`만 지정하면 학습하지 않는 batch 하나를 추가로 소비할 수 있다. 따라서 **유한 S-batch iterable을 넘기고 max_batches=None**으로 호출해야 한다. 작은 마지막 batch는 그대로 유지하며 optimizer도 구간마다 유지한다.

이는 원본 모델·데이터로더를 변경해야 하는 충돌은 아니다. 별도 실행기의 제어 흐름으로 해결 가능하다. 실제 어댑터 구현, 경계 검증, baseline/P0의 짧은 학습 동작 확인은 아직 하지 않았다.

| lead | 원래 Full train | S=ceil(N/8) | 최대 step | warmup step | val/test |
|---|---:|---:|---:|---:|---|
| 7 | 7,644 | 956 | 76,480 | 4,780 | 1,638 / 1,639 |
| 14 | 7,639 | 955 | 76,400 | 4,775 | 1,637 / 1,638 |

- 다음 batch 직전 virtual_epoch=floor(완료 step/S)+1. 각 구간 LR/물리 가중치를 일정하게 유지한다.
- S step마다 전체 validation, val_rmse 기준 best 선택 및 12회 연속 미개선 종료. 최대 80구간이며 실제 종료 step은 실행마다 다를 수 있다.
- AdamW betas=(0.9,0.95), weight_decay=0.1, LR=2e-4, initial LR=2e-5, warmup 5구간, gradient clipping=1.0을 원본대로 유지한다.
- 원래 실행기는 precision 인자를 지정하지 않으므로 기본 fp32다. 이번 단계에서 bf16 등으로 바꾸지 않는다.

## 통계와 반복 실험 해석

- 각 lead의 **원래 Full** 정규화 및 RHS scale을 비율·모델·seed 전체에 공통 적용한다. 100% subset 통계가 아니다.
- 선택되지 않은 날짜의 입력·타깃도 Full 통계에 포함된다. 연구 해석은 **Full 기준 통계를 유지하면서 gradient 학습 샘플을 줄인 실험**이다. 전체 정보 접근량을 비율만큼 줄였다는 의미가 아니다.
- subset seed=학습 seed: 42↔42, 7↔7, 123↔123. 동일 seed의 baseline/P0 차이를 짝지어 비교한다. 반복 간 변동에는 subset 선택과 학습 난수가 함께 들어간다.
- 100% subset은 3,995개이며 세 seed에서 같은 샘플이다. 학습 난수가 달라 3회 모두 실행한다. 기존 Full 7,644/7,639개 결과는 별도 기준점이다.
- 비율 4 × seed 3 × 모델 2 × lead 2 = 48회. 새 결과는 `outputs/data_efficiency/`에 두고 기존 Full과 분리한다.

## 환경과 검증 증거

- conda 이름: `mhw_subset`; 환경 위치: `/ceph_data/yeonwoo/conda_envs/mhw_subset` (서버 기존 envs_dirs 설정을 따른다).
- Python 3.11을 사용하고, 드라이버 535.230.02 환경에 맞춰 PyTorch 2.5.1 CUDA 12.1 wheel을 선택했다. 설치 명령의 근거: [PyTorch 공식 이전 버전 설치 안내](https://docs.pytorch.org/get-started/previous-versions/).
- 서버 node1: RTX A6000 48GB × 10, 점검 시 GPU 작업 없음. 메모리 약 1TiB, home 파일시스템 여유 약 205GB. 유휴 상태 확인은 향후 전용 할당을 뜻하지 않는다.
- 설치 로그와 환경 버전, 재현성 검증, Full 통계 대조 결과는 `docs/server_preflight/`에 저장한다. `audit_report.json`이 status=passed일 때 전체 점검 성공으로 판단한다.
- `scripts/audit_server_preparation.py`는 원본 함수로 Full bundle/RHS를 계산하고, 실제 subset sample의 current/target 날짜 인덱스, val/test 보존, Full 설정 일치, 전후 파일 해시를 검증한다. 모델 학습은 수행하지 않는다.

```bash
cd ~/mhw_subset
conda activate mhw_subset
python -B scripts/run_as_yeonwoo.py /ceph_data/yeonwoo/MHW_subset/scripts/make_train_subsets.py --check
python -B scripts/run_as_yeonwoo.py scripts/audit_server_preparation.py --device cuda:0
```

GPU 0은 준비 단계의 짧은 환경·RHS 계산 점검에만 사용한다. 본 실험의 GPU 배치 실행기는 아직 만들지 않았다.


## 완료된 검증 결과

- subset seed 42/7/123 모두 바이트 재현성 검사 통과.
- pip check, 원본 학습 CLI help, CUDA 실제 연산·autograd 통과.
- 12개 Full 설정 대조, 24개 subset view의 실제 sample 인덱스 점검, val/test 불변, 보호 파일 233개 전후 해시 동일.
- lead7: Full train/val/test=7644/1638/1639, RHS=9.73672297779631; 저장 Full 타깃 평균·표준편차와의 최대 절대차 1.14440918e-05 K. RHS는 저장값과 동일했고 타깃 통계는 rtol=1e-6, atol=1e-6 범위에서 일치했다.
- lead14: Full train/val/test=7639/1637/1638, RHS=9.72798431889206; 저장 Full 타깃 평균·표준편차와의 최대 절대차 5.7220459e-06 K. RHS는 저장값과 동일했고 타깃 통계는 rtol=1e-6, atol=1e-6 범위에서 일치했다.
- 통계 날짜 끝: lead7 2014-01-10, lead14 2014-01-12. 원본 ACC/MHW 기준 날짜 끝: 2013-12-31.
- 새 의존성 환경의 미세한 통계 수치 차이는 위와 같이 기록했다. 모델 학습·추가 학습 실행기 구현은 아직 수행하지 않았다.


## GPU 이름 규칙

모든 GPU 실행은 `scripts/run_as_yeonwoo.py`를 통해 Process name=yeonwoo를 강제한다. setproctitle 1.3.7을 설치했고 실제 nvidia-smi 조회 검증을 통과했다. 기존 src는 수정하지 않았다.


## 저장 위치 정정

로컬 데이터 실체는 제거하고 Ceph 원본 링크로 교체했다. `outputs/data_efficiency`도 Ceph 링크다. 로컬에는 코드·문서·작은 설정과 지표만 둔다. subset 바이트 재현성은 위 Ceph 원본 생성기로 검사하며 로컬 subset/기준 소스와의 바이트 동일성을 별도 확인한다. 원본 JSON과 모델은 변경하지 않는다.


## 본 학습 실행 상태 (2026-09-24)

이 문서의 과거 미구현 표기 이후 별도 실행기와 3-GPU 큐를 구현했다. 일정 단위 검증 및 baseline/P0×lead7/14 네 smoke, 원본 MHW 평가 연동을 통과했다. 사용자 요청으로 tmux mhw_subset에서 GPU 0/1/2로 48회 큐를 시작했다. 실제 상태는 Ceph outputs/data_efficiency/_queue/batch_status.json과 개별 progress.json을 확인한다.


## 현재 GPU 배치 (사용자 변경)

기존 3-GPU 시도는 stopped_by_user 상태로 보존했다. 현재 실행은 allocation_5plus4/이며 GPU 0에 10% 세 seed와 25% seed42/7, GPU 1에 25% seed123과 50% 세 seed를 배정한다. 각 슬롯의 모델·lead 네 실행 후 빈 슬롯에서 100% 12개를 이어서 실행한다. tmux 세션 mhw_subset_5plus4, 전체 48개/동시 최대 9개다. batch8과 학습 조건은 동일하다.


## 최신 실행: batch16 / 5+5

batch8 실행은 사용자 지시로 중단·보존했다. batch16, GPU0/1 각각 5개로 allocation_batch16_5plus5/에서 재시작했다. S=478, max=38240, warmup=2390. batch16 네 smoke와 작은 마지막 batch 메모리 검증 통과. 기존 Full은 batch8 기준점임을 구분한다. 현재 상태는 current_attempt.json 포인터를 따른다.
