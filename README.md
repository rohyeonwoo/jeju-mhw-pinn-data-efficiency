# SeaCast-PINN Data Efficiency

학습 데이터 양에 따른 SeaCast-inspired HiLAM baseline과 PINN(P0/P1)의 효과를 비교하는 연구 저장소입니다.
실험 조건은 [AGENTS.md](AGENTS.md), 변경 이력은 [연구 로그](docs/research_log.md)를 기준으로 합니다.

## 현재 진행 상태 — 2026-09-25

**[오늘 작업 정리](docs/daily/2026-09-25.md)**에서 완료된 결과, 해안 차분 보정 근거와 현재 실험을 확인할 수 있습니다. 상태 숫자는 문서 작성 시점의 스냅샷입니다.

| 단계 | 상태 | 자료 |
|---|---|---|
| batch16 subset48 + 원래 Full12 | 완료 | [60회 결과](docs/batch16_results/report.md) |
| lead7 10%/Full lambda 파일럿18 | 완료 | [해석](docs/lambda_pilot_review.md), [집계](docs/lambda_pilot_results/report.md) |
| 해안 차분 진단·보정 | 검증 완료 | [문제와 보정 근거](docs/physics_boundary_review.md) |
| 보정된 baseline/P0/P1 파일럿30 | 4+4 실행 중 | [확정 프로토콜](docs/server_preflight/physics_pilot_4plus4_protocol.json), [상태 스냅샷](docs/daily/2026-09-25_status.json) |

기존 P0의 저자료 개선을 물리 손실만의 효과로 귀속하지 않습니다. lambda0에도 Rres 보조손실이 남는 점과 해안 차분 결함을 확인했으며, 현재는 보조손실을 끄고 같은 초기화·학습 순서를 맞춘 대조 실험을 진행합니다. 후속30회 성능 결론은 아직 없습니다.

**100% block subset3,995개와 원래 Full7,644/7,639개는 별개입니다.** 원본 `src/`, 기존 분할·val/test 인덱스와 Full 결과를 보존하고 변경은 `scripts/`의 별도 실행기에 둡니다.

## 서버에서 이어서 확인하기

서버 코드 폴더는 `~/mhw_subset`, conda 환경은 `/ceph_data/yeonwoo/conda_envs/mhw_subset`입니다. NetCDF·체크포인트·예측 배열·실행 로그는 `/ceph_data/yeonwoo/MHW_subset/`에 둡니다. 홈 폴더에 대용량 파일을 복사하지 않습니다.

```bash
cd ~/mhw_subset
cat outputs/data_efficiency/current_attempt.json
tmux list-sessions
nvidia-smi
```

현재 tmux는 `mhw_physics_pilot_4plus4`이고 GPU0/1에 각4개를 실행합니다. 이미 실행 중인 큐를 중복 시작하지 않습니다. 모든 GPU 프로세스는 `yeonwoo` 계정이며 process name도 `yeonwoo`여야 합니다. GPU 진입에는 `scripts/run_as_yeonwoo.py`를 사용합니다.

- 학습 어댑터: `scripts/train_physics_pilot.py`
- 해양 이웃 차분: `scripts/physics_stencil.py`
- 기간평균 손실: `scripts/physics_interval.py`
- 4+4 관리자: `scripts/launch_physics_pilot_4plus4.py`
- 종료 후 전체 결과 집계: `scripts/summarize_physics_pilot.py`
- 환경·검증 기록: `docs/server_preflight/`, 서버 추가 의존성: `scripts/requirements-server.txt`

아래 환경 준비와 원본 진입점 설명은 최초 subset 준비 단계의 안내도 포함합니다. 현재 서버의 확정 실행 조건은 위 프로토콜을 우선합니다.

## 폴더 안내

```text
MHW_subset/
├── AGENTS.md
├── README.md
├── requirements.txt
├── data/
│   ├── ...glorys3d_budget2d.nc       # 최종 전처리 학습 데이터
│   └── subsets/                    # win44 subset, seed 42·7·123
├── src/
│   ├── train_jeju_hilam.py          # HiLAM baseline/P0 PINN 학습
│   ├── train_hybrid_jeju_forecast.py # 공통 물리 손실 및 ConvLSTM 학습
│   ├── run_main_experiment_final.ps1
│   ├── summarize_*.py              # 평가·결과 집계
│   └── mhw/
│       ├── datasets.py
│       ├── jeju_hilam_dataset.py
│       ├── jeju_hilam_graph.py
│       └── models/                 # HiLAM·ConvLSTM·PINN 모델 정의
├── outputs/
│   └── main_experiment_final/      # 기존 Full 체크포인트·예측·지표·설정
├── scripts/
│   └── make_train_subsets.py
└── docs/                           # 연구 로그
```

## 실행 환경 준비

작업공간 루트에서 실행합니다. Python 3.10 이상이 필요하며, 이번 subset 검증에는 Windows / Python 3.13.3 / CPU PyTorch 환경을 사용했습니다. `requirements.txt`는 보존된 소스의 직접 의존성 목록이며, 기존 Full 학습 환경을 복원하는 lockfile은 아닙니다.

Windows PowerShell에서 CPU 환경을 만드는 예:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

GPU 학습 시에는 CPU 설치 명령 대신 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 OS와 CUDA에 맞는 명령을 선택한 뒤 `requirements.txt`를 설치합니다. 아래 예시의 Python 경로도 사용 중인 환경에 맞게 바꿉니다.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`scikit-learn`은 HiLAM이 import하는 공통 학습 모듈에 필요합니다. `h5netcdf`와 `h5py`는 NetCDF 읽기에 사용하고, `matplotlib`은 선택적인 그래프 그림 생성에 사용합니다. `setproctitle`은 현재 서버 GPU 실행에서 필수이며 `scripts/requirements-server.txt`에 포함됩니다.

## 데이터 준비

대용량 원자료·예측 배열·체크포인트는 GitHub에 포함하지 않습니다. 저장소를 clone해도 NetCDF가 자동으로 내려오지 않습니다.

기존 최종 실험에서 사용한 아래 전처리 파일을 보유한 백업 또는 프로젝트 담당자에게 받아 `data/`에 넣습니다. 현재 확인된 공개 다운로드 주소나 이 전처리 파일 전체를 재생성하는 파이프라인은 저장소에 없습니다.

```text
data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc
```

| 항목 | 기준 |
|---|---|
| 파일 크기 | 1,300,083,528 bytes |
| 날짜 범위 | 1993-01-01~2022-12-31, 10,957일 |
| 격자 | latitude 29 × longitude 33 |
| SHA-256 | `fff22a44aeeb35fdd26f29b0d0ade7cf9549c18f5ae29aef2a415a5826708ec2` |

PowerShell에서 파일을 확인합니다.

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc"
```

기존 Full 모델을 다시 평가하려면 별도로 백업한 `*.pt`와 `*_test_fields.npz`도 `outputs/main_experiment_final/`의 원래 하위 경로에 복원해야 합니다. Git에는 설정·지표 JSON과 요약 CSV/Markdown을 남깁니다.

## Subset 확인

```powershell
.\.venv\Scripts\python.exe -B scripts/make_train_subsets.py --check
```

이 명령은 기존 `src/mhw/datasets.py`를 import하고, `outputs/main_experiment_final/experiment_manifest_final.json`을 읽어 날짜 분할·블록·nested·계절 균형·바이트 동일성을 검증합니다. 기본 코드와 데이터 위치는 현재 작업공간 기준으로 자동 결정됩니다.

JSON이 없는 경우에는 `--check` 없이 실행하여 생성할 수 있습니다. 설계 변경 후 기존 JSON을 갱신할 때만 `--overwrite`를 사용합니다.

WINDOW=44, 90일 블록, 4구간×회전 계절 round-robin을 사용합니다.
100/50/25/10%의 시작일 수는 3,995/1,974/987/376개입니다.
100%는 블록 내부 샘플 전체를 뜻하며, 기존 본 실험의 Full 결과와 구분합니다.

`scripts/train_data_efficiency.py`에서 v2 subset 연결을 완료했고, 후속 실행기는 이 연결 방식을 계승합니다.
`src/`의 모델·학습 코드는 내용을 변경하지 않고 이동했습니다.
기존 코드와 저장 결과의 `D:/MHW` 절대경로는 그대로이므로 학습·평가 실행 시 현재 위치에 맞는 경로 지정이 필요합니다.

## 기존 학습 진입점 확인

다음 명령은 옵션 도움말만 출력하며 학습이나 결과 저장을 시작하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -B src/train_jeju_hilam.py --help
```

기존 학습을 실행할 때는 `--data-path`와 `--output-root`를 현재 경로로 지정하고, 최종 실험과 동일하게 `--no-residual-feature`를 사용합니다. `--physics-mode none`은 baseline, `--physics-mode p0_original`은 P0 PINN입니다. 이 원본 진입점 자체에는 v2 subset 인자가 없으므로 기존 전체 학습 분할을 사용합니다. subset과 후속 실험에는 `scripts/`의 실행기를 사용합니다.

PowerShell 실행기는 별도의 Conda 환경을 전제로 합니다. 사용하려면 `-EnvName`, `-DataPath`, `-OutputRoot`를 지정하고, 이번 비교의 lead는 `-Leads "7,14"`로 한정합니다. 기존 Full 폴더를 덮어쓰지 않도록 새 출력 폴더를 사용합니다. 저장된 결과를 다시 집계하는 경우 JSON 안의 과거 절대경로도 확인해야 합니다.

## GitHub에 포함하는 파일

- 포함: `src/`, `scripts/`, `requirements.txt`, 문서, subset JSON 3개, 결과 설정·지표 JSON, 요약 CSV/Markdown.
- 제외: NetCDF, 체크포인트, 예측 배열, 실행 로그, 가상환경·캐시·로컬 비밀 설정, 로컬 파일 정리 기록.
- 현재 학습 출력인 `outputs/data_efficiency`는 심볼릭 링크를 포함해 업로드에서 제외하며, 공개할 작은 결과 표는 `docs/`에 스냅샷으로 보관합니다.
- `outputs/` 전체를 제외하지 않습니다. `experiment_manifest_final.json`은 subset 생성에 필요합니다.
- `.gitattributes`는 원본 소스·manifest·subset JSON의 줄바꿈 변환을 막아 SHA-256과 `--check` 재현성을 보존합니다.

업로드 전 Git 작업 사본에서 변경 파일을 확인합니다. 현재 서버의 실행 폴더와 Git 작업 사본은 별개일 수 있습니다.

```powershell
git status --short
git diff --stat
```

`.gitignore`는 로컬 파일을 삭제하지 않습니다. 이미 커밋된 파일에는 소급 적용되지 않으므로 기존 저장소에 이 설정을 추가할 경우 추적 상태도 확인해야 합니다.
