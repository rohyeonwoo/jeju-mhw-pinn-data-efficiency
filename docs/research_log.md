2026-09-23
- colab 환경 이용
- 학습 데이터 양에 따른 PINN 효과 비교 실험용 train subset 생성 (100/50/25/10%)
- 서브샘플링 seed 3개(42, 7, 123) 사용
- 최종 파일: train_subset_block_seed{42,7,123}.json

[변경점]
1. 날짜 단위 샘플링 → 90일 연속 블록 단위 샘플링으로 변경
: 날짜 단위로 뽑으면 30일 윈도우로 인해 10% subset도 고유 날짜의 97.8%를 사용하여 데이터 양이 실제로 줄지 않았음
: 블록 단위로 바꾼 뒤 고유 날짜 커버리지가 비율과 일치

2. 5년 구간 × 계절 round-robin 층화 추가
: 층화 전 10% subset에 검증 기간과 가장 가까운 시기인 2013~2017년이 누락됨

3. 90일 미만 자투리 블록(2017년 말 41일) 제외
: seed마다 샘플 수가 달라지는 문제를 제거하여 같은 비율이면 모든 seed 시작일 수가 동일함

4. 최종 subset
| 비율 | 블록 수 | 시작일 수 |
|------|---------|-----------|
| 100% | 101 | 6,161 |
| 50%  | 50  | 3,050 |
| 25%  | 25  | 1,525 |
| 10%  | 10  | 610   |

## 2026-09-24 — 기존 최종 분할 유지 및 win44 subset v2 재생성

- 2026-09-23의 subset 설계와 생성 결과는 폐기된 v1 기록이다. 아래 v2가 현재 기준이다.
- 변경 이유: 기존 최종 실험의 실제 분할은 날짜 고정 분할이 아니라 `mhw.datasets.make_time_splits`의 시간순 70/15/15였다. 실제 샘플 구간은 입력 history만이 아니라 history+lead였다. 기존 조건을 유지하기 위해 history=30, WINDOW=44(30+14)로 재생성했다.
- C 폴더 `MHW-20260923T120109Z-1-001 (2)/MHW/MHW/`의 원래 함수를 직접 import했다. 함수 재구현이나 기존 소스 수정은 하지 않았다. 데이터는 최종 실행기와 manifest의 `*glorys3d_budget2d.nc`를 복구된 로컬 위치에서 읽었다.
- 원래 학습 시작 인덱스: lead7은 0~7,643(7,644개), lead14는 0~7,638(7,639개). 교집합 후보는 0~7,638, 시작일 1993-01-01~2013-11-30이다.
- 원래 타깃 날짜 기준 분할:

| lead | 학습 | 검증 | 테스트 |
|------|------|------|--------|
| 7 | 1993-02-06~2014-01-10 | 2014-01-11~2018-07-06 | 2018-07-07~2022-12-31 |
| 14 | 1993-02-13~2014-01-12 | 2014-01-13~2018-07-07 | 2018-07-08~2022-12-31 |

- 블록 생성 날짜 범위는 두 lead의 학습 샘플 전체 사용 날짜 범위의 교집합인 1993-01-01~2014-01-10(7,680일)로 정했다. 후보 시작일 범위와 전체 사용 날짜 범위를 구분하며, 두 lead의 학습 범위를 모두 만족하도록 더 이른 끝 날짜를 사용했다.
- 90일 완전 블록 85개(1993-01-01~2013-12-11), 끝의 30일 제외. 블록당 44일 윈도우 시작일은 47개이며 모두 공통 후보에 속한다.
- 시간순 블록을 P1~P4에 22/21/21/21개 배분했다. 블록 중심 날짜의 계절별 블록과 계절 순서를 seed별로 섞어 구간 내 계절 round-robin 순위를 생성하고, 무작위로 정한 고정 구간 순서로 다시 round-robin했다. 비율별로 순위를 다시 만들지 않고 한 순위의 prefix를 사용했다.
- 목표 블록 수는 Python round의 ties-to-even 규칙을 사용한다. 50%의 42.5는 42, 10%의 8.5는 8이다.

| 비율 | 블록 수 | 시작일 수 | 고유 날짜 수 | 공통 학습 7,680일 대비 | 유지 블록 7,650일 대비 |
|------|---------|-----------|--------------|----------------------|---------------------|
| 100% | 85 | 3,995 | 7,650 | 99.6094% | 100.0000% |
| 50% | 42 | 1,974 | 3,780 | 49.2188% | 49.4118% |
| 25% | 21 | 987 | 1,890 | 24.6094% | 24.7059% |
| 10% | 8 | 376 | 720 | 9.3750% | 9.4118% |

- 생성 코드: `scripts/make_train_subsets.py`.
- 생성 결과: `data/subsets/train_subsets_v2_win44_seed{42,7,123}.json`.
- 각 비율에 selected_blocks/start_dates/start_indices와 구간×계절 교차표를 저장했다. 메타데이터에는 원래 분할, 단일 블록 순위, 데이터·분할 소스·실행기·manifest SHA-256을 기록했다.
- 검증: nested 블록/시작 인덱스, 44일 블록 내부 포함, 공통 학습 날짜 범위 포함, 양쪽 lead의 원래 train 인덱스 포함, 구간별 최대 1개 차이, 세 seed의 비율별 동일 샘플 수, 100% 동일 구성, 10%의 모든 구간 포함을 assert했다. 재실행 `--check`로 세 파일의 바이트 동일성도 검증했다.
- 사용자의 명시적 이동 지시에 따라 이전 workspace subset 3개를 `data/subsets/deprecated/`로, 층화가 없는 초기 노트북을 `notebooks/deprecated/`로 보관했다. 이동 전후 SHA-256이 동일함을 확인했다. 복구된 기존 코드 폴더 안의 구형 사본은 변경하지 않았으며 사용 대상이 아니다.
- 검증·테스트 분할, 모델 코드, 기존 학습 스크립트와 실행기는 유지했다. 데이터로더 연결, 학습량 기준, 정규화 통계 적용 방식은 다음 단계이며 아직 변경하지 않았다.
- 실행 환경은 임시 디렉터리의 별도 Python 환경을 사용했다. 기존 코드 import 시 bytecode 쓰기를 비활성화했다.
- C 폴더의 Python·PowerShell·shell·cmd 원본 127개 파일은 작업 전후 SHA-256이 모두 동일함을 확인했다.

## 2026-09-24 — 구간별 계절 순서 회전으로 10% subset 편향 수정

- 원인: 구간마다 계절 순서를 독립적으로 섞어 각 구간에서 두 블록만 선택되는 10%에서 특정 계절이 중복 우선 선택되었다. seed42·7의 기존 계절 분포는 겨울/봄/여름/가을 = 1/1/2/4였다.
- 수정: seed별 기본 계절 순서를 한 번만 무작위로 정한다. P1/P2/P3/P4는 이 순서를 0/1/2/3칸 왼쪽 회전시켜 시작한다. 각 계절 내부 블록 순서는 기존처럼 seed별 무작위로 섞고, 구간 round-robin과 하나의 전역 순위 prefix를 사용하는 nested 선택 방식은 유지한다. 기본 계절 순서는 JSON의 `_metadata.season_orders.P1`이며 나머지 구간 순서도 같은 형식으로 기록한다.
- `scripts/make_train_subsets.py --overwrite`로 기존 v2 JSON 3개를 재생성했다. 파일명·JSON 구조, WINDOW=44, 90일 블록, 4구간, 비율별 블록 수와 샘플 수, 고유 날짜 커버리지, 원래 분할과 데이터는 유지했다.
- 세 seed 모두 10%의 구간별 블록 수는 2/2/2/2이고, 계절별 블록 수도 2/2/2/2이다. 계절별 정확히 2개인지 확인하는 assert를 추가했다.

| seed | 비율 | P1/P2/P3/P4 블록 수 | 겨울/봄/여름/가을 블록 수 |
|------|------|--------------------|-------------------------|
| 42 | 100% | 22/21/21/21 | 21/21/22/21 |
| 42 | 50% | 10/11/10/11 | 11/11/10/10 |
| 42 | 25% | 5/6/5/5 | 5/5/5/6 |
| 42 | 10% | 2/2/2/2 | 2/2/2/2 |
| 7 | 100% | 22/21/21/21 | 21/21/22/21 |
| 7 | 50% | 11/10/11/10 | 11/10/10/11 |
| 7 | 25% | 5/5/6/5 | 5/5/6/5 |
| 7 | 10% | 2/2/2/2 | 2/2/2/2 |
| 123 | 100% | 22/21/21/21 | 21/21/22/21 |
| 123 | 50% | 10/10/11/11 | 10/10/11/11 |
| 123 | 25% | 5/5/6/5 | 6/5/5/5 |
| 123 | 10% | 2/2/2/2 | 2/2/2/2 |

- 검증 완료: 기존의 nested, 블록·학습 범위 내부 44일 포함, 양쪽 lead train 인덱스 포함, 구간별 균등 배분, seed별 동일 샘플 수, 100% 동일 구성 및 10% 모든 구간 포함 assert를 전부 다시 통과했다. 추가한 10% 계절 균등 assert도 통과했다.
- `--check` 재실행으로 세 파일의 바이트 동일성을 확인했다. 수정 전후 100% subset 전체 내용의 SHA-256, 고정 메타데이터와 비율별 샘플 수를 비교하여 동일함을 확인했다.
- Full 기준점: 100% subset은 3,995개로, 원래 전체 학습 샘플 7,639개(lead14 및 두 lead 공통 후보)보다 적다. 90일 블록 내부에 완전히 들어가는 44일 윈도우만 유지했기 때문이다. 따라서 이번 블록 subset의 100%와 원래 전체 학습 데이터의 Full을 구분하고, 기존 본 실험 결과(`outputs/main_experiment_final/`)를 Full 기준점으로 사용한다. 기존 Full의 lead7 학습 샘플은 7,644개이며, 기존 결과의 이 조건은 변경하지 않는다.
- `AGENTS.md`에도 현재 계절 순환 방식과 Full 기준점 구분을 반영했다. 모델·데이터로더·기존 학습 코드·실행기는 수정하지 않았다.

## 2026-09-24 — 추가 실험에 필요한 파일만 남기는 정리

- 사용자가 백업 완료를 알렸으며, 보존·삭제 목록을 검토한 뒤 삭제를 명시적으로 승인했다.
- `docs/cleanup_manifest.json`에 지정한 파일 4,113개, 21,703,372,796 bytes(20.213 GiB)를 삭제했다. 해당 파일들의 상위 경로 중 비어 있는 폴더 1,234개도 제거했다. 다른 파일을 추가로 탐색하여 삭제하지 않았다.
- 보존: 최종 budget2d 데이터 1개, v2 subset 3개, 생성 스크립트, 필수 학습·평가 코드와 공통 모듈 22개, 기존 Full 결과 `outputs/main_experiment_final/` 전체, AGENTS.md와 연구 로그. 총 380개 원래 파일, 약 2.949 GiB를 유지했다. 정리 목록·실행 기록은 별도로 남겼다.
- 삭제 전 각 파일의 경로가 작업공간 안인지, reparse point가 아닌지, 승인 목록의 크기·수정 시각과 같은지 확인했다. 보존/삭제 집합이 겹치지 않는지도 확인했다.
- 삭제 전후 보존 파일 380개의 SHA-256이 전부 동일했다. 이 확인 이후 AGENTS.md와 이 연구 로그만 현재 정리 상태에 맞게 갱신했다. 데이터·모델·데이터로더·학습 코드·실행기·Full 결과의 내용은 변경하지 않았다.
- 삭제 후 `scripts/make_train_subsets.py --check`를 실행해 기존 검증 assert, 10% 계절별 2블록 assert, 세 JSON의 바이트 동일성이 모두 통과했다. 승인 목록의 삭제 대상이 모두 사라진 것도 확인했다.
- 구형 subset과 초기 노트북을 담았던 deprecated 폴더도 삭제 대상에 포함되었다. 앞선 로그의 보관 경로는 당시의 역사적 기록이며, 현재는 존재하지 않는다.
- 보존 코드의 로컬 import 의존 관계를 삭제 계획 단계에서 확인했다. 기존 코드·결과에 남은 D:/MHW 절대경로는 변경하지 않았으며, 실행 환경에 맞춘 경로 지정은 여전히 필요하다.

## 2026-09-24 — 최상위 폴더 구조 정리

- 사용자가 제시된 이동 목록을 승인했다. 중첩된 `MHW-20260923T120109Z-1-001 (2)/MHW/` 아래의 주 코드 `MHW/`를 작업공간 `src/`로, 최종 전처리 NetCDF를 작업공간 `data/`로, 기존 Full 결과 `outputs/`를 작업공간 `outputs/`로 옮겼다.
- 코드 22개, 데이터 1개, 기존 Full 결과 351개 등 총 374개 파일을 이동했다. 이동 전후 SHA-256이 모두 동일했으며, 이동 후 빈 상위 폴더 3개를 제거했다. 원본 모델·학습·평가 코드와 결과 내용은 변경하지 않았다.
- 데이터는 `data/`에 모았고, v2 subset은 기존 `data/subsets/`에 그대로 두었다. 모델 정의는 `src/mhw/models/`이며 기존 Python 패키지·import 구조를 유지했다.
- 함께 이동된 과거 출력 경로 `outputs/inverse_feature_multilead/lead_3_history_21/`와 그 부모가 파일 없이 비어 있음을 확인해 제거했다. 현재 outputs에는 main_experiment_final만 남아 있다.
- `scripts/make_train_subsets.py`의 기본 코드 위치를 `src/`로 바꾸고, 새 위치에서 v2 JSON을 재생성했다. JSON에서는 `_metadata.source.code_folder`와 `resolved_data_path` 두 경로 필드만 바뀌었다. 두 필드를 제외한 전체 내용 해시가 이동 전과 동일하여 블록 선택·시작일·인덱스·통계·원본 해시가 모두 유지되었음을 확인했다.
- 새 위치에서 `--check`가 통과했다. 기존 모든 assert, 10% 계절별 2블록, 세 파일의 재생성 바이트 동일성을 확인했다.
- `AGENTS.md`의 현재 경로를 갱신하고 최상위 `README.md`에 폴더 안내를 추가했다. 파일별 이전·현재 경로와 해시는 `docs/layout_migration.json`에 기록했다. 이전 로그와 cleanup 기록의 경로는 당시 위치를 뜻한다.
- 원본 코드와 기존 Full 결과 내부의 `D:/MHW` 절대경로는 유지했다. 현재 환경에서 학습·평가 시 경로 지정이 필요하며, 이번 작업은 폴더 이동과 subset 생성기의 경로 연결까지만 수행했다.

## 2026-09-24 — GitHub 업로드 준비

- 사용자의 요청으로 `.gitignore`, `.gitattributes`, `requirements.txt`를 만들고 `README.md`에 환경 설치, 데이터 준비, subset 검증 및 기존 학습 진입점 안내를 추가했다.
- 대용량 NetCDF, 체크포인트, 예측 배열, 실행 로그, 가상환경·캐시·로컬 비밀 설정 및 로컬 정리 기록은 Git에서 제외한다. 코드, subset JSON 3개, 기존 Full manifest, 설정·지표 JSON과 요약 CSV/Markdown은 포함한다. 특히 Full manifest는 subset 생성에 필요하므로 outputs 전체를 제외하지 않는다.
- subset 메타데이터가 원본 소스와 manifest의 SHA-256을 저장하고 `--check`가 JSON 바이트를 비교하므로, `.gitattributes`로 해당 파일의 줄바꿈 변환을 막았다.
- `requirements.txt`는 보존 소스의 직접 의존성에 버전 범위를 지정한 설치 명세다. 기존 Full 실험의 환경 lockfile은 아니며, GPU 환경은 PyTorch 공식 설치 안내에 따라 선택하도록 했다. `scikit-learn`은 HiLAM이 가져오는 공통 학습 모듈의 import에 필요하다.
- 데이터는 기존 백업 또는 프로젝트 담당자에게 받아 준비하도록 안내하고 정확한 파일명·크기·날짜·격자·SHA-256을 README에 기록했다. 공개 다운로드 주소 및 전체 전처리 재생성 파이프라인은 확인되지 않아 제공하지 않았다.
- 검증 환경: Windows, Python 3.13.3, numpy 2.2.6, pandas 2.2.3, torch 2.14.0+cpu, xarray 2026.7.0, h5py 3.14.0, h5netcdf 1.8.1, scikit-learn 1.9.1, matplotlib 3.10.3. 임시 가상환경에서 의존성 버전 조건, `src/train_jeju_hilam.py --help` 실행 및 `scripts/make_train_subsets.py --check`를 확인했다. 세 seed의 모든 assert와 재생성 바이트 동일성이 통과했다. 학습이나 GPU 검증은 실행하지 않았다.
- Git 실행 파일이 확인되지 않아 실제 `git add --dry-run` 대신 pathspec으로 ignore 규칙을 검증했다. 대용량 파일이 제외되고 코드·subset·manifest가 포함됨을 확인했으며, 포함 대상은 약 3.3 MiB이고 100 MiB를 초과하는 개별 파일은 없었다. 원본 분할 코드·실행기·manifest의 해시도 subset 메타데이터와 일치했다.
- 이번 작업에서 기존 소스·데이터·모델·Full 결과를 수정하거나 삭제·이동하지 않았다. 현재 없는 과거 정리 기록을 재생성하지 않았고, Git 초기화·커밋·원격 업로드도 수행하지 않았다. v2 subset의 학습 데이터로더 연결은 여전히 다음 단계다.

## 2026-09-24 ? ?? ?? ?? ?? ??

- ???? ??? `/ceph_data/yeonwoo` ?? `MHW_subset` ??? ????? ????. ?? ??? `203.253.207.61:20001`, ??? `yeonwoo`??.
- ??, ?? ???, v2 subset, ??? Full ?????? ?? ???? ???. ?? Full ???????? ?????? ? ??? ????? ?? ?? ???? ????. ?? ??? ????.
- `docs/server_transfer.md`? ????? ?? ??? ?? ??? subset ?? ??? ????. ?? ?? ? ??? ???? ???.


## 2026-09-24 — 서버 실행 위치·환경 준비와 실험 방침 재확정

- 사용자 최신 지시에 따라 실행 작업공간을 `/home/yeonwoo/mhw_subset` (`~/mhw_subset`)으로 정했다. `/ceph_data/yeonwoo/MHW_subset` 원본은 보존하고 빈 실행 폴더에 사본을 만들었다. 원본 242개 파일의 SHA-256이 사본과 모두 일치했다. 기준 해시는 `docs/server_preflight/original_snapshot_sha256.json`에 저장했다.
- 사용자 메시지의 미결정 표기와 `docs/server_codex_handoff.md` 7절 확정 표기가 달라 확인했고, 사용자가 '문서 7절 방침으로 확정'을 선택했다. 같은 최대 step 예산+동일 early stopping, lead별 원래 Full 정규화·RHS 통계 공통 적용, subset seed=학습 seed(42/7/123)를 확정한다.
- batch size=8, lead7 S=956 / 최대 76,480 step / warmup 4,780 step; lead14 S=955 / 최대 76,400 step / warmup 4,775 step. S마다 전체 validation, val_rmse best 및 12회 연속 미개선 종료. virtual_epoch에 원본 LR·physics warmup 함수를 적용하며 optimizer와 진행 중 subset iterator를 검증 경계에서 초기화하지 않는다.
- 통계는 100% subset이 아닌 각 lead의 원래 Full 학습 분할 기준이다. 제외된 날짜와 관측 타깃도 정규화·RHS 통계에 포함된다. 따라서 연구 해석은 'Full 기준 통계를 유지하면서 gradient 학습 샘플을 줄인 실험'이다. 전체 정보 접근량까지 제한한 실험이라고 해석하지 않는다.
- 100% subset 3,995개와 기존 Full(lead7 7,644개, lead14 7,639개)을 구분한다. 같은 subset/학습 seed의 baseline/P0를 짝지어 비교하며 seed 간 변동에는 subset 선택과 학습 난수가 함께 들어간다. 100%도 학습 seed 3회, 전체 48회 구성이다.
- 사용자 요청대로 `mhw_subset` conda 환경을 새로 생성했다. 서버 기존 conda envs_dirs에 따라 실제 위치는 `/ceph_data/yeonwoo/conda_envs/mhw_subset`이다. Python 3.11.16, PyTorch 2.5.1+cu121 및 requirements.txt 의존성을 설치했고 pip check는 통과했다. 원래 Full 환경 lockfile이 없으므로 Full 환경과 동일하다고 주장하지 않는다. 설치 로그·freeze·conda explicit 기록은 `docs/server_preflight/`에 저장한다.
- 서버 node1에는 RTX A6000 48GB GPU 10장이 있고 확인 당시 모두 유휴 상태였다. driver 535.230.02, 시스템 메모리 약 1TiB. 이번 준비 단계의 CUDA·RHS 계산 점검만 GPU 0에서 수행한다.
- 코드 조사: 원본 Full bundle을 만든 후 Full RHS 계산까지 마치고 train에만 별도 Subset view를 적용할 수 있다. frozen bundle을 수정하지 않는다. 원본 main의 subset-epoch 검증 일정은 확정한 S-step 일정과 달라 별도 바깥 실행기가 필요하지만 모델·물리·분할 코드는 그대로 import 가능하다.
- 기존 run_epoch의 max_batches는 batch를 꺼낸 후 제한을 검사하므로 연속 iterator에서 추가 batch 하나를 소비할 수 있다. 새 실행기는 정확히 S개를 내는 유한 iterable을 주고 max_batches=None을 써야 한다. 실제 추가 학습 실행기는 아직 구현하지 않았다.
- `scripts/audit_server_preparation.py`와 `docs/server_integration_review.md`를 추가했다. 현재 subset 재현성과 실제 Full 통계·RHS·val/test 보존을 검증 중이며 결과는 다음 항목에 기록한다. 원본 src, subset, 데이터, Full 결과의 수정·삭제·이동 및 본 학습은 하지 않았다.


## 2026-09-24 — 서버 준비 검증 완료 (본 학습 미실행)

- 새 `mhw_subset` 환경에서 pip check와 원본 `src/train_jeju_hilam.py --help`가 성공했다. CUDA 12.1 tensor 연산·autograd 점검이 GPU 0에서 통과했다. GPU 10장이 인식된다.
- `python -B scripts/make_train_subsets.py --check`가 성공했다. seed 42/7/123 JSON 모두 바이트 단위 재현 일치, nested·Full train 소속·44일 블록 내부 포함·구간 균형·10% 계절별 2블록 assert 통과. 기존 JSON을 덮어쓰지 않았다.
- 원본 bundle을 사용하는 추가 점검에서 12개 HiLAM Full 설정(모델 2×lead 2×seed 3)을 대조했다. 두 lead × 세 seed × 네 비율의 실제 Subset view를 만들고 시작/중간/끝 sample의 current/target 인덱스, 공통 Full stats 참조, 원래 val/test 인덱스 불변을 확인했다. 두 lead가 같은 시작 인덱스를 사용하는 것도 확인했다.
- lead7: Full train/val/test=7644/1638/1639, RHS=9.73672297779631; 저장 Full 타깃 평균·표준편차와의 최대 절대차 1.14440918e-05 K. RHS는 저장값과 동일했고 타깃 통계는 rtol=1e-6, atol=1e-6 범위에서 일치했다.
- lead14: Full train/val/test=7639/1637/1638, RHS=9.72798431889206; 저장 Full 타깃 평균·표준편차와의 최대 절대차 5.7220459e-06 K. RHS는 저장값과 동일했고 타깃 통계는 rtol=1e-6, atol=1e-6 범위에서 일치했다.
- 타깃 통계는 새 Linux 의존성 환경에서의 미세한 부동소수점 차이가 있어 비트 단위 동일하다고 보고하지 않는다. 원래 함수를 사용해 계산한 이번 환경의 Full 통계를 lead별 공통 기준으로 확보했다. Full 환경 lockfile은 없어 과거 환경 자체의 재현을 검증한 것은 아니다.
- 정규화 통계 날짜 끝은 lead7 2014-01-10, lead14 2014-01-12이며 기존 ACC/MHW reference는 2013-12-31까지다. 각 실제 원본 계산 범위를 그대로 유지하고 서로 같다고 가정하지 않는다.
- 점검 전후 보호 대상 233개 파일(src, 데이터·subset, 기존 Full, 원본 subset 생성기)의 SHA-256이 모두 동일했다. 보고서 `docs/server_preflight/audit_report.json` status=passed. 두 lead의 통계와 RHS는 `full_statistics_lead7.json`, `full_statistics_lead14.json`에 저장했다.
- 이번 완료 범위는 conda/GPU 환경, subset 재현성, Full 통계·인덱스 연결 가능성 및 원본 보존 검증이다. 별도 S-step 학습 실행기 구현·일정 경계 테스트·모델 학습 smoke test·48회 본 학습은 아직 수행하지 않았다. 이후 실행기에서 이번 Full 통계 cache의 데이터·소스·설정 해시를 확인하고 재사용할 수 있다.


## 2026-09-24 — GPU 프로세스 이름 yeonwoo 강제

- 사용자 서버 규칙에 따라 실행 계정뿐 아니라 nvidia-smi Process name을 `yeonwoo`로 지정한다. 실제 실행 계정은 이미 yeonwoo였다. PID는 숫자이며 이름 칸에 yeonwoo가 표시되는 것을 확인했다.
- 기존 HiLAM 진입점은 원본 process_title 함수를 호출하지 않아 환경변수만으로 보장되지 않는다. 원본 src는 유지하고 `scripts/run_as_yeonwoo.py`를 추가했다. GPU 모듈 import 전에 setproctitle을 적용하며 패키지가 없거나 계정이 다르면 중단한다.
- conda mhw_subset에 setproctitle 1.3.7을 설치했고 추가 의존성은 `scripts/requirements-server.txt`, 전체 설치 버전은 `docs/server_preflight/requirements.freeze.txt`에 기록했다.
- 새 준비 점검 스크립트에도 GPU import 전 이름 지정을 추가했다. 앞으로 GPU 작업은 named wrapper로 실행하도록 실행 작업공간 AGENTS.md에 기록했다.
- 실제 CUDA 할당 후 nvidia-smi 조회 성공: PID 1348706, Process name=yeonwoo, GPU memory 306 MiB. 확인용 프로세스는 종료했고 학습하지 않았다. 증거: `docs/server_preflight/gpu_process_name_check.json`.
- 래퍼를 통한 원본 HiLAM --help 및 pip check도 성공했다. 기존 모델·학습 src와 Full 결과는 변경하지 않았다.


## 2026-09-24 — 로컬 대용량 사본 제거 및 Ceph 링크 연결

- 사용자가 `~/mhw_subset`에 데이터셋 같은 대용량 파일을 두지 말라고 정정했다. 앞선 실행용 사본 생성 시 NetCDF까지 로컬로 복사한 것은 잘못된 저장 배치였으며 이를 수정했다.
- 대상은 `data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc` 1개, 1,300,083,528 bytes다. 변경 대상을 먼저 알리고 로컬·Ceph 원본 SHA-256이 모두 기준 해시와 같은지 확인한 후 로컬 중복 실체를 Ceph 원본 심볼릭 링크로 교체했다. Ceph 원본은 수정하지 않았다.
- 로컬 `outputs/data_efficiency`도 Ceph의 `/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency`로 연결했다. 향후 체크포인트·예측 배열·대용량 로그와 smoke 출력은 Ceph에 둔다. conda 환경은 이미 Ceph에 있어 그대로 유지한다.
- 모델·분할 코드·subset JSON·기존 Full 결과는 그대로다. 기존 생성기가 symlink를 resolve한 절대경로를 기록하는 특성 때문에 재현성 검사는 Ceph 원본 생성기에서 수행한다. 로컬 subset 3개와 분할 소스·실행기·manifest·생성기 자체는 Ceph 원본과 바이트가 같은지 별도로 확인했다. JSON 경로 메타데이터를 임의로 변경하지 않았다.
- 경로 기록: `docs/server_preflight/storage_layout.json`. 이번 사용자 규칙을 로컬 AGENTS.md와 서버 조사 문서에 반영했다.

- 저장 위치 수정 후 확인 완료: 로컬 사용량 4.4MB, 50MiB 초과 일반 파일 0개, Ceph에서 subset 3개 바이트 재현성 재검증 통과.


## 2026-09-24 — GPU 3개를 이용한 48회 본 학습 실행

- 사용자가 학습 시작과 GPU 2~3개 사용을 명시적으로 요청했다. 확인 당시 비어 있던 GPU 0/1/2를 선정했다. 조건을 바꾸지 않고 독립 실행을 GPU당 1개씩, 최대 3개 동시에 처리하는 동적 큐로 구성했다.
- 추가 실행기 `scripts/train_data_efficiency.py`는 원본 모델·Full bundle·make_loader·run_epoch·LR·physics warmup·최종 예측/평가 함수를 import한다. 원본 src는 수정하지 않았다. Full bundle의 모든 통계와 val/test 인덱스 해시를 기존 감사 결과와 비교하고 Full RHS cache의 데이터·소스 해시를 확인한 뒤 train에만 Subset을 적용한다.
- 검증 경계에서 유지되는 BatchStream이 정확히 S개 batch를 전달한다. 기존 run_epoch에 max_batches=None을 주어 추가 batch 소비를 막고, subset 소진 시에만 다시 shuffle한다. 작은 마지막 batch, 기존 AdamW 상태와 fp32/batch8 설정을 유지한다. virtual_epoch에 원본 LR와 physics warmup 함수를 전달하고 S마다 전체 validation을 평가한다. val_rmse best checkpoint와 12회 미개선 종료, 최대 80*S는 확정 방침 그대로다.
- `scripts/test_data_efficiency_schedule.py` 검증 통과: 독립적인 일반 DataLoader 반복과 shuffle/작은 마지막 batch/검증 경계 전후 순서 동일, 원본 run_epoch의 optimizer update와 분할 호출 결과 정확히 동일, lead7/14의 S 및 5*S/80*S LR·physics 경계 확인.
- baseline/P0 × lead7/14 네 smoke 실행이 각각 4 update와 두 검증을 완료했다. 체크포인트 저장·재로딩·test 예측 저장 및 타깃 인덱스 대조가 성공했다. 원본 MHW 평가기도 smoke 예측 4개를 정상 처리했다. 짧은 smoke 지표는 본 실험 성능이 아니며 `_validation/`에 분리했다. 증거: `docs/server_preflight/launch_validation.json`.
- `scripts/launch_data_efficiency.py`는 48개(4비율×3seed×2모델×2lead)를 배정한다. GPU가 끝나면 다음 작업을 시작하며 실패 시 새 작업 배정을 중지하고 이미 실행 중인 작업은 마무리한다. 모든 학습 종료 후 비율별 원본 MHW 평가기를 자동 실행한다.
- 모든 CUDA 프로세스는 setproctitle로 yeonwoo를 강제하고 실제 자신의 PID가 nvidia-smi에 yeonwoo로 나타나는지 학습 시작 전에 assert한다. subset seed=training seed이며 각 subprocess의 PYTHONHASHSEED도 설정한다.
- tmux 세션 `mhw_subset`에서 큐를 시작했다. 모든 checkpoint/예측/학습로그 실체는 `/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/` 아래에 저장한다. 로컬 `~/mhw_subset`은 코드·문서·작은 메타데이터만 유지한다.
- 큐 상태: `outputs/data_efficiency/_queue/batch_status.json`; 개별 진행: `ratio_{비율}/{jeju_hilam,pi_jeju_hilam}/lead{lead}_seed{seed}/progress.json`. 로그는 `_queue/job_*.log`, 검증 이력은 개별 `jeju_hilam_epoch_history.csv`에 지속 저장한다. best/실제 step, 검증 횟수, 통계·인덱스·해시·환경을 기록한다.
- 이번 기록은 실행 시작이며 48회 완료 보고가 아니다. 원본 모델·분할·subset·기존 Full 결과의 해시 보존을 실행 전 확인했다.

- 실행 후 확인: GPU 0/1/2의 첫 세 본 학습이 모두 25 step 이상 진행했으며 PID는 1365327, 1365329, 1365330이다. nvidia-smi에서 모두 yeonwoo로 표시됨을 확인했다.


## 2026-09-24 — 사용자 지정 GPU 0의 5개 / GPU 1의 4개 동시 실행으로 재시작

- 사용자가 기존 작업을 멈추고 GPU 0에 10% 3개 및 25% 2개, GPU 1에 25% 1개 및 50% 3개를 배정하도록 명시했다. 개별 실행은 subset/training seed로 해석했고 사용자가 확인했다. 추가로 100%도 빈 슬롯에서 이어서 실행하도록 명시했다.
- 기존 큐 관리자 PID 1365302와 학습 PID 1365327/1365329/1365330만 종료했다. 당시 약 4천 step까지 진행한 기존 출력은 그대로 보존하고 상태를 stopped_by_user로 기록했다. 저장된 best checkpoint에는 optimizer 및 iterator/RNG 진행 상태가 없으므로 정확한 resume를 하지 않고 새 출력 경로에서 동일 seed로 처음부터 시작한다.
- 학습 실행기와 원본 src, batch size=8, fp32, Full 통계·RHS, step/warmup/early stopping은 변경하지 않았다. 데이터 비율은 저장 샘플 수를 바꾸지만 GPU에 한 번에 올리는 batch와 모델 크기는 같으므로 GPU 메모리는 거의 같은 범위다. 실제 기존 baseline 메모리는 약 4.4GiB/프로세스였다.
- 새 GPU 배치: GPU 0의 슬롯 5개 = 10% seed42/7/123, 25% seed42/7; GPU 1의 슬롯 4개 = 25% seed123, 50% seed42/7/123. GPU 2는 사용하지 않는다.
- 각 슬롯은 해당 비율·seed의 lead7 baseline → lead7 P0 → lead14 baseline → lead14 P0 순서로 진행한다. 해당 네 작업을 마친 슬롯은 공용 100% 큐(3seed×2모델×2lead=12개)를 가져간다. 전체 48개 구성 유지, 동시에 최대 9개 학습이다.
- 새 관리기: `scripts/launch_fixed_allocation.py`. 지정 배치와 Cartesian product 48개에 누락·중복이 없음을 검사했다. 계획 검증 기록은 `docs/server_preflight/fixed_allocation_plan.json`에 저장했다. 기존 학습 실행기 내용은 그대로다.
- 새 tmux 세션: `mhw_subset_5plus4`. 새 결과 실체: `/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/allocation_5plus4/`. 현재 시도 포인터는 상위 `current_attempt.json`, 실제 큐 상태는 새 경로의 `_queue/batch_status.json`이다. 과거 시도의 지표와 섞지 않는다.
- 모든 GPU 프로세스 이름은 yeonwoo, 모든 대용량 출력은 Ceph. 큐는 한 작업이 실패하면 새 배정을 중지하고 현재 실행 중인 작업은 마무리한다. 전체 학습 종료 후 비율별 기존 MHW 평가기를 자동 실행한다.
- 이 기록은 재시작 기록이며 48개 실험 완료를 의미하지 않는다.


## 2026-09-24 — lead/P0별 메모리 확인

- 사용자 메모리 질문에 따라 GPU 1에서 별도 임시 모델로 네 조합을 각각 3-step 측정했다. batch8/history30/fp32 및 원본 모델·손실을 사용했고 본 학습 프로세스·체크포인트는 수정하지 않았다. 모델과 GPU 메모리는 종료 시 해제했다.
- 결과: [{"lead": 7, "mode": "none", "batch_size": 8, "history": 30, "steps": 3, "peak_allocated_mib": 3371.82177734375, "peak_reserved_mib": 4112.0}, {"lead": 7, "mode": "p0_original", "batch_size": 8, "history": 30, "steps": 3, "peak_allocated_mib": 3375.7236328125, "peak_reserved_mib": 4124.0}, {"lead": 14, "mode": "none", "batch_size": 8, "history": 30, "steps": 3, "peak_allocated_mib": 3373.99169921875, "peak_reserved_mib": 4120.0}, {"lead": 14, "mode": "p0_original", "batch_size": 8, "history": 30, "steps": 3, "peak_allocated_mib": 3375.5517578125, "peak_reserved_mib": 4124.0}]
- 측정값은 PyTorch allocated/reserved peak이며 nvidia-smi의 CUDA context 포함 전체 메모리와 다르다. 3-step 측정으로 모든 학습 단계의 절대 최대를 보장하지 않는다.


## 2026-09-24 — 사용자 확정: batch16, GPU 0/1 각각 5개로 재시작

- 사용자가 앞선 9+9 증설을 취소하고 batch size 16 및 GPU 0/1에 5개씩 동시 실행을 지시하며 현재 학습 종료를 요청했다. 9+9 증설은 실행되지 않았다. 5+4 큐 관리자 PID 1381072 및 학습 PID 1381183/1381190/1381191/1381193/1381194/1381199/1381200/1381201/1381202를 종료했고 기존 출력은 stopped_by_user 상태로 보존했다.
- batch size는 모든 새 subset 실험에서 16으로 통일한다. history30, lead7/14, fp32, 모델·물리식·입력·원래 분할·val/test, Full 통계 및 기존 train-fixed RHS cache는 유지한다. LR=2e-4/initial2e-5, AdamW, lambda_physics=0.1 등 다른 조건은 변경하지 않는다.
- Full 기준 구간의 정의 S=ceil(원래 Full train 샘플 수/current batch size)를 유지한다. lead7 ceil(7644/16)=478, lead14 ceil(7639/16)=478. 두 lead 모두 최대 80*S=38,240 step, LR·physics warmup 5*S=2,390 step, 478 step마다 전체 validation, val_rmse 기준 12회 연속 미개선 종료다. 기존 audit cache의 batch8 S는 과거 기준값으로 보존하고 새 실행 config에 실제 batch16 S를 기록한다.
- 기존 Full 결과는 batch8 실험이다. 새 subset 결과는 batch16이므로 기존 Full과의 차이를 데이터 양만의 효과라고 단정할 수 없다. Full을 재학습·수정하지 않으며 별도 기준점과 batch 설정 차이를 명시한다. 새 48개 subset 실험 내부의 baseline/P0 및 비율 비교는 batch16으로 통일한다.
- 새 학습 실행기에 --batch-size {8,16} 인자를 추가했다. 이전 batch8 실행기 사본은 Ceph allocation_5plus4/_queue/train_data_efficiency_batch8_snapshot.py에 보존했다. 원본 src 모델·학습·분할 코드는 수정하지 않았다.
- 검증: batch8/16 양쪽 lead의 일정 경계 검사 통과. batch16 baseline/P0 × lead7/14 네 smoke가 각각 4 update, 검증, checkpoint 저장·재로딩, 예측 저장을 완료했다. 각 config의 S=478/max_steps=38240/warmup_steps=2390 확인. 증거: docs/server_preflight/batch16_launch_validation.json.
- GPU 0에서 batch16 및 마지막 작은 batch 크기 8/11/6까지 테스트한 결과 peak allocated는 약 6,538~6,541 MiB, peak reserved는 8,034~8,048 MiB였다. CUDA context 여유 512 MiB/프로세스를 포함한 5개 추정치는 42,800 MiB(약 41.8 GiB)로 48GB GPU 용량 안이다. 장시간 전체 단계의 절대 상한을 보장하는 값은 아니다. 메모리 보고서: docs/server_preflight/batch16_memory.json.
- GPU 0의 5개 슬롯 = 10% seed42/7/123 및 25% seed42/7. GPU 1의 5개 슬롯 = 25% seed123, 50% seed42/7/123, 추가로 100% seed42. 각 슬롯은 lead7 baseline→lead7 P0→lead14 baseline→lead14 P0를 처리하고, 빈 슬롯에서 나머지 100% seed7/123의 8개 실행을 가져간다. 총 48개 유지, 최대 10개 동시 학습이다.
- 새 관리자 scripts/launch_batch16_5plus5.py, tmux mhw_subset_b16_5plus5, 새 Ceph 출력 경로 outputs/data_efficiency/allocation_batch16_5plus5/. current_attempt.json이 현재 시도를 가리킨다. 기존 중단 결과와 섞거나 덮어쓰지 않는다. 모델 optimizer/iterator 상태가 checkpoint에 없어 동일 seed로 처음부터 실행한다.
- 모든 GPU process name은 yeonwoo, 모든 대용량 출력은 Ceph다. 이 기록은 새 실행 시작이며 48개 완료 보고가 아니다.


## 2026-09-24 — batch16 Full 기준점 12회 후속 학습 예약

- 사용자가 현재 batch16 subset 실험 완료 후 Full을 batch16으로 다시 학습하는 방법을 제안했다. 기존 batch8 Full을 보존한 채 새 batch16 Full 기준점을 추가한다. 현재 subset 학습을 중단·재시작하지 않는다.
- 새 Full은 원래 각 lead의 전체 train: lead7 7,644개, lead14 7,639개다. 블록 100% subset 3,995개와 다르며 Full의 lead7 샘플을 공통 교집합 7,639개로 줄이지 않는다.
- Full 새 실행도 3 학습 seed(42/7/123)×2 lead(7/14)×2 모델(baseline/P0)=12개다. 기존 subset 48개와 합쳐 batch16 본 학습 총 60개다. Full에는 subset seed/ratio를 적용하지 않고 training seed만 사용한다.
- Full과 subset 모두 batch16, history30, fp32, 같은 모델·물리식·Full 정규화 및 RHS cache·학습 하이퍼파라미터·val/test·평가 정의를 사용한다. 각 lead S=478, 최대 38,240 step, warmup 2,390 step, validation 478 step마다, val_rmse best 및 12회 미개선 종료를 맞춘다. 실제 종료 step은 실행별로 다를 수 있다.
- 공통 추가 실행기에 --full과 --ratio를 상호 배타적으로 추가했다. Full은 원본 bundle.train 자체를 선택한다. CPU에서 두 lead의 Full 수·경계 target 인덱스·val/test 보존과 3seed×4비율의 기존 subset 선택이 그대로인지 확인했다. 일정 단위 검사도 통과했다. 증거: docs/server_preflight/full_batch16_selection_check.json. 기존 batch16 subset 실행기 사본은 현재 subset 큐 폴더에 보존했다. 원본 src 모델·분할·학습 코드는 수정하지 않았다.
- 후속 관리자 scripts/run_full_batch16_after_subset.py를 tmux mhw_subset_full_b16_followup에서 대기 상태로 시작했다. 현재 48개 학습과 원본 MHW 평가까지 완료된 뒤에만 Full 네 조합의 짧은 GPU smoke를 하고, 통과하면 GPU0/1 각각 최대 5개로 Full 12개를 실행한다. 대기 중 GPU를 점유하지 않는다.
- 의존 subset 큐가 중단/실패하면 Full을 시작하지 않는다. 예약 후 학습 실행기 파일이 변경되어도 검토 없이 자동 실행하지 않는다. 새 Full 출력은 Ceph outputs/data_efficiency/full_batch16_matched/full/ 아래이며 기존 outputs/main_experiment_final은 보존한다.
- 후속 상태: outputs/data_efficiency/full_batch16_matched/_queue/batch_status.json. 전체 계획: outputs/data_efficiency/batch16_study_plan.json. 연결 포인터: outputs/data_efficiency/full_followup.json. 기존 subset current_attempt.json은 유지하다가 Full 실제 실행 단계에서 갱신한다.
- 이 방식은 진행 중인 subset 결과를 버리지 않고 batch가 일치하는 Full 비교점을 만든다. 총 소요시간이 batch8 방식보다 반드시 짧다고 보장하지 않으며 Full 12회의 추가 비용은 든다. 이번 기록은 후속 예약 완료로, Full 학습 완료 보고가 아니다.


## 2026-09-25 — batch16 본 학습 60회 완료 및 결과 분석

- 사용자 요청에 따라 baseline/P0 × lead7/14 × 3seed의 subset 48회와 별도 Full 12회를 비교했다. 두 큐 모두 completed이고 60개 모두 종료 코드 0이다. 중단된 batch8 시도, smoke, 기존 Full batch8은 이번 비교에서 제외했다. 새 학습·모델 변경은 하지 않았다.
- scripts/analyze_batch16_results.py를 추가했다. 원본 출력은 읽기만 하며 작은 집계 CSV·Markdown·PNG/PDF와 분석 메타데이터는 docs/batch16_results/에 저장했다. 데이터·체크포인트·예측 배열은 Ceph에 그대로 유지했다.
- 기록된 batch16/S478/최대38240/공통 하이퍼파라미터·Full 통계·소스 해시·lead별 train/val/test 분할 해시의 일관성을 확인했다. 저장 예측 60개의 관측·마스크·타깃 인덱스를 lead별로 대조해 동일함을 확인하고, 해양 격자 RMSE를 재계산해 평가 파일과 2e-6 K 이내 일치했다. baseline과 P0의 같은 seed끼리 비교하며 100% subset 3995개와 Full 7644/7639개를 구분했다.
- 평균 SST RMSE 감소율(양수=PINN 개선), 데이터 10/25/50/100% 순: lead7은 4.63/0.32/1.53/1.27%, lead14는 2.85/2.58/1.46/0.40%. Full에서는 각각 0.98%, 0.91% 악화했다. 가장 분명한 작은 데이터 사례는 10%/lead7: baseline 0.9568±0.0358 K → PINN 0.9125±0.0268 K(3seed 평균±표본 SD), 3개 seed 모두 개선했다. 25%/lead7은 평균 개선 폭이 작고 1/3 seed만 개선했다.
- MHW CSI는 10%/lead14에서 0.1806→0.2098, 25%/lead14에서 0.1810→0.2308로 각각 3seed 모두 개선했다. 단 25%/lead14 FAR도 0.5218→0.5582로 증가해 탐지율·오경보 사이 교환관계를 함께 보고한다. 50%/lead14 CSI는 0.2146→0.2123으로 평균 악화했다. Full/lead14 CSI는 0.2415→0.1943으로 세 seed 모두 악화했다.
- 통계 판단: 조건당 반복은 3쌍뿐이다. 쌍 내 부호 교환 가능성을 가정한 양측 exact sign-flip 검정은 최소 p=0.25이며, 이번 결과로 seed 반복에 대한 p<0.05 확증을 주장하지 않는다. nested subset, 두 lead, 동일 테스트 격자·날짜를 독립 반복으로 풀링하지 않았다.
- 보조적으로 30/90/180일 circular time-block bootstrap을 각 4000회 실시했다. 모든 seed와 모델에 같은 날짜 블록을 적용해 시간·공간·모델 간 대응을 유지했다. 10%/lead7의 90일 블록 ΔRMSE(PINN−baseline) 95% 구간은 [-0.0620,-0.0274] K, 10%/lead14는 [-0.0565,-0.0112] K이며 두 조건 모두 다른 블록 길이에서도 0을 제외했다. 이는 이미 학습한 모델에 조건부인 테스트 시계열 변동의 탐색적 분석이고, 새로운 seed 변동을 포함하거나 다중비교를 보정한 유의성 검증은 아니다.
- 해석 결정: “데이터 제한 조건에서 P0의 정규화 효과와 부합하는 개선이 관찰되며, Full에서는 이득이 사라지거나 악화한다”는 제한적 결론을 채택한다. 데이터가 적을수록 항상 효과가 커진다거나 P0가 보편적으로 우월하다고 말하지 않는다. 모든 subset PINN의 평균 SST RMSE는 Full baseline보다 높아 Full 대체·확정적 표본 절감률을 주장하지 않는다.
- 공통 정규화가 선택되지 않은 Full 학습 데이터의 정보를 포함하고, subset seed와 training seed가 묶여 있으며, early stopping으로 실제 step 수가 다른 점을 명시했다. 각 실행의 step·지표, 개선 seed 수, 부호 검정, 조건부 시간 구간은 보고서와 CSV에 남겼다. 추후 새 실험을 할 경우 가설·조건을 먼저 고정한 추가 seed/시간 구간 검증이 필요하며 이번 분석에서 새 학습을 실행하지 않았다.


## 2026-09-25 — 연구적 차별성을 위한 후속 실험 제안 (미확정)

- 사용자가 novelty를 강화할 실험 방향을 질문했다. 관련 원문 HW-OPINN, MARINA, SSTODE, SeaCast를 확인하고, 단순 물리 제약 추가·자료 부족 성능 개선·adaptive lambda만으로 최초성을 주장하기 어렵다고 판단했다. 링크와 상세 제안은 docs/next_experiments_proposal.md에 기록했다.
- 우선 제안은 물리 P0 vs 일반 정규화 vs 물리 forcing 시간 대응을 깨뜨린 대조군이며, 다음은 자료량·lambda·물리 근사 오차에 따른 적용 범위, subset-only 통계와 정보 접근 통제, 독립 기간/해역 평가, subset/training seed 교차 및 학습량 민감도다. Full 악화는 고정 lambda=0.1 결과이므로 PINN 일반의 한계로 단정하지 않는다.
- 현재 모델 코드·시간 분할·검증/테스트 인덱스·Full 결과를 보존한 별도 프로토콜을 제안했다. 10/25/Full×두 lead×3seed에서 새 대조군 두 개를 추가하면 36 본 실행이지만 튜닝과 구조 일치 확인 비용은 별도다. 이것은 확정 결정이 아니며 새 학습·subset 생성·모델 변경은 실행하지 않았다.


## 2026-09-25 — 후속 제안 축소 및 lambda=0 구현상의 차이 발견

- 사용자 비평을 검토하고 next_experiments_proposal.md를 단계별 수정안으로 갱신했다. 핵심을 lambda 스윕·tendency-only 대조로 좁히고, 기후값 forcing은 선택 단계, forcing 섞기·새 해역·교차 seed 확대는 후순위로 옮겼다. 새 본 학습이나 모델 수정은 수행하지 않았다.
- 실제 P0 코드는 lambda_physics=0에도 lambda_rres=10의 rres_l2를 별도로 더한다. zero_rres_physics=True는 물리식의 Rres만 제외하며 보조 손실은 유지한다. P0에는 we/rres head도 있다. 따라서 기존 baseline, 모든 추가 손실 off인 수치 일치 검증, 보조 손실만 있는 physics lambda=0 ablation을 분리해야 한다. 현재 결과는 설정 전체 효과이며 물리 잔차 단독 효과로 귀속할 수 없다. 수치 일치 테스트는 아직 실행하지 않았다.
- 현재 daily_rollout은 SST 끝점 사이 선형 경로이고 경향은 (T_pred−Tm)/lead이다. RHS는 예측 공간 기울기·we에도 의존한다. tendency-only 대조의 단위·분모·마스크·보조 손실·gradient 비교를 명시했다. 기후값 forcing도 스케일 교란을 자동 제거하지 않는다.
- MHW 평가기의 지역 정의(해양 MHW 비율>15%, 최소5일)로 lead7/14 관측 사건은 각각30개다. 두 lead를 독립60건으로 합치지 않는다. 212381은 양성 격자×날짜 수이며 사건 수가 아니다. headline CSI/POD/FAR는 pooled grid-time 지표다. 매칭 지역 사건은 실행별8~20개이므로 RMSE를 주 지표로 하고 세부 단계 분석은 미뤘다.
- 기존 P0 실측 평균: 10% lead7/14 1.38/1.73시간, 25% 1.49/1.56시간, Full 2.67/2.29시간. GPU5+5 공유 환경의 process wall time이다. 대표18회 합33.35시간, 새 lambda 두 값36회는 같은 종료 패턴 가정 시66.7 process-hours로 추정되며 전용GPU-hours가 아니다. Colab 속도로 환산하지 않았다.
- 10%·Full/lead7/3seed에서 lambda0.01·0.3 12회 + 보조 손실만 있는 lambda0 6회 + tendency-only 6회, 기본24회 파일럿을 제안했다. 전체 대표 그리드 확대 시기본72회이며 강도튜닝 별도, 기후값 대조는추가18회다. 이는 실행 확정이 아니다.
- 해석상 보완: P0와 tendency-only의 비슷한 결과는 동등성 증명이 아니며 “물리의 추가 이득을 확인 못함”으로 먼저 해석한다. 최적lambda가 자료량과 함께 감소해도 반비례 법칙은 아니다. Full 손해 없음에는 사전 비열등성 허용폭이 필요하다. paired차이, 조건부90일 블록bootstrap4000회(30/180일 민감도), 단일 주비교와 다중비교 범위를 초안으로 명시했고 실용성·비열등성 허용폭은 미정으로 남겼다.


## 2026-09-25 — 사용자 승인: lead7 10%·Full lambda 파일럿 18회 시작

- 사용자가 먼저 파일럿을 실행하고 결과를 확인한 뒤 나머지를 확장하도록 승인했다. 직전 논의의 핵심인 물리 lambda와 보조 손실 분리에 집중해 lead7, 10%(376개)·원래 Full(7644개), seed42/7/123, lambda_physics=0/0.01/0.3의 18개를 새로 실행했다. 기존 baseline과 P0 lambda=0.1 완료 결과를 비교점으로 사용한다. tendency-only·기후값·25%·lead14 등은 이번 큐에 넣지 않았다.
- lambda=0은 lambda_rres=10을 유지한 보조 손실 대조군이다. lambda_we_tie=0, 기존 모델·Full 정규화/RHS·history30·batch16/fp32·S478/max38240/warmup2390·478 step 검증·patience12를 유지한다. best checkpoint는 기존과 동일한 val_rmse로 고른다.
- 별도 scripts/train_lambda_pilot.py를 추가했고 원래 src와 scripts/train_data_efficiency.py를 변경하지 않았다. 새 실행은 물리 lambda와 출력 경로·식별 메타데이터를 추가했으며 원래 P0 초기화와 loader RNG 흐름은 보존했다. 기존 결과와 Full batch8/batch16 파일은 덮어쓰지 않는다.
- 실제 원본 모델·graph·데이터를 사용한 CPU 검증에서 3seed 모두 추가 손실을 전부 끈 P0와 baseline의 공유 초기 가중치·gradient·두 update 뒤 파라미터·예측 일치를 허용 오차 내 확인했다. 최대 파라미터 차이 5.96e-8, 최대 clipping 후 gradient 차이 1.49e-8, 최대 예측 차이 5.96e-7이다. 보조 손실의 공유 파라미터 gradient가 0이 아님도 확인했다. 증거: docs/server_preflight/lambda_pilot_cpu_check.json.
- 추가 head 생성은 RNG를 소비한다. 같은 seed여도 기존 baseline/P0의 실제 shuffle 순서까지 동일하지는 않았음을 확인했다. CPU 일치 검증에는 동일 batch를 사용했고, 새 파일럿 arm들은 기존 P0 RNG 흐름을 그대로 유지해 lambda 간 비교와 기존 P0 재사용을 맞췄다. 기존 baseline 대비 전체 효과와 보조 손실 대조 대비 물리 효과를 구분해 보고한다.
- GPU0에서 10%·Full×lambda0/0.01/0.3 여섯 조합이 batch16 smoke 4 update/검증/체크포인트 재로딩/테스트 예측 저장을 통과했다. lambda0.1의 기존 실행기와 새 실행기 smoke는 체크포인트 모든 파라미터와 예측이 정확히 일치했다(max diff=0). 검증 보고서: docs/server_preflight/lambda_pilot_validation.json. GPU process name은 모두 yeonwoo다.
- 실행 전 분석 규칙은 docs/server_preflight/lambda_pilot_protocol.json에 코드 해시와 함께 고정했다. 자료 조건별 세 seed 평균 best validation RMSE로 lambda0/0.01/0.1/0.3 중 하나를 선택하고 정확한 동률은 작은 lambda를 우선한다. 테스트로 lambda를 선택하지 않는다. 모든 arm의 paired RMSE·CSI/POD/FAR와 조건부 time-block bootstrap(90일 주, 30/180일 보조, 4000회)을 보고한다. seed n=3의 한계와 비열등성 margin 미설정을 명시한다.
- 큐: scripts/launch_lambda_pilot.py, tmux mhw_lambda_pilot, Ceph outputs/data_efficiency/lambda_pilot_lead7_20260925/_queue/batch_status.json. GPU0/1 각각 최대5개 동적 배정이며 긴 Full을 먼저 배정했다. 파일럿 lease와 기존 디렉터리 존재 검사를 사용한다. 기존 current_attempt.json은 새 파일럿을 가리키고 lambda_pilot_current.json도 추가했다.
- 시작 확인 시 coordinator PID2154557, 학습 10개 모두 225step 이상 진행, 나머지8개 대기였다. 각 GPU 약41830MiB 사용, 프로세스당8362MiB, 모든 GPU 이름 yeonwoo를 확인했다. 상세 증거: docs/server_preflight/lambda_pilot_launch_check.json.
- 18개 완료 후 원본 MHW 평가기를 실행하고 scripts/summarize_lambda_pilot.py가 기존 참조 결과까지 포함한 _analysis/report.md·CSV·validation_selection.json을 Ceph에 자동 생성한다. 부정적/혼재 결과도 모두 보고한다. 더 큰 큐를 자동 실행하지 않고 파일럿의 구현 정합성·효과 크기와 한계를 검토한 뒤 확장을 판단한다. 이 기록은 학습 시작 보고이며 18개 완료 보고가 아니다.


## 2026-09-25 — lambda 파일럿 18회 완료 검토: 대규모 확장보다 대조 확인 우선

- 사용자가 완료 결과와 추가 실행 가치를 요청했다. 18개 모두 completed/returncode0, 평가와 자동 분석 완료, OOM/traceback 없음. 03:53~07:55 UTC 약4시간2분 소요. seed별 지표·실제 step·lambda 및 보조 손실 설정을 교차 확인했다.
- lead7 10% 평균 test RMSE: baseline0.95677, aux0(physicsλ0/rresλ10)0.88348, P0λ0.01=0.89827, λ0.1=0.91246, λ0.3=0.88425. aux0가 baseline 대비7.66% 개선(3/3seed), 기존λ0.1보다3/3seed 우수했다. λ0.3은 baseline 대비7.58% 개선이나 aux0 대비 평균+0.00077 K여서 물리 잔차의 추가 이득은 명확하지 않다.
- Full 평균: baseline0.77731, aux0 0.78368, λ0.01 0.78188, λ0.1 0.78493, λ0.3 0.79052. 이번 범위에서 Full 평균 손해를 없앤 arm은 없었다. 검증 선택은10%λ0.3, Fullλ0.1이며 테스트 평균최저 arm으로 선택을 바꾸지 않았다. 10%의λ0.3과0.1 검증평균차이는약0.00063K다.
- 10% λ0.1−aux0의90일조건부ΔRMSE구간은[+0.00861,+0.05130]K, λ0.3−aux0는[-0.01241,+0.01427]K. 이는 현재 학습된 모델의 테스트시기 변동이며 3seed 모집단 유의성·동등성 증명이 아니다. 10% aux0는CSI0.3548로λ0.3의0.3207보다 높지만 FAR도0.4383대0.4190으로 높아 교환관계를 함께 기록했다.
- 해석 수정: 기존 “저자료에서 P0 설정 전체 개선”은 유지되지만 이를 물리 잔차 단독 효과로 귀속할 근거는 약해졌다. 보조 정규화와 최적화 변동을 먼저 분리해야 한다. 기존baseline/P0의추가head RNG소비에따른shuffle차이가 남아 aux0 우세를 보조손실만의 효과로 확정하지 않는다.
- 후속 권고는 10%·Full/lead7/3seed에서 P0와같은초기화·shuffle흐름, 모든추가손실0인 맞춘baseline6회다. 25%·lead14 등 넓은 확장은 우선 보류 권고. 새로운 학습을 이번 점검에서 시작하지 않았다. 상세근거는docs/lambda_pilot_review.md에 기록했다.


## 2026-09-25 — P0 개선 방향 검토 (제안, 실행 미승인)

- 사용자 질문은 P0 개선 방법이다. 이번 검토에서는 원본 코드와 완료된 lambda 파일럿을 읽었으며 모델 수정·새 학습·기존 결과 덮어쓰기는 하지 않았다. 아래는 확정 프로토콜이 아니라 검토 우선순위다.
- 우선 학습 데이터만으로 예산식의 적합성을 점검하는 방안을 제안한다. 관측 SST 변화와 열속/이류/연행 항을 같은 K/day 단위로 비교하고, 해안 차분 stencil·단위/부호·forcing 날짜·물리 residual clamp 포화 비율을 확인한다. 관측값도 잔차가 클 수 있으며 이것만으로 코드 오류는 아니다. 누락된 과정, 자료 오차, SST를 혼합층 평균 수온 대신 쓰는 근사도 구분해야 한다. 정답 관측 경로와 기존 선형 경로를 모두 점검하면 경로 근사 영향도 나눠 볼 수 있다.
- 실제 P0는 src/train_jeju_hilam.py의 daily_rollout_physics_loss를 호출한다. src/train_hybrid_jeju_forecast.py:667 이후는 끝점 SST 사이를 선형 보간하고 매일 residual을 제곱하여 평균한다. 예측 대상은 끝점 하나이므로 기간 평균 경향과 기간 평균 열수지 RHS를 비교하는 손실이 우선 비교할 후보다: L_int = mean_ocean{[((T_pred-T_now)/L - mean_day(RHS_Kday))/s]^2}. s는 공통 고정 RHS 스케일의 제곱근으로 두면 단위를 유지하면서 시간 집계 방식을 비교할 수 있다.
- 이것을 성능 저하의 입증된 원인이나 반드시 좋아지는 해법으로 부르지 않는다. clip이 없는 단순 MSE에서 일별 RHS가 예측과 무관하면 일별 residual 제곱 평균과 평균 residual 제곱은 예측과 무관한 RHS 분산만큼만 달라 gradient가 같을 수 있다. 현재 코드는 이류의 예측 공간 기울기, 학습되는 we, clipping 등이 있으므로 실제 gradient 차이를 먼저 확인할 가치가 있다. 기간 적분형도 이류 계산에 같은 선형 경로를 쓰면 경로 근사 자체는 남고, 일별 오차가 상쇄될 수 있다.
- 원본의 integrate_budget_forecast는 별도로 존재하지만 SST 예측 자체를 물리 적분으로 바꾸는 다른 경로다. 위의 끝점 예측에 대한 기간 평균 손실과 동일하지 않으며 이번 제안을 그 함수 활성화로 대체하지 않는다.
- 보조 손실과 물리 손실을 분리한다. 맞춘 baseline(모든 추가 손실 0), 기존 방식의 순수 P0(lambda_rres=0, zero_rres_physics=True), 기간 평균 방식의 순수 P0를 동일 초기화/배치 순서에서 비교하고, 기존 aux0를 실용 성능 비교점으로 유지하는 것이 후보 설계다. 기존 권고인 맞춘 baseline 6회(10%/Full, lead7, 3seed)는 여전히 미실행이며, 범위를 확정하기 전 새 실행 수를 고정하지 않는다. 모델 구조 변경 없이 별도 실험 경로를 설계하고 원본은 보존한다.
- 연행 항은 Td_step=prev-entrainment_dt를 쓰며 배치의 Td_glorys를 직접 쓰지 않는다. Tm은 SST다. 이 근사 및 자유롭게 예측되는 we의 물리 규모/보상 효과도 진단 대상이지만, 기간 평균 손실 변경과 동시에 여러 항을 바꾸지는 않는다.
- lambda를 다시 고르기 전 데이터 손실/물리 손실의 공유 파라미터 gradient 크기와 방향을 확인한다. 손실의 숫자가 작다는 이유로 영향도 작다고 간주하지 않는다. gradient 불균형이 확인되면 warmup 또는 gradient 기반 가중치 조정을 후속 후보로 두고 검증 RMSE로 선택한다. 관련 원문: Wang et al., https://arxiv.org/abs/2001.04536 .
- 적분/약형 물리 손실 자체는 기존 연구 방향이다(예: Kharazmi et al., https://arxiv.org/abs/1912.00873). 이를 그대로 구현했다거나 그 자체가 새로운 방법이라고 주장하지 않는다. 성공 기준은 lambda_physics=0을 포함한 공정한 대조에서 물리 항의 추가 이득을 확인하는 것이며, 기존 test를 보고 발전시킨 후속 설계라는 한계도 유지한다.


## 2026-09-25 — 해양 이웃 차분 보정과 P0/P1 30회 실행 승인

- 사용자는 새로 제시한 해양 이웃 차분 방식으로 실험을 진행하고 하던 작업을 마치라고 명시했다. 해안 처리 선택 대기는 해소됐다. 기존 원본src·Full결과·평가인덱스는 보존한다.
- 관측 진단에서 육지NaN→0 채움이 바다의 공간 미분에 들어가는 결함을 확인했다. 해양661개중141개가영향받고, 관측이류RMS는 내부0.1816대경계6.7494 K/day였다. 보정은 축별로 바다양쪽이면중앙차분, 한쪽이면한쪽차분이며660개물리격자를유지한다. 남북이웃없는[24,14]만물리손실제외, 온도학습/평가는661개전체다.
- 해양이웃차분의선형장정확성·육지채움값불변성·육지gradient0·유한차분역전파, 기간평균손실의해석해·P1부호/단위 검사를통과했다. 새공통Full-train P0 RHS제곱평균은0.08996321398(K/day)^2, 기존9.7367229778은보존한다. 입력/타깃정규화는변경하지않는다.
- 범위는10%/lead7/3seed에서맞춘baseline3회 + P0일별/P0기간평균/P1일별 각각lambda.01/.1/.3의9회, 총30회. 보조손실0, 같은초기화·배치순서, batch16/S478/기존earlystop, GPU0/1최대5개씩이다. 모든큰출력Ceph, GPU이름yeonwoo를유지한다.
- GPU사전검증중cuDNN GRU가eval모드에서backward를거부하여초기gradient진단만train모드로고쳤다. 난수상태와모델가중치가진단전후같은지assert한다. 첫실패smoke는검증폴더에보존했고본학습실패가아니다.
- 실행전프로토콜은docs/server_preflight/physics_pilot_protocol.json에고정했다. lambda0공통baseline포함검증평균으로선택, 모든arm/test결과보고, 새seed유의성·동등성·경계보정단독인과효과주장없음, 추가자동확장없음. 상세진단과경계근사한계는docs/physics_boundary_review.md참조.


## 2026-09-25 — 보정된 P0/P1 30회 본 학습 시작 확인

- GPU사전검증은맞춘baseline/P0일별/P0기간평균/P1일별4조건모두통과했다. 초기모델/RNG/배치순서일치,4step업데이트·검증·checkpoint재로딩·예측저장,온도평가661개유지,물리660개를확인했다.
- tmux mhw_physics_pilot, coordinator PID 2538881. 큐는/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/physics_pilot_lead7_20260925/_queue/batch_status.json이다. 현재10개running/20개pending이며모두50step이상, GPU0/1각5개, process name yeonwoo를확인했다. 생산로그에Traceback/OOM은없다.
- 검증초기batch의lambda.3가중물리/데이터gradient비는P0일별2.3675,기간평균2.3551,P1일별2.3825였다. 기존physics warmup을유지하며이한batch진단으로가중치를자동변경하지않았다.
- current_attempt.json과physics_pilot_current.json을새큐로갱신했다. 각조건3seed완료뒤MHW평가와paired/RMSE분석을자동실행한다. 원본src/기존Full및subset결과는검증해시로보존한다. 이번기록은실행시작이며전체완료보고가아니다.


## 2026-09-25 — 사용자요청으로5+5중단후4+4재시작

- 5+5에서는GPU당48,394MiB사용/282MiB여유로OOM여유가작아사용자가현재실행을끄고4+4로줄이라고지시했다. 해당큐관리자와학습10개만종료했고GPU메모리반환을확인했다. 이전결과는삭제하지않고stopped_by_user/미시작작업cancelled_before_start로기록했다.
- 학습/손실파일을그대로두고관리자만4슬롯/GPU로복제했다. 원래GPU검증보고서와학습코드해시일치를확인해학습동작변경없는단순동시실행수축소에추가smoke를반복하지않았다.
- 새큐/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/physics_pilot_lead7_20260925_4plus4/_queue/batch_status.json, tmux mhw_physics_pilot_4plus4, coordinator PID2546949. 8개running/22개pending, GPU0/1각4개,모두25step이상,이름yeonwoo/오류없음을확인했다. 새30회는같은seed에서처음부터실행하며기존checkpoint의optimizer/stream/RNG누락때문에불완전재개하지않는다.
- 과학적조건·batch16·정규화·분할·평가·자동후속보고는유지했다. 실제메모리와PID/step은docs/server_preflight/physics_pilot_4plus4_launch_check.json에기록했다.


## 2026-09-25 — GitHub 공유용 일일 정리

- 사용자 요청으로 오늘 완료한60회본비교·18회lambda파일럿, 해안차분결함과보정근거, 진행중인30회/4+4상태를 docs/daily/2026-09-25.md에 정리했다. 진행중 실험의결론을완료결과로표기하지않고기록시각을포함한상태스냅샷을분리했다.
- 서버실행폴더에Git이력이없어 /home/yeonwoo/jeju-mhw-pinn-data-efficiency에지정원격저장소를별도clone했다. 원격main이력을보존하며코드·문서·작은결과표/검증JSON만공유한다. 원본src·subset·Full결과파일은이번변경에포함하지않았다.
- README에서최신진행문서로연결하고오래된subset미연결안내를갱신했다. 완료lambda분석CSV/JSON/Markdown을docs/lambda_pilot_results에보관했다. NetCDF·체크포인트·예측배열·실행/설치로그·Ceph출력symlink는업로드에서제외한다.
- 현재4+4학습은계속실행하며Git정리작업으로중단하거나학습코드를수정하지않는다.

- GitHub 공유 중 PNG blob 업로드 도구가 `user rejected MCP tool call`을 반환해 해당 PNG는 제외했다. 승인된 PDF 그래프를 보고서에서 연결하며 코드·문서·결과표 업로드는 유지한다.
