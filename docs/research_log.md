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


## 2026-10-01 — Figure 4 geospatial 재생성 및 provenance 검증

- 원본 main batch16 ratio_10의 baseline/P0 original, lead7/14, seed7/42/123 총12개 저장 prediction만 CPU로 재계산했다. 학습·추론·GPU 사용은 없다.
- 실제 NetCDF SHA-256이 run config와 일치하고, 12개 저장 target 배열 전체가 NetCDF 동일 날짜·native grid와 정확히 일치했다. C-order graph/node↔grid mapping과 661-cell mask를 검증했다.
- latitude는 실제30→37°N 오름차순이며 stored_direction=decreasing 속성은 불일치하는 잔존 메타데이터다. 실제 좌표와 target 대조를 기준으로 row0=south, column0=west를 확인했다.
- pcolormesh에 실제 좌표의 midpoint/half-spacing 경계를 사용했다. 재생성한 기존 index-grid와 새 지도 배열 차이는0; 예전12자리 CSV 대비 최대 차이는4.81e-13K 미만이다. 모든661개 해양cell의 날짜가 유효하며 blank296개는 평가마스크 밖이다.
- 재현 코드: paper_analysis_20260929/generate_figure4_geospatial.py. 산출물과9개 Yes/No 판정: paper_analysis_20260929/results/figure4_geospatial/provenance_report.md 및 verification.json. 원본 결과·기존 index-grid 그림은 보존했다.


## 2026-10-08 — subset별 전처리 재구축

- 사용자 요청으로 Full 통계 공유와 별개의 subset-fit 경로를 추가한다.
- 각 비율·seed·lead별 unique input-history 날짜, selected target 날짜, selected sample RHS scale을 사용한다.
- 24개 manifest를 data/subsets/preprocessing_subset_v1/에 생성하며 원본 v2 선택·기존 결과·모델 구조는 유지한다.
- Validation/Test 인덱스는 그대로 두고 subset 통계를 적용한다. 학습은 실행하지 않는다.
- 기존 anomaly 변수의 upstream climatology 재추정은 포함하지 않는다.
- 정의·적용 API·검증: docs/subset_preprocessing_v1.md.

- 완료: 24개 manifest 생성 및 실제 dataset 변환 검사 통과. 선택되지 않은 관측 변경 불변성 등을 포함한 5개 테스트 통과. manifest/source/subset 해시 및 저장 manifest 재로딩 검증 통과. 학습 미실행. 검증 기록: data/subsets/preprocessing_subset_v1/validation_report.json.


## 2026-10-08 13:54 KST — 사용자 운영 규칙 영구 기록 및 작업 사본 위치 정리

- 대상: `/ceph_data/yeonwoo/AGENTS.md`, `MHW_subset/AGENTS.md`, 이 수정 이력 파일.
- 변경/이유: 사용자 요청에 따라 Ceph 내부 저장, GPU process name `yeonwoo` 설정·검증, 한국시간 분 단위 단일 수정 로그를 이후 작업의 운영 규칙으로 명시했다.
- 위치 변경: `/home/yeonwoo/subset_preprocessing_work` → `/ceph_data/yeonwoo/_work/subset_preprocessing_work`. 앞서 만든 코드 작업 사본·원본 백업·검증 파일을 Ceph에 보관하기 위해 이동했다.
- 검증: 이동 후 목적지 존재 및 로컬 원본 부재 확인. 규칙 두 파일 및 수정 이력 반영 확인. 모델 코드 수정·학습·GPU 실행 없음.


## 2026-10-08 13:58 KST — 학습 전 최종 감사 스크립트 추가

- 대상: docs/pretraining_audit_20261008/audit.py. anomaly 생성 코드 재현 및 24개 manifest·Baseline/PINN 설정 적용을 검증하기 위해 작성했다.
- 원본 모델/전처리 소스와 manifest는 수정하지 않는다. CPU 전용 설정 검사와 optimizer 없는 loss 검증만 수행하며 본 학습·GPU 실행은 하지 않는다. 결과는 동일 감사 폴더에 저장한다.


## 2026-10-08 14:03 KST — 学習前 최종 감사 완료

- 대상: docs/pretraining_audit_20261008/의 metadata, anomaly_reconstruction, runtime_verification, report.md, verdict.json.
- 결과: thetao anomaly 6개를1993-01-01~2013-12-31 공통 climatology로 정확히 재현했다(최대오차0). 선택되지 않은 학습자료 포함으로 subset-only FAIL. sst 생성 코드는 미확인(메타데이터1971–2000).
- 실행경로 검증: 24개 통계/RHS 재계산, 48개 Baseline/PINN CPU 설정·평가, Full 덮어쓰기 차단, nested/split 대조 PASS. optimizer/backward/GPU/본학습 없음.
- 추가 실행 전 조치: 직접 학습 진입점에 GPU process title 및 Ceph output 경로 강제가 필요함을 보고했다. 원본 학습 소스·데이터셋·manifest는 변경하지 않았다.


## 2026-10-08 14:10 KST — v2 subset-only climatology 구현

- 대상: src/mhw/subset_climatology.py 신규, src/mhw/subset_preprocessing.py v2 manifest 분기 추가.
- 이유: 선택되지 않은 관측을 사용하던 SST/thetao anomaly를 대체하기 위해366개 월·일 bin과 subset-only 순환 선형보간을 구현한다. 육지는0·통계 제외, 해양 raw 결측은 오류 처리한다. Feb29는 별도 bin이며 관측이 없으면 동일 순환보간을 쓴다.
- 기존 v1 manifest/원본 raw 데이터는 변경하지 않으며, 변경 전 소스는 /ceph_data/yeonwoo/_work/subset_v2_originals에 보존했다. 본 학습/GPU 미실행.


## 2026-10-08 14:13 KST — v2 실행 경로 운영 규칙·일정 연결

- 대상: src/train_jeju_hilam.py. CUDA 접근 전에 yeonwoo title을 강제하고 출력 경로가 Ceph 내부인지 검사한다.
- 기존 자료량 비교의 고정 optimizer-step/validation 간격을 유지할 수 있도록 --virtual-epoch-steps와 epoch 사이 지속되는 순환 train stream을 추가했다. 기본 미지정 경로의 epoch 동작은 유지한다.
- 이번 요청 범위는 사용자 확인에 따라 v2 구축·검증·실행 계획까지이며 학습은 시작하지 않는다. 학습 코드 변경은 CPU 검사로 검증 예정.


## 2026-10-08 14:17 KST — v2 생성기 및 회귀 검사 추가

- scripts/build_subset_preprocessing_v2.py, tests/test_subset_climatology.py 추가. raw 관측 변조 불변성·DOY/FEB29/육지 검증·실제 loss 분모 검증·고정-step 순환 검사를 포함한다.
- QC는 해양 유한값/양의 분산/선택 raw 범위 안 climatology를 요구한다. 수치 이상 검토 기준은 train max|z|20, 전체 날짜 max|z|30, 전체/학습std 비율0.1~10으로 사전 선언했다. 이 기준은 물리적 극단현상의 분포 보증이 아니며 기준 초과 시 자동 학습하지 않는다.
- 학습 실행은 하지 않고 계획만 생성한다.


## 2026-10-08 14:21 KST — v2 데이터셋 구축·검증

- scripts/build_subset_preprocessing_v2.py로24개 manifest와12개 climatology를 별도 생성했다.
- 선택되지 않은 raw 날짜 변경 불변성, target/RHS 불변성, 실제 실행 진입점/CPU loss 및 Baseline/PINN 동일성을 검증했다.
- QC 결과: {'subset_only_invariance': 'PASS', 'baseline_pinn_equal': 'PASS', 'coverage_missing_extremes': 'PASS', 'actual_v2_entrypoint_no_full_override': 'PASS'}. v1 파일/원본 subset 보존 및 MHW 평가 함수 해시 유지. 본 학습 미실행;48개 비교+24개 ablation은 계획만 저장했다.


## 2026-10-08 14:29 KST — v2 최종 보고서 및 실행 계획 정리

- 대상: docs/subset_preprocessing_v2.md 신규, README.md의 subset 진입점 설명 갱신 및 보고서 링크 추가.
- 이유: v2 검증 결과·코드 위치·10% 희소 DOY/분산 축소 한계·미실행 48+24개 실험 계획을 재현 가능하게 정리했다.
- 검증: 24개 manifest/12개 climatology 및 기록된 소스 해시, 기존 인덱스/분할/target/RHS 일치 확인. 수정 전 대비 add_acc_and_sw_metrics/anomaly_acc/run_epoch AST 동일. 네 gate PASS 및 12개 회귀 검사 결과를 기록했다. 최대 abs(z)=29.942는 수치 기준 통과이며 물리적 타당성 보증은 아님을 명시했다.
- 원본 raw/v1 유지. 모델 학습·optimizer step·GPU 실행 없음.


## 2026-10-08 22:51 KST — v2 최종 정밀 감사 시작

- scripts/audit_v2_input_outliers.py 추가; outputs/v2_final_preflight_20261008/에 신규 분석을 저장한다. 실제 input-history 중복을 반영한 변수·split별 통계/꼬리 비율과 이상값 좌표를 계산한다.
- v1/v2/raw/기존 결과 보존. 해안 차분 보정 미연결을 확인해 별도 재현 검사 예정. GPU 드라이버 접근 실패; 본 학습 및 GPU smoke 미실행.


## 2026-10-08 22:54 KST — v2 해안 차분 및 gradient 경로 감사 추가

- scripts/audit_v2_training_path.py 추가. 기존 해안 보정의 순수 함수를 읽어 상수 해수/육지값 변조 검사와 10% RHS 진단을 별도 CPU 프로세스에서 실행한다. 원본 소스·manifest는 변경하지 않는다.
- 실제 모델의 CPU 초기 predictor 동일성 및 손실별 gradient 분리를 검사한다. GPU smoke를 대체하는 PASS로 보고하지 않으며 optimizer step은 수행하지 않는다.


## 2026-10-08 22:58 KST — GPU 조회 결과 정정 및 CPU 진단 완료

- scripts/audit_v2_training_path.py, outputs/v2_final_preflight_20261008/gpu_smoke.json: GPU 미실행 이유를 해안 차분 FAIL로 명확히 했다. 제한된 실행에서 초기 nvidia-smi가 실패했지만 후속 Ceph 실행에서는 returncode=0, RTX A6000 10개/활성 GPU 프로세스 없음이 확인됐다. GPU 불가를 고정 문구로 기록한 부분을 정정했다.
- CPU 진단: 해양 상수장의141개 격자에서 legacy 차분이 비영 gradient를 생성. 해안 보정 시0. 10%/seed42/lead7 RHS 10.045714→0.084931(118.28배 차이). 데이터·manifest는 그대로 보존했다.
- 실제 hidden256 모델의 CPU 손실별 gradient 검사 통과. GPU smoke/optimizer step/정식 학습은 실행하지 않았다.


## 2026-10-08 23:03 KST — v2 정밀 감사 최종 FAIL 보고

- scripts/report_v2_final_preflight.py 추가, outputs/v2_final_preflight_20261008/의 보고서·CSV·verdict·blocked_execution_plan·29.942 위치 진단 생성.
- 입력24조건/1,368변수-split 검사 및 전체 abs(z)>5 좌표 기록. 기존29.942는 실제 입력 미사용 날짜. 입력 최대28.355; abs(z)>20은10% held-out에만 존재.
- 해안 보정 미연결 및118.28배 RHS 차이로 FAIL. zero-weight branch가 계산 자체를 끄지 않는 점, 기존72-run 계획의 Rres 가중치 불일치, 완전 resume 미지원도 기록했다.
- 24개 manifest/climatology/source 해시·nested·원본 인덱스 보존 검사 PASS. GPU smoke·정식 학습·optimizer step 없음. 사용자 지시에 따라 정식 실행 준비를 중단하고60회 기본/최대108회 후보 포함 계획은 BLOCKED 초안으로만 저장했다.


## 2026-10-08 23:12 KST — v3 해안 차분·손실 분기 수정 시작

- 대상: src/train_hybrid_jeju_forecast.py, src/train_jeju_hilam.py, src/mhw/subset_preprocessing.py, src/mhw/subset_climatology.py 및 신규 mhw/ocean_stencil.py. 수정 전 사본을 outputs/v3_preflight_20261008/originals에 보존했다.
- 기존 보정된 해양 이웃 차분을 공통 경로에 연결하고 비활성 loss의 연산을 생략한다. 기존 v1/v2·평가기 해시 및 기존 결과 파일 메타데이터를 기록했다. Input/target/anomaly는 공간차분과 독립이므로 v3에서 재생성하지 않는다.
- 승인 범위는24개 RHS 재계산·회귀검사·조건부2모델/최대2GPU batch smoke까지이며 본 학습 금지.


## 2026-10-08 23:17 KST — v3 loss 분기 회귀검사 및 수정

- tests/test_v3_physics.py 추가. 상수장/NaN·Inf 육지/기존 pilot 일치/선형장/고립격자/P0·P1 RHS 일관성/비활성 손실 분기를 검사한다.
- 첫24개 검사 중22개 통과, 분리한 Rres 함수의 device 지역변수 누락2개를 발견해 rres_pred.device로 수정했다. 기준 완화 없이 전체 검사를 재실행한다.


## 2026-10-08 23:21 KST — v3 24개 RHS manifest 생성 및 CPU 검증 완료

- scripts/build_subset_preprocessing_v3.py 및 data/subsets/preprocessing_subset_v3 생성. 기존climatology를 경로·해시로 참조하고 Input/Target 통계·인덱스는 그대로 유지했다.24조건/20개 고유 RHS 계산.
- 24개 회귀검사,48개 실제 main setup/CPU objective, 실제 v3 public loader, 기존 해안 구현과 전 해양 격자 일치, 보존 해시/결과 메타데이터 검사를 통과했다. 수정 전후 RHS·잔차는 outputs/v3_preflight_20261008에 저장했다.
- GPU smoke는 아직 대기, 본 학습 미실행.


## 2026-10-08 23:25 KST — v3 제한 GPU smoke 완료

- scripts/smoke_subset_v3_gpu.py 및 outputs/v3_preflight_20261008/gpu_smoke_attempt_001. 유휴 GPU1개에서 Baseline/P0 Physics only 각2batch16 forward/backward 및 각1회optimizer update만 실행했다.
- 실제 v3 loader/main setup/native run_epoch, finite loss/gradient, physics shared gradient 기여, norm1 clipping, checkpoint 상태 복원을 검증했다. NVML PID process name yeonwoo 확인. 초기 predictor/배치 순서 일치. 본 학습 미실행.


## 2026-10-08 23:31 KST — v3 최종 PASS 보고 및 사용 안내

- scripts/verify_v3_loss_arms.py의 실제CPU4조건 loss/gradient 검사 완료. scripts/report_subset_preprocessing_v3.py, outputs/v3_preflight_20261008/report.md·final_verdict.json·rhs_scales.csv, docs/subset_preprocessing_v3.md 및 README 안내를 저장했다. v3 index/validation 상태를 CPU 대기에서 최종PASS로 갱신했다.
- 24개 회귀검사/48개main setup/4개CPU손실조건/2개GPU smoke 통과. GPU 각2batch 및각1update만 실행. 모든 v1/v2·기존 결과·MHW평가·모델 정의 보존을 재확인했다.
- native baseline과aux head 모델의 parameter 수/실제gradient연결 수 차이, Rres강도 변경 관측, GPU smoke 범위와한계도 기록했다. 정식 학습 및ablation 미실행.

## 2026-10-08 23:43 KST — 과거 해안 보정 30회 실험 소급 검증

- 대상: `outputs/historical_coastal_audit_20261008/report.md`, `verification.json` 신규 생성. 기존 코드·데이터·checkpoint·결과 변경 없음, 학습/GPU 실행 없음.
- 이유: v2 원본 진입점의 해안 차분 오류가 과거 P0/P1/기간평균 실험에도 존재했는지 구분.
- 검증: 실제 30개 checkpoint 내부 설정과 외부 JSON 일치, 보관 src 22개 및 protocol scripts 11개 해시 일치, 동일 ocean_neighbors 보정 및 RHS scale 0.08996321398271197 확인. CPU 실제 격자/합성장 검사에서 상수해수 기울기 0, 육지값 불변, 세 물리손실과 공통 RHS 경로 일치·finite.
- 판정: 과거 30회 실험의 해안 보정 일관성 PASS. 다만 Full-train P0 scale 공통사용은 subset-only 조건과 다르며, λ=0 손실 계산 생략은 당시 미구현. Full RHS 전체 재계산이나 과거 GPU 학습 재실행은 하지 않음.

- 2026-10-08 23:51 KST — scripts/train_v3_study.py 신규 작성: native v3 모델/손실 재사용, epoch 경계 optimizer/RNG/loader 복구, loss/gradient finite 및 clipping 기록, Test 평가 분리. 기존 src/데이터/결과는 보존. 사용자 변경: 최대3GPU, 메모리 확인 후 GPU당3–4프로세스. 구문검사 PASS; 실행검증 예정.

- 2026-10-08 23:57 KST — scripts/manage_v3_study.py 및 analyze_v3_study.py 신규 작성: 첫쌍검증 후48조건 실행/완료후Test평가, GPU최대3×3–4 및외부프로세스 확인, 기존MHW함수/paired circular block bootstrap30/90/180일4000회, 전체CSV·그림·최종보고 자동생성. 구문검사 PASS. 기존코드/데이터 보존.

- 2026-10-08 23:58 KST — v3 study: /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/v3_baseline_p0_20261008: 48조건 protocol 저장. Baseline/P0 λ=.1/Rres0, 478step×80/patience12 유지. 최대3GPU/각4프로세스, 4GiB 및1.15배 메모리 여유 확인.

- 2026-10-09 00:01 KST — v3 study: 정식48회 실행 사전검사 PASS: 24회귀검사, v3 manifest/코드/참조무결성, CPU 실제 pair초기값/입력/통계/배치 일치 및loader복구 검증. 기존src 무수정. 최초test호출의PYTHONPATH 누락은 올바른경로로 재실행해해결. 추가관리자검증: 완료재사용config/manifest해시및사전검사PASS필수. 최대3GPU, GPU당3–4실행으로사용자변경반영.

- 2026-10-09 00:01 KST — v3 study: 실행 실패 ratio_10/lead7_seed42/p0 phase=train; 오류 로그 보존.

- 2026-10-09 00:01 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/baseline, PID 181533, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 00:03 KST — v3 study: 첫 실행 보존/중단: outputs/data_efficiency/v3_baseline_p0_20261008. P0 초기진단model.eval()의cuDNN RNN backward오류를model.train()으로수정(optimizer전). 자체Baseline PID181533및supervisor181520중단. setproctitle이/proc environ을덮으므로manager재접속PID확인은소유자+kernel start ticks로보강. 실패로그/원본실행기보존; _r1에서동일hash로둘다처음부터재실행.

- 2026-10-09 00:04 KST — v3 study: /ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency/v3_baseline_p0_20261008_r1: 48조건 protocol 저장. Baseline/P0 λ=.1/Rres0, 478step×80/patience12 유지. 최대3GPU/각4프로세스, 4GiB 및1.15배 메모리 여유 확인.

- 2026-10-09 00:06 KST — v3 study: 진단수정 GPU재검증 PASS: _r1/diagnostic_recheck. model.train에서 실제P0 predictor gradient>0·finite 및새Monitor/native run_epoch 1batch optimizer smoke PASS, Rres호출0. 이 smoke모델은폐기하고정식학습은원seed초기화. 최초실패1/중단1 로그원경로보존. 모든src/손실/데이터무수정.

- 2026-10-09 00:06 KST — v3 study: 정식재시작전 최종집계코드 실행횟수회계 보완: 이전 진단실패1/paired중단1/새초기화재시작2를현재study재시도와구분해 execution_summary.json에기록하도록수정. _r1 protocol의analysis해시를첫정식학습시작전에확정.

- 2026-10-09 00:07 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/p0, PID 185184, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 00:07 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/baseline, PID 185183, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 00:09 KST — v3 study: 실제첫쌍 config 대조PASS: predictor초기값/manifest/통계/split/입력지문동일. _r1/resource_plan.json 및OPERATIONS.md 작성. 9.9GiB예약메모리+15%여유+4GiB잔여기준으로 현재GPU당3개, 최대3GPU×3=9동시실행계산; 첫쌍완료gate후확장.

- 2026-10-09 00:10 KST — v3 study: 첫 정식epoch 건강검사 PASS: _r1/initial_training_health.json. 양모델 loss/gradient finite, 배치해시일치, Rres호출0, baseline physics호출0. best checkpoint CPU열기/모든weight finite/config·manifest 해시일치확인. 학습진행중, 첫쌍완료gate는아직대기.

- 2026-10-09 00:28 KST — v3 study: 학습 완료 ratio_10/lead7_seed42/baseline, best_epoch=4, val_RMSE=1.065011, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 00:29 KST — v3 study: 학습 완료 ratio_10/lead7_seed42/p0, best_epoch=2, val_RMSE=1.032487, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 00:29 KST — v3 study: 첫 정식 Baseline/P0 한 쌍 실행검증 PASS. 성능과 무관하게 나머지46조건 확장.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed42/baseline, PID 198862, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed7/baseline, PID 198858, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed7/baseline, PID 198864, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed7/p0, PID 198859, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed7/p0, PID 198865, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed123/baseline, PID 198860, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed42/p0, PID 198863, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed123/baseline, PID 198866, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 00:30 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed123/p0, PID 198861, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 01:03 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/baseline, best_epoch=2, val_RMSE=1.318040, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:04 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed123/p0, PID 220627, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 01:07 KST — v3 study: 학습 완료 ratio_10/lead7_seed7/baseline, best_epoch=1, val_RMSE=0.987262, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:07 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/baseline, best_epoch=1, val_RMSE=1.241226, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:07 KST — v3 study: 정식 학습 시작 ratio_25/lead7_seed42/baseline, PID 222819, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead7_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 01:08 KST — v3 study: 정식 학습 시작 ratio_25/lead7_seed42/p0, PID 222820, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead7_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 01:10 KST — v3 study: 학습 완료 ratio_10/lead7_seed123/baseline, best_epoch=3, val_RMSE=1.015813, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:10 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/baseline, best_epoch=3, val_RMSE=1.272743, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:10 KST — v3 study: 학습 완료 ratio_10/lead7_seed7/p0, best_epoch=1, val_RMSE=1.036861, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:11 KST — v3 study: 정식 학습 시작 ratio_25/lead7_seed7/p0, PID 225058, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead7_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 01:11 KST — v3 study: 정식 학습 시작 ratio_25/lead7_seed7/baseline, PID 225057, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead7_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 01:11 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/p0, best_epoch=1, val_RMSE=1.212450, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:11 KST — v3 study: 정식 학습 시작 ratio_25/lead7_seed123/baseline, PID 225406, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead7_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 01:12 KST — v3 study: 정식 학습 시작 ratio_25/lead7_seed123/p0, PID 225873, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead7_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 01:14 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/p0, best_epoch=2, val_RMSE=1.315208, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:14 KST — v3 study: 학습 완료 ratio_10/lead7_seed123/p0, best_epoch=3, val_RMSE=1.000552, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:14 KST — v3 study: 정식 학습 시작 ratio_25/lead14_seed42/baseline, PID 227326, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead14_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 01:15 KST — v3 study: 정식 학습 시작 ratio_25/lead14_seed42/p0, PID 227582, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead14_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 01:42 KST — v3 study: 학습 완료 ratio_25/lead7_seed42/baseline, best_epoch=3, val_RMSE=0.890116, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:43 KST — v3 study: 정식 학습 시작 ratio_25/lead14_seed7/baseline, PID 245569, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead14_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 01:44 KST — v3 study: 학습 완료 ratio_25/lead7_seed123/baseline, best_epoch=2, val_RMSE=0.881430, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:45 KST — v3 study: 정식 학습 시작 ratio_25/lead14_seed7/p0, PID 246828, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead14_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 01:47 KST — v3 study: 학습 완료 ratio_25/lead7_seed7/baseline, best_epoch=2, val_RMSE=0.886682, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:48 KST — v3 study: 정식 학습 시작 ratio_25/lead14_seed123/baseline, PID 248695, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead14_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 01:55 KST — v3 study: 학습 완료 ratio_25/lead14_seed42/baseline, best_epoch=3, val_RMSE=1.070345, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:55 KST — v3 study: 학습 완료 ratio_25/lead7_seed123/p0, best_epoch=2, val_RMSE=0.874258, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:56 KST — v3 study: 정식 학습 시작 ratio_50/lead7_seed42/baseline, PID 253481, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead7_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 01:56 KST — v3 study: 정식 학습 시작 ratio_25/lead14_seed123/p0, PID 253480, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_25_lead14_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 01:59 KST — v3 study: 학습 완료 ratio_25/lead14_seed42/p0, best_epoch=1, val_RMSE=1.083879, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 01:59 KST — v3 study: 정식 학습 시작 ratio_50/lead7_seed42/p0, PID 255847, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead7_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 02:05 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/p0, best_epoch=7, val_RMSE=1.303823, epochs=19; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:05 KST — v3 study: 학습 완료 ratio_25/lead7_seed42/p0, best_epoch=6, val_RMSE=0.923304, epochs=18; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:05 KST — v3 study: 정식 학습 시작 ratio_50/lead7_seed7/baseline, PID 259463, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead7_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 02:06 KST — v3 study: 정식 학습 시작 ratio_50/lead7_seed7/p0, PID 259706, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead7_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 02:13 KST — v3 study: 학습 완료 ratio_25/lead7_seed7/p0, best_epoch=9, val_RMSE=0.893228, epochs=21; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:13 KST — v3 study: 학습 완료 ratio_25/lead14_seed7/baseline, best_epoch=1, val_RMSE=1.079397, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:14 KST — v3 study: 정식 학습 시작 ratio_50/lead7_seed123/baseline, PID 264540, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead7_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 02:14 KST — v3 study: 정식 학습 시작 ratio_50/lead7_seed123/p0, PID 264818, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead7_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 02:27 KST — v3 study: 학습 완료 ratio_25/lead14_seed123/baseline, best_epoch=2, val_RMSE=1.082230, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:28 KST — v3 study: 정식 학습 시작 ratio_50/lead14_seed42/baseline, PID 273306, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead14_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 02:30 KST — v3 study: 학습 완료 ratio_25/lead14_seed7/p0, best_epoch=1, val_RMSE=1.072164, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:31 KST — v3 study: 정식 학습 시작 ratio_50/lead14_seed42/p0, PID 275375, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead14_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 02:35 KST — v3 study: 학습 완료 ratio_50/lead7_seed42/baseline, best_epoch=2, val_RMSE=0.859824, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:36 KST — v3 study: 정식 학습 시작 ratio_50/lead14_seed7/baseline, PID 278580, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead14_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 02:44 KST — v3 study: 학습 완료 ratio_25/lead14_seed123/p0, best_epoch=2, val_RMSE=1.086505, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:45 KST — v3 study: 정식 학습 시작 ratio_50/lead14_seed7/p0, PID 283902, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead14_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 02:47 KST — v3 study: 학습 완료 ratio_50/lead7_seed7/p0, best_epoch=1, val_RMSE=0.876548, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:48 KST — v3 study: 정식 학습 시작 ratio_50/lead14_seed123/baseline, PID 285972, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead14_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 02:49 KST — v3 study: 학습 완료 ratio_50/lead7_seed7/baseline, best_epoch=7, val_RMSE=0.859159, epochs=19; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:50 KST — v3 study: 정식 학습 시작 ratio_50/lead14_seed123/p0, PID 287339, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_50_lead14_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 02:57 KST — v3 study: 학습 완료 ratio_50/lead7_seed123/baseline, best_epoch=4, val_RMSE=0.853718, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 02:58 KST — v3 study: 정식 학습 시작 ratio_100/lead7_seed42/baseline, PID 292305, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead7_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 03:00 KST — v3 study: 학습 완료 ratio_50/lead7_seed42/p0, best_epoch=7, val_RMSE=0.850605, epochs=19; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:00 KST — v3 study: 정식 학습 시작 ratio_100/lead7_seed42/p0, PID 293603, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead7_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 03:02 KST — v3 study: 학습 완료 ratio_50/lead7_seed123/p0, best_epoch=3, val_RMSE=0.874828, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:03 KST — v3 study: 정식 학습 시작 ratio_100/lead7_seed7/baseline, PID 295419, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead7_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 03:18 KST — v3 study: 학습 완료 ratio_50/lead14_seed42/p0, best_epoch=2, val_RMSE=1.015881, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:19 KST — v3 study: 정식 학습 시작 ratio_100/lead7_seed7/p0, PID 304781, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead7_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 03:20 KST — v3 study: 학습 완료 ratio_50/lead14_seed42/baseline, best_epoch=6, val_RMSE=1.010546, epochs=18; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:21 KST — v3 study: 정식 학습 시작 ratio_100/lead7_seed123/baseline, PID 306402, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead7_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 03:28 KST — v3 study: 학습 완료 ratio_50/lead14_seed123/baseline, best_epoch=4, val_RMSE=1.044238, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:29 KST — v3 study: 정식 학습 시작 ratio_100/lead7_seed123/p0, PID 311268, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead7_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 03:32 KST — v3 study: 학습 완료 ratio_50/lead14_seed7/baseline, best_epoch=7, val_RMSE=1.011772, epochs=19; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:32 KST — v3 study: 정식 학습 시작 ratio_100/lead14_seed42/baseline, PID 313301, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead14_seed42.json, λ=0.0, Rres=0.

- 2026-10-09 03:40 KST — v3 study: 학습 완료 ratio_50/lead14_seed123/p0, best_epoch=3, val_RMSE=1.043021, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:41 KST — v3 study: 정식 학습 시작 ratio_100/lead14_seed42/p0, PID 318391, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead14_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 03:48 KST — v3 study: 학습 완료 ratio_50/lead14_seed7/p0, best_epoch=7, val_RMSE=1.014063, epochs=19; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 03:49 KST — v3 study: 정식 학습 시작 ratio_100/lead14_seed7/baseline, PID 323572, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead14_seed7.json, λ=0.0, Rres=0.

- 2026-10-09 04:05 KST — v3 study: 학습 완료 ratio_100/lead7_seed7/baseline, best_epoch=13, val_RMSE=0.832004, epochs=25; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:05 KST — v3 study: 정식 학습 시작 ratio_100/lead14_seed7/p0, PID 333531, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead14_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 04:15 KST — v3 study: 학습 완료 ratio_100/lead7_seed42/baseline, best_epoch=15, val_RMSE=0.820307, epochs=27; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:15 KST — v3 study: 정식 학습 시작 ratio_100/lead14_seed123/baseline, PID 339464, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead14_seed123.json, λ=0.0, Rres=0.

- 2026-10-09 04:20 KST — v3 study: 학습 완료 ratio_100/lead7_seed123/p0, best_epoch=4, val_RMSE=0.838303, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:21 KST — v3 study: 정식 학습 시작 ratio_100/lead14_seed123/p0, PID 343216, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_100_lead14_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 04:21 KST — v3 study: 학습 완료 ratio_100/lead7_seed123/baseline, best_epoch=9, val_RMSE=0.823478, epochs=21; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:25 KST — v3 study: 학습 완료 ratio_100/lead7_seed42/p0, best_epoch=15, val_RMSE=0.838140, epochs=27; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:30 KST — v3 study: 학습 완료 ratio_100/lead14_seed42/baseline, best_epoch=9, val_RMSE=0.972959, epochs=21; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:32 KST — v3 study: 학습 완료 ratio_100/lead14_seed7/baseline, best_epoch=6, val_RMSE=0.996266, epochs=18; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:40 KST — v3 study: 학습 완료 ratio_100/lead7_seed7/p0, best_epoch=14, val_RMSE=0.843768, epochs=26; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:51 KST — v3 study: 학습 완료 ratio_100/lead14_seed42/p0, best_epoch=13, val_RMSE=0.993949, epochs=25; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:55 KST — v3 study: 학습 완료 ratio_100/lead14_seed123/baseline, best_epoch=11, val_RMSE=0.974374, epochs=23; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 04:56 KST — v3 study: 학습 완료 ratio_100/lead14_seed7/p0, best_epoch=6, val_RMSE=0.995471, epochs=18; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 05:15 KST — v3 study: 학습 완료 ratio_100/lead14_seed123/p0, best_epoch=12, val_RMSE=0.980871, epochs=24; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead7_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead7_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead14_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead14_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead7_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead7_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead14_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead7_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:16 KST — v3 study: 평가 완료 ratio_10/lead7_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_25/lead7_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_25/lead7_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_10/lead14_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_25/lead7_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_25/lead7_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_25/lead7_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_25/lead7_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_10/lead14_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:18 KST — v3 study: 평가 완료 ratio_10/lead14_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_25/lead14_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_50/lead7_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_50/lead7_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_25/lead14_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_25/lead14_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_25/lead14_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_25/lead14_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_25/lead14_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:19 KST — v3 study: 평가 완료 ratio_50/lead7_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead14_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead14_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead14_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead7_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead14_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead14_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead14_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead7_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:21 KST — v3 study: 평가 완료 ratio_50/lead7_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead14_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead7_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead7_seed42/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead7_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead14_seed7/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead7_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead7_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead7_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:23 KST — v3 study: 평가 완료 ratio_100/lead14_seed42/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:24 KST — v3 study: 평가 완료 ratio_100/lead14_seed123/baseline; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:24 KST — v3 study: 평가 완료 ratio_100/lead14_seed7/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:24 KST — v3 study: 평가 완료 ratio_100/lead14_seed123/p0; test_metrics.json/test_fields.npz 저장.

- 2026-10-09 05:24 KST — v3 study: 48개 학습 및48개 best-checkpoint Test 평가 완료. 통계/그림 집계 시작.

- 2026-10-09 05:25 KST — v3 study: v3 정식48회 실험·집계 완료; final_report.md 참조.

- 2026-10-09 10:16 KST — v3 48회 완료 결과 정리: _r1/results_summary_ko.md 및 analysis/posthoc_result_verification.json 신규 생성. 저장 예측48개에서RMSE/일별MSE재계산(최대차5.29e-9K), CSV/개별지표 및validation best선정/loss·gradient finite/비활성손실검증. 8조건중P0평균개선1개(25%/lead14 +0.35%), seed쌍6/24개선. 90일CI및30/180일민감도/MHW/기존결과차이 한글보고. 기존코드/결과수정·추가학습없음.

- 2026-10-09 10:20 KST — 사용자 성능저하 원인질문에 대한 읽기전용 진단: _r1/analysis/loss_balance_diagnosis.csv, bias_diagnosis.csv, diagnosis_notes_ko.md 생성. best epoch weighted physics/data중앙값1.569, 14/24 warmup전best, 6/8조건더찬bias확인. 직선경로/P0근사/냉각전용entrainment/손실균형을미확정원인후보로구분. 소스수정·추가학습·λ탐색없음.

- 2026-10-09 10:26 KST — 사용자성능저하해결요청: scripts/diagnose_p0_alignment.py 신규작성. 먼저10% 6조건 선택train의실제/직선경로잔차·we부호제약·공유gradient충돌을진단. 진단용기간평균비교는새모델학습아님. 기존모델/데이터/Test/결과보존. GPU0제외명시, GPU1유휴확인; process name yeonwoo.

- 2026-10-09 10:30 KST — v3 study: p0_repair_20261009/diagnosis.json: 10%×2lead×3seed의선택train 실제/직선경로budget 및oraclewe제약, 초기/best모델의data/physics predictorgradient진단완료. optimizer/Test평가없음. GPU0미사용.

- 2026-10-09 10:32 KST — v3 study: scripts/train_p0_guarded.py 및test_p0_guarded.py 신규작성/8개단위검사PASS. 선택train진단best모델24batch중12개gradient충돌, nominal.1물리/data norm중앙값1.16455. P0daily/양수we/Rres0/모델/통계유지, 비대칭projection+기존λeff를sharedgradient비율상한으로적용. 10%×2lead×3seed6회검증pilot 사전등록, validation만평가, GPU0금지·1–3만허용. oldprotocol/src/manifest보존.

- 2026-10-09 10:33 KST — v3 study: P0guarded GPU1 smoke PASS: train2batch/backward/2update及validation2batch, clip/finite/Rres0/checkpoint복원. projected cosine>=0, weightedsharedphysics/data비율<=.1확인. smoke모델pilot재사용없음.

- 2026-10-09 10:34 KST — v3 study: scripts/manage_p0_guarded.py 작성: GPU0강제제외, 유휴GPU1–3만사용·메모리여유확인·각최대3개,6회완료후validation만자동집계. 기존Baseline/P0와초기값/manifest/통계/배치해시비교필수. 통과기준미충족시성능해결로표시하지않고자동확장금지. GPU smoke2batch PASS후pilot시작.

- 2026-10-09 10:35 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed7/p0_guarded, PID 575654, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 10:35 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed123/p0_guarded, PID 575652, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 10:35 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed7/p0_guarded, PID 575651, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed7.json, λ=0.1, Rres=0.

- 2026-10-09 10:35 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed42/p0_guarded, PID 575653, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 10:35 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/p0_guarded, PID 575650, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.1, Rres=0.

- 2026-10-09 10:35 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed123/p0_guarded, PID 575655, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed123.json, λ=0.1, Rres=0.

- 2026-10-09 10:38 KST — P0 repair: initial_comparison.json/repair_report_ko.md 생성. 6개 초기값·통계·split·표본tensor hash 기존 B/P0와 모두 일치. NVML GPU1/2 각각3개·yeonwoo 확인, GPU0 미사용. 재사용 worker의 정식/48회Test 문구는 pilot 제어와 무관함을 명시; 6회 validation-only 검증 진행 중, 성능 개선 미판정.

- 2026-10-09 10:40 KST — P0 repair: first_interval_check.json 및 보고서 갱신. 6개×478 update 및validation 완료, finite/guard상한/투영수치오차/Rres0/기존B·P0배치hash 모두PASS. GPU1/2 3개씩 학습 계속 진행, 성능 판정은 완료후 validation 비교까지 보류.

- 2026-10-09 10:49 KST — v3 study: scripts/train_p0_lambda.py/analyze_p0_lambda.py/manage_p0_lambda.py 신규작성. 사용자승인 λ민감도: 원래 scalar P0 유지·λ만변경, 10%×2lead×3seed×3새λ=18회, 기존0/.1 12결과재사용. GPU0제외·1/2/3만사용. 투영pilot와분리, validation선택·Test미평가.

- 2026-10-09 10:51 KST — v3 study: λ검증 사전준비: protocol/DESIGN/configuration_check 생성 및 기존12참조확인PASS. Baseline 비활성physics RHS=None 대 P0 subset RHS 차이를 설정비교에서 명시적으로 구분; P0끼리는RHS일치필수. 아직새학습전 hash갱신. validation_result는12/30 PENDING.

- 2026-10-09 10:51 KST — v3 study: outputs/p0_lambda_v3_20261009/smoke.py 작성/실행: 실제3λ scalar gradient 선형성 및 finite/GPU title/checkpoint 검증. GPU3 사용, 모든캐시Ceph.

- 2026-10-09 10:51 KST — v3 study: λ検証 GPU3 smoke PASS: .001/.01/.03 실제autograd=data_grad+λphysics_grad·finite·Rres無gradient·3update/val2batch/checkpoint復元. formal jobs原seed開始.

- 2026-10-09 10:52 KST — v3 study: native λ18회 validation 학습 supervisor 시작: GPU3부터 최대3개, 기존guarded GPU1/2는건드리지않고종료후빈GPU활용, 전체GPU1–3만허용. scalargradient smoke PASS. 모든참조/후보/선정규칙실행전고정.

- 2026-10-09 10:52 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed123/lambda_0.001, PID 587990, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed123.json, λ=0.001, Rres=0.

- 2026-10-09 10:52 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed7/lambda_0.001, PID 587989, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed7.json, λ=0.001, Rres=0.

- 2026-10-09 10:52 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/lambda_0.001, PID 587988, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.001, Rres=0.

- 2026-10-09 10:53 KST — v3 study: native λ검증 실제시작확인: GPU3 PID587988/587989/587990 모두yeonwoo, λ.001/lead7/seed42·7·123. 시작config의λ/실제RHS/공유초기값/통계/split/표본tensor 일치PASS(startup_check.json). 기존guarded와합쳐GPU1/2/3만사용; 나머지15개대기.

- 2026-10-09 11:28 KST — v3 study: 학습 완료 ratio_10/lead7_seed7/p0_guarded, best_epoch=1, val_RMSE=1.023404, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:31 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/p0_guarded, best_epoch=1, val_RMSE=1.228555, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:34 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/p0_guarded, best_epoch=2, val_RMSE=1.305300, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:34 KST — v3 study: 학습 완료 ratio_10/lead7_seed7/lambda_0.001, best_epoch=1, val_RMSE=0.988260, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:35 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed42/lambda_0.001, PID 619979, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed42.json, λ=0.001, Rres=0.

- 2026-10-09 11:35 KST — v3 study: 학습 완료 ratio_10/lead7_seed123/p0_guarded, best_epoch=3, val_RMSE=0.997249, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:36 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/p0_guarded, best_epoch=3, val_RMSE=1.273788, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:37 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed123/lambda_0.001, PID 621105, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed123.json, λ=0.001, Rres=0.

- 2026-10-09 11:37 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/lambda_0.01, PID 621106, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.01, Rres=0.

- 2026-10-09 11:37 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed7/lambda_0.001, PID 621100, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed7.json, λ=0.001, Rres=0.

- 2026-10-09 11:37 KST — v3 study: 학습 완료 ratio_10/lead7_seed42/p0_guarded, best_epoch=4, val_RMSE=1.048298, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:37 KST — v3 study: P0guarded6회검증완료: technicalPASS, performance=False. validation_result.json/validation_report.md. Test미평가·48조건확장없음.

- 2026-10-09 11:38 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed123/lambda_0.01, PID 622469, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed123.json, λ=0.01, Rres=0.

- 2026-10-09 11:38 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed7/lambda_0.01, PID 622468, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed7.json, λ=0.01, Rres=0.

- 2026-10-09 11:38 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed42/lambda_0.01, PID 622470, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed42.json, λ=0.01, Rres=0.

- 2026-10-09 11:40 KST — v3 study: 학습 완료 ratio_10/lead7_seed123/lambda_0.001, best_epoch=3, val_RMSE=1.007526, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:41 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed7/lambda_0.01, PID 625080, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed7.json, λ=0.01, Rres=0.

- 2026-10-09 11:42 KST — v3 study: 학습 완료 ratio_10/lead7_seed42/lambda_0.001, best_epoch=4, val_RMSE=1.070733, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 11:43 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed123/lambda_0.01, PID 626722, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed123.json, λ=0.01, Rres=0.

- 2026-10-09 12:17 KST — v3 study: 학습 완료 ratio_10/lead7_seed7/lambda_0.01, best_epoch=1, val_RMSE=0.985242, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:18 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed42/lambda_0.03, PID 653836, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json, λ=0.03, Rres=0.

- 2026-10-09 12:19 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/lambda_0.001, best_epoch=1, val_RMSE=1.235924, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:20 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed7/lambda_0.03, PID 655195, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed7.json, λ=0.03, Rres=0.

- 2026-10-09 12:24 KST — v3 study: 학습 완료 ratio_10/lead7_seed123/lambda_0.01, best_epoch=3, val_RMSE=1.001524, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:24 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/lambda_0.01, best_epoch=3, val_RMSE=1.328980, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:25 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/lambda_0.01, best_epoch=1, val_RMSE=1.228941, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:25 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed42/lambda_0.03, PID 659323, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed42.json, λ=0.03, Rres=0.

- 2026-10-09 12:25 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/lambda_0.001, best_epoch=3, val_RMSE=1.272792, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:25 KST — v3 study: 정식 학습 시작 ratio_10/lead7_seed123/lambda_0.03, PID 659322, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed123.json, λ=0.03, Rres=0.

- 2026-10-09 12:25 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed7/lambda_0.03, PID 659916, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed7.json, λ=0.03, Rres=0.

- 2026-10-09 12:25 KST — v3 study: 학습 완료 ratio_10/lead7_seed42/lambda_0.01, best_epoch=4, val_RMSE=1.048115, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:26 KST — v3 study: 정식 학습 시작 ratio_10/lead14_seed123/lambda_0.03, PID 660265, manifest=/ceph_data/yeonwoo/MHW_subset/data/subsets/preprocessing_subset_v3/ratio_10_lead14_seed123.json, λ=0.03, Rres=0.

- 2026-10-09 12:31 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/lambda_0.01, best_epoch=3, val_RMSE=1.298716, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:36 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/lambda_0.001, best_epoch=7, val_RMSE=1.315786, epochs=19; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:50 KST — v3 study: 학습 완료 ratio_10/lead7_seed7/lambda_0.03, best_epoch=1, val_RMSE=0.983056, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:52 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/lambda_0.03, best_epoch=1, val_RMSE=1.221361, epochs=13; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 12:57 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/lambda_0.03, best_epoch=2, val_RMSE=1.318127, epochs=14; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 13:06 KST — v3 study: 학습 완료 ratio_10/lead7_seed42/lambda_0.03, best_epoch=4, val_RMSE=1.078161, epochs=16; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 13:09 KST — v3 study: 학습 완료 ratio_10/lead7_seed123/lambda_0.03, best_epoch=3, val_RMSE=0.996084, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 13:10 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/lambda_0.03, best_epoch=3, val_RMSE=1.311765, epochs=15; Test는 전체48회 학습 완료 후 평가.

- 2026-10-09 13:11 KST — v3 study: Native P0 lambda sensitivity complete: validation_result.json/validation_report.md; no Test inference. {'scope': '10% only; finite candidate set, not a global optimum or independent test proof', 'shared_lambda': 0.01, 'scores': [{'lambda_physics': 0, 'mean_relative_validation_rmse': 1.0}, {'lambda_physics': 0.001, 'mean_relative_validation_rmse': 0.9987137607394915}, {'lambda_physics': 0.01, 'mean_relative_validation_rmse': 0.9978053728944547}, {'lambda_physics': 0.03, 'mean_relative_validation_rmse': 1.0005931130625525}, {'lambda_physics': 0.1, 'mean_relative_validation_rmse': 1.0006274483648725}], 'claim': 'Candidate selected on validation; generalization unverified'}

- 2026-10-09 14:32 KST — native λ18회 완료 결과 확인: completion_audit.json/results_summary_ko.md 생성. 18새+12재사용=30조건 무결성/집계/finite/설정·배치검사PASS, 실패0. 10% validation 공통λ=.01 paired평균개선0.21946%; lead7 .01 +1.066%, lead14 .001 +0.198%. Test미평가·추가학습없음. 별도guarded6회는사전성능기준미충족.

- 2026-10-09 14:38 KST — v3 study: dense_eval_common.py/train_dense_validation.py 작성: 24update validation, 기존478update LR/λ 스케줄과38240 budget/5736 patience 유지. dense validation RNG보존·기존478경계batch/val대조. validation we=0 공통잔차/learnedwe별도. src/기존산출물무수정.

- 2026-10-09 14:39 KST — v3 study: analyze_dense_validation.py/manage_dense_validation.py 작성: 기존30checkpoint 공통물리진단6그룹+새dense12학습, 완료시마다별도집계. GPU1/2/3각최대3개·타사용자GPU배치회피·실패중단. Test미평가.

- 2026-10-09 14:41 KST — v3 study: outputs/p0_dense_validation_20261009 protocol/DESIGN/cpu_checks/smoke.py 작성. 6진단그룹30checkpoint+12dense학습 사전고정. schedule분할·80경계·38240budget·5736patience·RNG CPU검사PASS. GPU smoke전 모든해시고정.

- 2026-10-09 14:41 KST — v3 study: dense validation GPU1 smoke 실행: 두arm 연속26update vs24+평가+2update, 원래학습경로보존 및공통잔차 검사.

- 2026-10-09 14:43 KST — v3 study: dense validation GPU1 smoke PASS: Baseline/P0 continuous26 vs24+validation+2 update 일치·RNG보존·SST평가동일·공통잔차finite·sampler/checkpoint복원.

- 2026-10-09 14:44 KST — v3 study: dense GPU smoke Baseline/P0 모두PASS. supervisor 설명/완료로그를dense실험명으로정정(스케줄무변경), smoke당시protocol별도보존·본실행전hash갱신. 기존30checkpoint진단+12dense학습 시작; GPU1/2/3만허용.

- 2026-10-09 14:44 KST — v3 study: dense validation training start runs/lead7_seed7/baseline; 24-step eval,478-step schedule,38240 max,5736 patience; GPU3

- 2026-10-09 14:44 KST — v3 study: dense validation training start runs/lead7_seed42/baseline; 24-step eval,478-step schedule,38240 max,5736 patience; GPU3

- 2026-10-09 14:44 KST — v3 study: dense validation training start runs/lead7_seed42/p0_lambda_0.01; 24-step eval,478-step schedule,38240 max,5736 patience; GPU3

- 2026-10-09 14:46 KST — v3 study: dense startup_check.json 생성: GPU1/2/3 각3개 PID의yeonwoo확인; 첫3학습24update validation finite·lr5.6e-5·P0 λeff=.002·Rres호출0 PASS. DESIGN에SST-only history의physics0 placeholder를미측정값으로명시, 실제물리비교는별도best_diagnostics만사용.

- 2026-10-09 14:48 KST — v3 study: 共通残差診断完了 diagnostics/lead14_seed123; validation only, we=0 common and learned-we separate.

- 2026-10-09 14:48 KST — v3 study: 共通残差診断完了 diagnostics/lead7_seed7; validation only, we=0 common and learned-we separate.

- 2026-10-09 14:48 KST — v3 study: 共通残差診断完了 diagnostics/lead7_seed123; validation only, we=0 common and learned-we separate.

- 2026-10-09 14:48 KST — v3 study: 共通残差診断完了 diagnostics/lead7_seed42; validation only, we=0 common and learned-we separate.

- 2026-10-09 14:48 KST — v3 study: 共通残差診断完了 diagnostics/lead14_seed42; validation only, we=0 common and learned-we separate.

- 2026-10-09 14:48 KST — v3 study: 共通残差診断完了 diagnostics/lead14_seed7; validation only, we=0 common and learned-we separate.

- 2026-10-09 14:48 KST — v3 study: dense validation training start runs/lead7_seed123/baseline; 24-step eval,478-step schedule,38240 max,5736 patience; GPU1

- 2026-10-09 14:49 KST — v3 study: dense validation training start runs/lead7_seed123/p0_lambda_0.01; 24-step eval,478-step schedule,38240 max,5736 patience; GPU1

- 2026-10-09 14:49 KST — v3 study: dense validation training start runs/lead14_seed42/baseline; 24-step eval,478-step schedule,38240 max,5736 patience; GPU2

- 2026-10-09 14:49 KST — v3 study: dense validation training start runs/lead14_seed7/baseline; 24-step eval,478-step schedule,38240 max,5736 patience; GPU2

- 2026-10-09 14:49 KST — v3 study: dense validation training start runs/lead7_seed7/p0_lambda_0.01; 24-step eval,478-step schedule,38240 max,5736 patience; GPU1

- 2026-10-09 14:49 KST — v3 study: dense validation training start runs/lead14_seed42/p0_lambda_0.01; 24-step eval,478-step schedule,38240 max,5736 patience; GPU2

- 2026-10-09 15:45 KST — coastal_vs_v3_audit_20261009/analyze.py 작성/CPU실행. 과거해안보정 대v3 기존log/checkpoint/predictions·입력분포 감사. 학습/데이터/기존코드수정없음, GPU미사용.

- 2026-10-09 15:53 KST — outputs/coastal_vs_v3_audit_20261009/write_report.py 및 report.md 생성. 기존 해안보정 대비 v3 10%/lead7/3seed의 Baseline+9.1101%, P0+17.7982% 재현; anomaly 분산 축소·loss scale 및 기존 gradient 기록을 구분 분석. 과거도 해안보정/Rres0임을 확인. 기존 데이터/모델/결과 수정 및 새 학습/forward/backward/GPU 실행 없음. 인과기여율·과거best gradient 미확인을 명시.

- 2026-10-09 15:54 KST — coastal_vs_v3_audit_20261009 보고서와 생성기의 표 내부 절댓값 기호 Markdown escaping 수정. 수치 변경 없음.

- 2026-10-09 16:25 KST — v3 study: scripts/train_v3_ablation.py/smoke_v3_ablation.py/manage_v3_ablation.py/analyze_v3_ablation.py 및 outputs/v3_ablation_lead7_20261009 생성. λ요청12조건 기존결과 무결성PASS; .01 선택. all-head4조건=physics3재사용+baseline/aux/full9신규; aux10,478update validation,Test금지. CPU/GPU smoke통과후 기존dense완료대기, GPU1/2/3각최대3개. src/데이터/기존결과 변경없음.

- 2026-10-09 16:27 KST — ablation CPU smoke의 설정비교가 cpu와기존cuda:0 차이를 잡아 중단(학습전). train_v3_ablation.setup에서 실행device만별도검증하도록수정; 기타설정일치요구유지. manager에 CPU PASS gate추가. 실패로그보존, protocol 코드해시재고정.

- 2026-10-09 16:28 KST — v3 study: v3 ablation cpu smoke PASS: four active/inactive loss paths, shared objective gradients, finite updates and checkpoint roundtrip.

- 2026-10-09 16:29 KST — ablation GPU1 smoke 시작: CPU4조건PASS. GPU1 기존본인3작업/여유16GiB이상확인, 짧은smoke1개추가(총4이내); 각arm실제16samples×2updates, processyeonwoo검증. 정식9작업은기존dense완료후대기계획유지.

- 2026-10-09 16:30 KST — v3 study: v3 ablation cuda:0 smoke PASS: four active/inactive loss paths, shared objective gradients, finite updates and checkpoint roundtrip.

- 2026-10-09 16:30 KST — v3_ablation_lead7_20261009 CPU/GPU4조건smoke PASS, preflight.json/md 저장. 실제GPU1 batch16×2/arm, checkpoint/finite/비활성gradNone검증; peak9896MiB, 초기aux10/sharedgradient ratio2277.0기록. Supervisor PID980773 등록: 기존dense완료뒤 GPU1/2/3각최대3개, all-head9학습+physics3재사용. Test금지, 기존src/데이터/결과보존.

- 2026-10-09 16:50 KST — 사용자요청에따라 aux10 ablation supervisor PID980773를 UID/start_ticks 대조후 종료. 신규9회 시작0 확인; 기존dense학습/기존결과보존. 낮은Aux가중치 gradient진단→Validation선택→4조건비교로 변경.

- 2026-10-09 16:51 KST — diagnose_aux_weights.py/aux_weight_selection_20261009 작성. seed42/7/123×8고정batch의 shared data/aux gradient norm·cosine·aux10비율 진단, optimizer없음. 후보는 pooled중앙비율로정하는사전규칙(base/10,base,3base); Validation이전고정.

- 2026-10-09 16:53 KST — v3 study: Auxiliary multi-batch initial gradient audit complete: {'min': 419.66347281164224, 'median': 1115.9514332870867, 'p90': 6766.73361145731, 'max': 15795.8207271596}; candidates=[0.0001, 0.001, 0.003]

- 2026-10-09 16:56 KST — aux_weight_selection 프로토콜고정:24batch aux10 gradient중앙1115.95/P906766.73/max15795.82,16/24충돌. 후보.0001/.001/.003; Aux9+matchedBaseline3 파일럿5×478updates. Validation양수후보공통선택(0baseline우세시효과미입증명시), 선택Aux/Baseline 정확재개+Full3+Physics3재사용하여동일478update4조건비교. 새로운worker/analyzer/manager/smoke작성; src·데이터보존.

- 2026-10-09 16:58 KST — v3 study: auxiliary candidates cuda:0 smoke PASS: four active/inactive loss paths, shared objective gradients, finite updates and checkpoint roundtrip.

- 2026-10-09 16:59 KST — aux_weight_selection_20261009 사전검사PASS:3후보Aux/Full+Baseline/Physics8조건GPU smoke(2batch), finite/비활성경로/체크포인트검증. Validation선택기best/tie/all-worse/불균등예산거부·5epoch이후미참조 회귀검사PASS. Supervisor PID1031205 실행;12개5epoch파일럿후공통양수가중치선택→정확재개/Full→최종4조건자동비교. GPU1/2/3총4process/GPU이내(기존작업포함),Test미사용.

- 2026-10-09 17:00 KST — v3 study: Auxiliary selection start lead7_seed123/baseline_aux0; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 17:00 KST — v3 study: dense validation training complete runs/lead7_seed7/baseline; best_step=478, val_RMSE=0.987262; Test not evaluated.

- 2026-10-09 17:00 KST — v3 study: Auxiliary selection start lead7_seed42/baseline_aux0; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 17:00 KST — v3 study: Auxiliary selection start lead7_seed7/baseline_aux0; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 17:00 KST — v3 study: Auxiliary selection start lead7_seed42/auxiliary_only_aux0.0001; physics=0.0, auxiliary=0.0001, identical heads; Validation only.

- 2026-10-09 17:01 KST — GPU자원조정:aux manager만종료후 coordinated wrapper PID1035615로기존worker인계. 현재시작된파일럿유지, 추가launch는기존dense완료후허용하여독립scheduler동시배치경합방지. 학습protocol/config/hash·체크포인트변경없음. wrapper SHA는coordinated_launch.json기록.

- 2026-10-09 17:02 KST — aux pilot startup_check PASS:실행4개(Baseline3+Aux .0001 seed42), 각실제config/manifest/초기gradient·GPU PID yeonwoo검증, GPU1/2/3총process{'1': 4, '2': 4, '3': 4}. 파일럿Validation선택아직미완료. 추가작업은dense완료후이어짐.

- 2026-10-09 17:03 KST — v3 study: dense validation training complete runs/lead14_seed7/baseline; best_step=456, val_RMSE=1.202407; Test not evaluated.

- 2026-10-09 17:03 KST — v3 study: dense validation training complete runs/lead14_seed42/baseline; best_step=480, val_RMSE=1.246160; Test not evaluated.

- 2026-10-09 17:08 KST — v3 study: dense validation training complete runs/lead7_seed7/p0_lambda_0.01; best_step=478, val_RMSE=0.985242; Test not evaluated.

- 2026-10-09 17:11 KST — v3 study: 학습 완료 lead7_seed42/baseline_aux0, best_epoch=4, val_RMSE=1.065011, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:11 KST — v3 study: dense validation training complete runs/lead14_seed42/p0_lambda_0.01; best_step=480, val_RMSE=1.242897; Test not evaluated.

- 2026-10-09 17:12 KST — v3 study: dense validation training start runs/lead14_seed123/baseline; 24-step eval,478-step schedule,38240 max,5736 patience; GPU2

- 2026-10-09 17:12 KST — v3 study: dense validation training start runs/lead14_seed7/p0_lambda_0.01; 24-step eval,478-step schedule,38240 max,5736 patience; GPU2

- 2026-10-09 17:12 KST — 사용자요청 학습중aux/data gradient추이를 위해 비침습 checkpoint capture observer PID1054814 시작. 기존worker/프로토콜/가중치/평가주기 변경없음. atomic-replace last/best inode를 별도경로hardlink로보존(수정금지), 실제epoch는사후확인. pilot Auxiliary만수집; 유실시점은추정하지않음.

- 2026-10-09 17:12 KST — v3 study: dense validation training start runs/lead14_seed123/p0_lambda_0.01; 24-step eval,478-step schedule,38240 max,5736 patience; GPU2

- 2026-10-09 17:14 KST — CPU observer PID1058598 시작:보존checkpoint마다원래동일8×16 train windows에서aux/data gradient norm비율·cosine측정. seed별CPU초기기준도별도측정, optimizer없음/GPU없음/학습설정·RNG무변경. 실제학습batch gradient와고정probe구분, 유실epoch미측정명시.

- 2026-10-09 17:15 KST — v3 study: 학습 완료 lead7_seed123/baseline_aux0, best_epoch=3, val_RMSE=1.015813, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:16 KST — 비침습 report observer PID1061766 추가:Validation seed/평균±표본SD,loss/clipping곡선,best update,4쌍비교(Full vs Auxiliary포함). Physics기존history도2390update prefix로분리, 최종공통max38240/patience5736과실제earlystop update구분. 기존학습/집계프로토콜파일무변경.

- 2026-10-09 17:16 KST — v3 study: dense validation training complete runs/lead7_seed42/baseline; best_step=1224, val_RMSE=1.021440; Test not evaluated.

- 2026-10-09 17:18 KST — v3 study: dense validation training complete runs/lead7_seed42/p0_lambda_0.01; best_step=1032, val_RMSE=1.023697; Test not evaluated.

- 2026-10-09 17:18 KST — v3 study: 학습 완료 lead7_seed7/baseline_aux0, best_epoch=1, val_RMSE=0.987262, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:18 KST — v3 study: 학습 완료 lead7_seed42/auxiliary_only_aux0.0001, best_epoch=4, val_RMSE=1.068182, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:19 KST — v3 study: dense validation training complete runs/lead7_seed123/baseline; best_step=1368, val_RMSE=0.980604; Test not evaluated.

- 2026-10-09 17:26 KST — v3 study: dense validation training complete runs/lead7_seed123/p0_lambda_0.01; best_step=1368, val_RMSE=0.975409; Test not evaluated.

- 2026-10-09 17:32 KST — 사용자요청 GPU1/2/3 빈슬롯순차충원. 기존전체완료대기 coordinator PID1035615만종료, slots scheduler PID1093309 시작. dense15완료+3실행/미시작0 확인후 매poll1개씩배치, 전체GPU당4개/본study3개 및메모리여유검사. 모든학습소스/프로토콜해시불변, 데이터/체크포인트/관찰작업보존.

- 2026-10-09 17:33 KST — v3 study: Auxiliary selection start lead7_seed7/auxiliary_only_aux0.0001; physics=0.0, auxiliary=0.0001, identical heads; Validation only.

- 2026-10-09 17:33 KST — v3 study: Auxiliary selection start lead7_seed123/auxiliary_only_aux0.0001; physics=0.0, auxiliary=0.0001, identical heads; Validation only.

- 2026-10-09 17:33 KST — v3 study: Auxiliary selection start lead7_seed42/auxiliary_only_aux0.001; physics=0.0, auxiliary=0.001, identical heads; Validation only.

- 2026-10-09 17:34 KST — v3 study: Auxiliary selection start lead7_seed7/auxiliary_only_aux0.001; physics=0.0, auxiliary=0.001, identical heads; Validation only.

- 2026-10-09 17:34 KST — v3 study: Auxiliary selection start lead7_seed123/auxiliary_only_aux0.001; physics=0.0, auxiliary=0.001, identical heads; Validation only.

- 2026-10-09 17:34 KST — v3 study: Auxiliary selection start lead7_seed42/auxiliary_only_aux0.003; physics=0.0, auxiliary=0.003, identical heads; Validation only.

- 2026-10-09 17:35 KST — v3 study: Auxiliary selection start lead7_seed7/auxiliary_only_aux0.003; physics=0.0, auxiliary=0.003, identical heads; Validation only.

- 2026-10-09 17:38 KST — GPU 빈 슬롯 충원 최종 확인: 신규 Auxiliary 파일럿 7개 실행, 기존 파일럿 4개 완료, 1개 대기, 실패 0. GPU1/2/3 총 프로세스 3/4/3개; 모든 신규 PID의 시작 시각 및 yeonwoo 이름 확인. slots scheduler 생존 확인, 다음 빈 슬롯 자동 충원 유지. 증거: outputs/aux_weight_selection_20261009/slot_refill_check.json.

- 2026-10-09 17:47 KST — v3 study: 학습 완료 lead7_seed7/auxiliary_only_aux0.0001, best_epoch=1, val_RMSE=0.988878, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:48 KST — v3 study: Auxiliary selection start lead7_seed123/auxiliary_only_aux0.003; physics=0.0, auxiliary=0.003, identical heads; Validation only.

- 2026-10-09 17:49 KST — v3 study: 학습 완료 lead7_seed42/auxiliary_only_aux0.003, best_epoch=4, val_RMSE=1.078072, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:49 KST — v3 study: 학습 완료 lead7_seed7/auxiliary_only_aux0.001, best_epoch=1, val_RMSE=0.989250, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:49 KST — v3 study: 학습 완료 lead7_seed42/auxiliary_only_aux0.001, best_epoch=4, val_RMSE=1.065256, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:49 KST — v3 study: 학습 완료 lead7_seed123/auxiliary_only_aux0.0001, best_epoch=3, val_RMSE=1.004808, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:49 KST — v3 study: 학습 완료 lead7_seed123/auxiliary_only_aux0.001, best_epoch=3, val_RMSE=1.008040, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:53 KST — v3 study: 학습 완료 lead7_seed7/auxiliary_only_aux0.003, best_epoch=1, val_RMSE=0.993811, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:56 KST — v3 study: 학습 완료 lead7_seed123/auxiliary_only_aux0.003, best_epoch=3, val_RMSE=1.018166, epochs=5; Auxiliary selection stage=pilot; Validation only; Test disabled.

- 2026-10-09 17:56 KST — v3 study: Auxiliary pilot Validation selection: 0.0001; pilot Validation improvement only; not independent generalization evidence

- 2026-10-09 17:57 KST — v3 study: epoch 경계 checkpoint 복구 lead7_seed42/baseline_aux0, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-09 17:57 KST — v3 study: epoch 경계 checkpoint 복구 lead7_seed7/baseline_aux0, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-09 17:57 KST — v3 study: epoch 경계 checkpoint 복구 lead7_seed123/baseline_aux0, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-09 17:58 KST — v3 study: epoch 경계 checkpoint 복구 lead7_seed42/auxiliary_only_aux0.0001, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-09 17:58 KST — v3 study: epoch 경계 checkpoint 복구 lead7_seed7/auxiliary_only_aux0.0001, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-09 17:58 KST — v3 study: epoch 경계 checkpoint 복구 lead7_seed123/auxiliary_only_aux0.0001, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-09 17:59 KST — v3 study: Auxiliary selection start lead7_seed42/full_pinn_aux0.0001; physics=0.01, auxiliary=0.0001, identical heads; Validation only.

- 2026-10-09 18:22 KST — v3 study: 학습 완료 lead7_seed7/baseline_aux0, best_epoch=1, val_RMSE=0.987262, epochs=13; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:23 KST — v3 study: Auxiliary selection start lead7_seed7/full_pinn_aux0.0001; physics=0.01, auxiliary=0.0001, identical heads; Validation only.

- 2026-10-09 18:23 KST — v3 study: 학습 완료 lead7_seed7/auxiliary_only_aux0.0001, best_epoch=1, val_RMSE=0.988878, epochs=13; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:24 KST — v3 study: Auxiliary selection start lead7_seed123/full_pinn_aux0.0001; physics=0.01, auxiliary=0.0001, identical heads; Validation only.

- 2026-10-09 18:28 KST — v3 study: 학습 완료 lead7_seed123/baseline_aux0, best_epoch=3, val_RMSE=1.015813, epochs=15; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:30 KST — v3 study: 학습 완료 lead7_seed123/auxiliary_only_aux0.0001, best_epoch=3, val_RMSE=1.004808, epochs=15; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:30 KST — v3 study: 학습 완료 lead7_seed42/baseline_aux0, best_epoch=4, val_RMSE=1.065011, epochs=16; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:32 KST — v3 study: 학습 완료 lead7_seed42/auxiliary_only_aux0.0001, best_epoch=4, val_RMSE=1.068182, epochs=16; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:48 KST — v3 study: 학습 완료 lead7_seed7/full_pinn_aux0.0001, best_epoch=1, val_RMSE=0.984424, epochs=13; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 18:52 KST — v3 study: 학습 완료 lead7_seed123/full_pinn_aux0.0001, best_epoch=3, val_RMSE=1.006915, epochs=15; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 19:01 KST — v3 study: 학습 완료 lead7_seed42/full_pinn_aux0.0001, best_epoch=4, val_RMSE=1.048134, epochs=16; Auxiliary selection stage=final; Validation only; Test disabled.

- 2026-10-09 19:01 KST — v3 study: Auxiliary selected-weight four-arm comparison complete: [{'arm': 'auxiliary_only', 'reference': 'baseline', 'mean_paired_improvement_percent': 0.2073122073128079, 'improved_seeds': 1, 'by_seed': {42: -0.29774977035723627, 7: -0.16359751055043947, 123: 1.0832839028460994}}, {'arm': 'full_pinn', 'reference': 'physics_only', 'mean_paired_improvement_percent': -0.15236515595919617, 'improved_seeds': 1, 'by_seed': {42: -0.0018848761560663618, 7: 0.08301637355857538, 123: -0.5382269652800975}}, {'arm': 'full_pinn', 'reference': 'baseline', 'mean_paired_improvement_percent': 0.9160190045412427, 'improved_seeds': 3, 'by_seed': {42: 1.5846237107783767, 7: 0.2875024254235581, 123: 0.8759308774217933}}, {'arm': 'physics_only', 'reference': 'baseline', 'mean_paired_improvement_percent': 1.0659072815435693, 'improved_seeds': 3, 'by_seed': {42: 1.586478683776027, 7: 0.20465594981279311, 123: 1.4065872110418876}}]

- 2026-10-09 19:40 KST — v3 study: dense validation training complete runs/lead14_seed123/baseline; best_step=720, val_RMSE=1.198710; Test not evaluated.

- 2026-10-09 19:48 KST — v3 study: dense validation training complete runs/lead14_seed7/p0_lambda_0.01; best_step=456, val_RMSE=1.194311; Test not evaluated.

- 2026-10-09 19:52 KST — v3 study: dense validation training complete runs/lead14_seed123/p0_lambda_0.01; best_step=816, val_RMSE=1.198270; Test not evaluated.

- 2026-10-09 19:52 KST — v3 study: Dense validation complete: outputs/p0_dense_validation_20261009/results.json/report.md; no Test inference. [{'lead': 7, 'paired_improvement_percent': 0.17113287192867005, 'improved_seeds': 2, 'baseline': {'rmse_mean': 0.9964353361398249, 'rmse_sd': 0.02190915083897333, 'original_rmse_mean': 1.0226953089859834, 'common_zero_we_residual_mean': 0.31265337277344407, 'best_steps': [1224, 478, 1368]}, 'p0': {'rmse_mean': 0.9947827346441742, 'rmse_sd': 0.025518454538905236, 'original_rmse_mean': 1.0116269917581153, 'common_zero_we_residual_mean': 0.3004595650787008, 'best_steps': [1032, 478, 1368]}}, {'lead': 14, 'paired_improvement_percent': 0.3239494853751359, 'improved_seeds': 3, 'baseline': {'rmse_mean': 1.215759105371959, 'rmse_sd': 0.026392919952544842, 'original_rmse_mean': 1.277336423470165, 'common_zero_we_residual_mean': 0.31047044454114564, 'best_steps': [480, 456, 720]}, 'p0': {'rmse_mean': 1.211826174588901, 'rmse_sd': 0.02698084042538539, 'original_rmse_mean': 1.2855456077659249, 'common_zero_we_residual_mean': 0.2997637378850697, 'best_steps': [480, 456, 816]}}]

- 2026-10-09 21:17 KST — v3 study: v3 generalization evaluation protocol frozen:12 Validation-best checkpoints; no training;90-day paired circular bootstrap4000,30/180 sensitivity; unchanged MHW; GPU1 inference.

- 2026-10-09 21:22 KST — scripts/evaluate_v3_generalization.py / analyze_v3_generalization.py 작성: 기존12개Validation-best고정 추론 및90일4000회 pairedbootstrap/MHW/표그림. 첫Validation추론후 출력dict split중복오류 수정; 수치계산변경없음, Test미실행상태에서 protocol sourcehash갱신, 초기protocol/실패/metrics보존. Validation부터재검증.

- 2026-10-09 21:22 KST — v3 study: v3 generalization evaluation protocol frozen:12 Validation-best checkpoints; no training;90-day paired circular bootstrap4000,30/180 sensitivity; unchanged MHW; GPU1 inference.

- 2026-10-09 21:23 KST — v3일반화 분석코드/후속계획코드 sourcehash고정. pairedbootstrap 인덱스 재현·동일모델0차이·상수차이·pooledSSE검사PASS. 후속학습은계획만하며실행안함. GPU1추론 프로세스1406004 yeonwoo확인.

- 2026-10-09 21:27 KST — 일반화 평가 보조 코드 추가·검토: scripts/compare_v3_generalization_prior.py(기존λ=.1설정/예측비교·6RQ), scripts/verify_v3_generalization.py(체크포인트보존·36필드수치·통계산출물무결성). Validation12개전부재현PASS,최대오차3.6e-9K. Test추론진행;후속학습실행없음.

- 2026-10-09 21:30 KST — v3 study: v3 generalization:12/12 checkpoints Validation reproduced; fixed eval-mode Train/Test inference complete; original checkpoints preserved.

- 2026-10-09 21:31 KST — 일반화보고서의MHW함수line인용을실제정의107/128로정정. 분석수식·기준·결과변경없음; 분석sourcehash초기본보존후갱신.

- 2026-10-09 21:31 KST — v3 일반화 평가 완료:12 Validation-best checkpoint의 Train/Val/Test 고정추론,36개필드/12개MHW/90일4000회부트스트랩(30/180민감도),4그림,6RQ보고 및후속계획 저장. 신규학습/데이터변경/Test재선택없음. 무결성PASS. Test RMSE baseline=1.040660, physics_only=1.029241, auxiliary_only=1.040006, full=1.028102. 경로 outputs/v3_generalization_20261009/report.md; 평가/분석/계획/비교/검증 스크립트 sourcehash final_integrity.json. 탐색적평가로한정.

- 2026-10-09 21:32 KST — v3 일반화 평가 완료:12 Validation-best checkpoint의 Train/Val/Test 고정추론,36개필드/12개MHW/90일4000회부트스트랩(30/180민감도),4그림,6RQ보고 및후속계획 저장. 신규학습/데이터변경/Test재선택없음. 무결성PASS. Test RMSE baseline=1.040660, physics_only=1.029241, auxiliary_only=1.040006, full=1.028102. 경로 outputs/v3_generalization_20261009/report.md; 평가/분석/계획/비교/검증 스크립트 sourcehash final_integrity.json. 탐색적평가로한정.

- 2026-10-09 22:04 KST — v3 study: Fixedλ=.01 ratio/lead:reuse audit 9/48, new39; all48manifest prepared; GPU1/2/3 only; no Test before global48checkpoint gate.

- 2026-10-09 22:08 KST — v3 study: Fixedλ=.01 all24condition preflight PASS; memory peak9908.0MiB; no changes to raw/v3/model/loss equations; production authorized scope39 new jobs.

- 2026-10-09 22:14 KST — 고정λ=.01 자료량/Lead확장 실행준비완료 및감독프로세스시작 PID1476109:9개엄격재사용/39개신규,24조건사전검사PASS,기존native all-head trainer호출. prepare/preflight/worker/manage/evaluate/analyze_ratio001 스크립트와operatorhash고정. GPU1/2/3,각최대4+4GiBreserve+1.2배smokepeak검사,매20초1개충원. 전체48Validation-best고정후에만Test및통계/5그림자동실행. Metrics마스크/SSE검사와bootstrap회귀PASS. 기존raw/v3/checkpoint보존.

- 2026-10-09 22:15 KST — v3 study: Ablation start ratio_10/lead14_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:15 KST — v3 study: Ablation start ratio_10/lead14_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:15 KST — v3 study: Ablation start ratio_10/lead14_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:16 KST — v3 study: Ablation start ratio_25/lead7_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:16 KST — v3 study: Ablation start ratio_25/lead7_seed42/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:16 KST — v3 study: Ablation start ratio_25/lead7_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:16 KST — 고정λ=.01확장 실행확인:기존9검증완료/신규완료0/실행7/대기32/실패0. GPU가시프로세스모두yeonwoo,PID시작시각확인,실행된config는동일heads/λ0또는.01/Rres0. launch_verification.json 및progress.md저장. Test는전체48학습확정전금지상태유지.

- 2026-10-09 22:17 KST — v3 study: Ablation start ratio_25/lead7_seed7/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:17 KST — v3 study: Ablation start ratio_25/lead7_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:17 KST — v3 study: Ablation start ratio_25/lead7_seed123/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:18 KST — v3 study: Ablation start ratio_25/lead14_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:18 KST — v3 study: Ablation start ratio_25/lead14_seed42/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-09 22:18 KST — v3 study: Ablation start ratio_25/lead14_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.


### 2026-10-10 17:05 KST — 홈 디스크 정리 및 Ceph 이전
- 선행 완료: .vscode-server 검증 이전 및 링크 연결 (최초 홈 약 4.9GiB).
- 대상: /home/yeonwoo → /ceph_data/yeonwoo/_home_storage
- 변경: 재생성 가능한 VS Code CachedExtensionVSIXs 삭제; .local, .cache, .config, .copilot, .dotnet, .rest-client, .codex/.tmp, .codex/cache, .codex/plugins, .codex/skills, .codex/attachments, .codex/shell_snapshots 이전 후 원래 경로에 심볼릭 링크 연결. 기존 Ceph .codex/.vscode-server는 병합하거나 덮어쓰지 않음.
- 이유: 사용자 요청에 따른 홈 로컬 디스크 경량화.
- 검증: 각 이전 디렉터리 rsync checksum 비교 통과. 홈 실제 할당량 333475840 → 71127040 bytes. 활성 Codex SQLite, sessions, tmp 및 셸 기본 파일은 로컬 유지.

- 2026-10-10 18:09 KST — GPU 자원 변경: 사용자 지시에 따라 GPU0/1/2, 각 최대3개(총9개), 4GiB reserve 및 1.2배 peak admission 유지. scripts/*ratio001*_gpu012.py 새 운영 사본과 resource_policy_gpu012.json 생성; 원본 protocol/model/trainer/operator/checkpoint 보존. 원래 code SHA 검증 유지. 손상된 last.pt job8/16은 승인 전 보류, 나머지는 epoch 경계 재개 예정.

- 2026-10-10 18:11 KST — v3 study: GPU012 재개 감사: 재사용9개 best SHA 일치; 중단12개 중10개 last/model/optimizer/config/history/batch-chain finite 및 일치 확인(5~7 epoch 완료); job8/16 last.pt 손상으로 보류. 빈 원본 queue_state 및 supervisor launch 기록을 recovery_gpu012에 복사 보존하고 queue 재구성. 원본 모델/학습설정 불변.

- 2026-10-10 18:12 KST — GPU012 재개 감독프로세스 PID230306 시작: 각최대3(총9), 검증된10개 checkpoint 재개/27개미시작 순차실행, 기존9개재사용. 손상2개(job8/16) 보류, 전체48개완료전 Test gate 유지. scheduler 8검사 PASS.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_10/lead14_seed42/baseline, next_epoch=8, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_10/lead14_seed123/baseline, next_epoch=7, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed42/physics_only, next_epoch=7, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed42/baseline, next_epoch=7, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed7/baseline, next_epoch=7, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed7/physics_only, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed123/physics_only, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-10 18:15 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead14_seed42/baseline, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-10 18:16 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead14_seed42/physics_only, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-10 18:16 KST: GPU0/1/2 각3개, 총9개 yeonwoo 프로세스 확인. 실행된 9개 모두 체크포인트 복원 완료. 재사용완료9/실행9/대기28/손상보류2/새오류0. GPU012 정책 및원본학습코드 SHA 확인 PASS.

- 2026-10-10 18:21 KST — 사용자 승인: 손상 Baseline job8(10%/L14/seed7), job16(25%/L7/seed123)를 /ceph_data/yeonwoo/MHW_subset/outputs/v3_physics_ratio_lead_lambda001_20261009/recovery_gpu012/approved_restart_20261010_182029/runs 아래로 보존 이동. 이동 전후 모든 파일 크기/SHA256 일치 검사 및 기존 감사의 best/last SHA 재확인. 원래 경로에 동일 config.json만 복사하여 epoch1 재시작, 모델/seed/manifest/optimizer/예산/protocol 불변. 기존9개 학습 PID 유지, blocked 해제하고 두 작업을 다음 빈 슬롯 우선 배치. 감사 JSON timestamp 직렬화 오류는 기록 사본을 보존하고 수정; 학습 코드 영향 없음.

- 2026-10-10 18:22 KST — 승인된2개 재실행 대기열 최종검증 PASS: 감독 PID238366 기존9개 PID 모두 인계, GPU0/1/2 각3개 yeonwoo 확인. 재사용완료9/실행9/대기30/보류0/새오류0. job8/16 다음 두 빈슬롯 순서; 아직 재실행 시작 아님. approved_restart_audit.json 및 approved_restart_verification.json 저장.

- 2026-10-10 18:29 KST — 사용자 지시에 따라 GPU0만 사용하도록 변경. 감독프로세스 중단 후 GPU1/2의 6개 사용자 작업을 안정된 last/best/config/history 검사 후 종료하여 GPU0 대기열로 환원. GPU0 기존 3개 PID 유지. 저장 완료 epoch의 optimizer/RNG/batch stream 복원, 미완료 epoch는 재계산됨. 학습 설정/모델 변경 없음. 이전 운영기록 resource_switch_gpu0에 보존, gpu0_migration_audit.json 저장.

- 2026-10-10 18:30 KST — scripts/*ratio001*_gpu0.py 운영 사본 및 resource_policy_gpu0.json 생성: 학습/평가 GPU0만 허용, 최대3개, 4GiB reserve/1.2 peak 유지. 원본 코드와 protocol 보존. 스케줄러 제한 및 whitelist 회귀검사 PASS.

- 2026-10-10 18:30 KST — GPU0 전용 감독 PID250525 시작. 기존 GPU0 학습3개 인계, GPU1/2에서 중단한6개 포함36개 대기. 모델/원본 protocol/code SHA 불변 확인; 이후 학습 및 평가 GPU0만, 최대3개.

- 2026-10-10 18:31 KST — GPU0-only 최종검증 PASS: 감독PID250525 생존, GPU0 기존3개 PID 유지 및 yeonwoo 이름/NVML UUID 검증, GPU1/2 중단6개 NVML 해제 확인. 재사용완료9/실행3/대기36/오류0. gpu0_launch_verification.json 저장.

- 2026-10-10 18:37 KST — v3 study: 학습 완료 ratio_10/lead14_seed42/baseline, best_epoch=2, val_RMSE=1.318040, epochs=14; Ablation Validation only; Test disabled.

- 2026-10-10 18:38 KST — v3 study: Ablation start ratio_10/lead14_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 18:43 KST — v3 study: 학습 완료 ratio_25/lead7_seed42/physics_only, best_epoch=3, val_RMSE=0.890113, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 18:44 KST — v3 study: epoch 경계 checkpoint 복구 ratio_10/lead14_seed123/baseline, next_epoch=11, 설정/manifest 해시 확인.

- 2026-10-10 18:46 KST — v3 study: 학습 완료 ratio_25/lead14_seed42/baseline, best_epoch=3, val_RMSE=1.070345, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 18:48 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed42/baseline, next_epoch=12, 설정/manifest 해시 확인.

- 2026-10-10 19:00 KST — v3 study: 학습 완료 ratio_25/lead7_seed42/baseline, best_epoch=3, val_RMSE=0.890116, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 19:01 KST — v3 study: 학습 완료 ratio_10/lead14_seed123/baseline, best_epoch=3, val_RMSE=1.272743, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 19:01 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed7/baseline, next_epoch=11, 설정/manifest 해시 확인.

- 2026-10-10 19:01 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed7/physics_only, next_epoch=10, 설정/manifest 해시 확인.

- 2026-10-10 19:13 KST — v3 study: 학습 완료 ratio_25/lead7_seed7/baseline, best_epoch=2, val_RMSE=0.886682, epochs=14; Ablation Validation only; Test disabled.

- 2026-10-10 19:14 KST — v3 study: Ablation start ratio_25/lead7_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 19:20 KST — v3 study: 학습 완료 ratio_10/lead14_seed7/baseline, best_epoch=1, val_RMSE=1.241226, epochs=13; Ablation Validation only; Test disabled.

- 2026-10-10 19:20 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead7_seed123/physics_only, next_epoch=10, 설정/manifest 해시 확인.

- 2026-10-10 19:38 KST — v3 study: 학습 완료 ratio_25/lead7_seed7/physics_only, best_epoch=9, val_RMSE=0.880135, epochs=21; Ablation Validation only; Test disabled.

- 2026-10-10 19:39 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead14_seed42/physics_only, next_epoch=10, 설정/manifest 해시 확인.

- 2026-10-10 19:42 KST — v3 study: 학습 완료 ratio_25/lead7_seed123/physics_only, best_epoch=4, val_RMSE=0.876772, epochs=16; Ablation Validation only; Test disabled.

- 2026-10-10 19:42 KST — v3 study: epoch 경계 checkpoint 복구 ratio_25/lead14_seed7/baseline, next_epoch=6, 설정/manifest 해시 확인.

- 2026-10-10 19:54 KST — v3 study: 학습 완료 ratio_25/lead7_seed123/baseline, best_epoch=2, val_RMSE=0.881430, epochs=14; Ablation Validation only; Test disabled.

- 2026-10-10 19:55 KST — v3 study: Ablation start ratio_25/lead14_seed7/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 19:57 KST — v3 study: 학습 완료 ratio_25/lead14_seed42/physics_only, best_epoch=3, val_RMSE=1.069818, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 19:58 KST — v3 study: Ablation start ratio_25/lead14_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:06 KST — v3 study: 학습 완료 ratio_25/lead14_seed7/baseline, best_epoch=1, val_RMSE=1.079397, epochs=13; Ablation Validation only; Test disabled.

- 2026-10-10 20:06 KST — v3 study: Ablation start ratio_25/lead14_seed123/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:26 KST — 사용자 지시에 따라 GPU0/1/2 사용 재개, 각최대3(총9) 및 메모리검사 유지. GPU1/2 유휴 확인. 감독만 PID250525→343677 교체, 기존 GPU0 학습3개 PID 유지. 기존 검증된 gpu012 실행기 재사용, 모델/원본 protocol/code SHA 불변. 전환전 완료21(기존재사용포함)/실행3/대기24. 이전 운영상태 /ceph_data/yeonwoo/MHW_subset/outputs/v3_physics_ratio_lead_lambda001_20261009/resource_return_gpu012_20261010_202656 보존.

- 2026-10-10 20:27 KST — v3 study: Ablation start ratio_50/lead7_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:27 KST — v3 study: Ablation start ratio_50/lead7_seed42/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:28 KST — GPU0/1/2 재전환 검증 PASS. 기존 GPU0 PID3개 유지, GPU1/2 추가 작업 NVML yeonwoo 확인. GPU별{'0': 3, '1': 2, '2': 2}, 각최대3 제한 유지, 완료21/실행·로딩7/대기20/오류0. /ceph_data/yeonwoo/MHW_subset/outputs/v3_physics_ratio_lead_lambda001_20261009/resource_return_gpu012_20261010_202656/verification.json 저장.

- 2026-10-10 20:28 KST — v3 study: Ablation start ratio_50/lead7_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:28 KST — v3 study: Ablation start ratio_50/lead7_seed7/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:28 KST — v3 study: Ablation start ratio_50/lead7_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:29 KST — v3 study: Ablation start ratio_50/lead7_seed123/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:37 KST — v3 study: 학습 완료 ratio_25/lead14_seed123/baseline, best_epoch=2, val_RMSE=1.082230, epochs=14; Ablation Validation only; Test disabled.

- 2026-10-10 20:38 KST — v3 study: Ablation start ratio_50/lead14_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:38 KST — v3 study: 학습 완료 ratio_25/lead14_seed7/physics_only, best_epoch=1, val_RMSE=1.075733, epochs=13; Ablation Validation only; Test disabled.

- 2026-10-10 20:39 KST — v3 study: Ablation start ratio_50/lead14_seed42/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 20:57 KST — v3 study: 학습 완료 ratio_25/lead14_seed123/physics_only, best_epoch=3, val_RMSE=1.058112, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 20:58 KST — v3 study: Ablation start ratio_50/lead14_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:09 KST — v3 study: 학습 완료 ratio_50/lead7_seed42/baseline, best_epoch=2, val_RMSE=0.859824, epochs=14; Ablation Validation only; Test disabled.

- 2026-10-10 21:10 KST — v3 study: Ablation start ratio_50/lead14_seed7/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:15 KST — v3 study: 학습 완료 ratio_50/lead7_seed123/physics_only, best_epoch=3, val_RMSE=0.855912, epochs=15; Ablation Validation only; Test disabled.

- 2026-10-10 21:16 KST — v3 study: Ablation start ratio_50/lead14_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:17 KST — v3 study: 학습 완료 ratio_50/lead7_seed123/baseline, best_epoch=4, val_RMSE=0.853718, epochs=16; Ablation Validation only; Test disabled.

- 2026-10-10 21:18 KST — v3 study: Ablation start ratio_50/lead14_seed123/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:20 KST — v3 study: 학습 완료 ratio_50/lead7_seed7/baseline, best_epoch=7, val_RMSE=0.859159, epochs=19; Ablation Validation only; Test disabled.

- 2026-10-10 21:21 KST — v3 study: Ablation start ratio_100/lead7_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:28 KST — v3 study: 학습 완료 ratio_50/lead7_seed42/physics_only, best_epoch=8, val_RMSE=0.852128, epochs=20; Ablation Validation only; Test disabled.

- 2026-10-10 21:29 KST — v3 study: Ablation start ratio_100/lead7_seed42/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:30 KST — v3 study: 학습 완료 ratio_50/lead14_seed42/baseline, best_epoch=6, val_RMSE=1.010546, epochs=18; Ablation Validation only; Test disabled.

- 2026-10-10 21:31 KST — v3 study: Ablation start ratio_100/lead7_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:33 KST — v3 study: 학습 완료 ratio_50/lead7_seed7/physics_only, best_epoch=9, val_RMSE=0.852702, epochs=21; Ablation Validation only; Test disabled.

- 2026-10-10 21:34 KST — v3 study: Ablation start ratio_100/lead7_seed7/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:36 KST — v3 study: 학습 완료 ratio_50/lead14_seed42/physics_only, best_epoch=6, val_RMSE=1.012350, epochs=18; Ablation Validation only; Test disabled.

- 2026-10-10 21:37 KST — v3 study: Ablation start ratio_100/lead7_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 21:58 KST — v3 study: 학습 완료 ratio_50/lead14_seed7/baseline, best_epoch=7, val_RMSE=1.011772, epochs=19; Ablation Validation only; Test disabled.

- 2026-10-10 22:01 KST — v3 study: Ablation start ratio_100/lead7_seed123/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:05 KST — v3 study: 학습 완료 ratio_50/lead14_seed123/baseline, best_epoch=4, val_RMSE=1.044238, epochs=16; Ablation Validation only; Test disabled.

- 2026-10-10 22:06 KST — v3 study: Ablation start ratio_100/lead14_seed42/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:11 KST — v3 study: 학습 완료 ratio_50/lead14_seed123/physics_only, best_epoch=4, val_RMSE=1.043514, epochs=16; Ablation Validation only; Test disabled.

- 2026-10-10 22:12 KST — v3 study: Ablation start ratio_100/lead14_seed42/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:12 KST — v3 study: 학습 완료 ratio_50/lead14_seed7/physics_only, best_epoch=7, val_RMSE=1.012765, epochs=19; Ablation Validation only; Test disabled.

- 2026-10-10 22:13 KST — v3 study: Ablation start ratio_100/lead14_seed7/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:44 KST — v3 study: 학습 완료 ratio_100/lead7_seed42/baseline, best_epoch=15, val_RMSE=0.820307, epochs=27; Ablation Validation only; Test disabled.

- 2026-10-10 22:44 KST — v3 study: 학습 완료 ratio_100/lead7_seed123/baseline, best_epoch=9, val_RMSE=0.823478, epochs=21; Ablation Validation only; Test disabled.

- 2026-10-10 22:44 KST — v3 study: Ablation start ratio_100/lead14_seed7/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:45 KST — v3 study: Ablation start ratio_100/lead14_seed123/baseline; physics=0.0, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:50 KST — v3 study: 학습 완료 ratio_100/lead7_seed7/baseline, best_epoch=13, val_RMSE=0.832004, epochs=25; Ablation Validation only; Test disabled.

- 2026-10-10 22:51 KST — v3 study: Ablation start ratio_100/lead14_seed123/physics_only; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-10 22:53 KST — v3 study: 학습 완료 ratio_100/lead7_seed42/physics_only, best_epoch=15, val_RMSE=0.809203, epochs=27; Ablation Validation only; Test disabled.

- 2026-10-10 22:55 KST — v3 study: 학습 완료 ratio_100/lead7_seed7/physics_only, best_epoch=13, val_RMSE=0.826831, epochs=25; Ablation Validation only; Test disabled.

- 2026-10-10 23:05 KST — v3 study: 학습 완료 ratio_100/lead14_seed7/baseline, best_epoch=6, val_RMSE=0.996266, epochs=18; Ablation Validation only; Test disabled.

- 2026-10-10 23:05 KST — v3 study: 학습 완료 ratio_100/lead14_seed42/baseline, best_epoch=9, val_RMSE=0.972959, epochs=21; Ablation Validation only; Test disabled.

- 2026-10-10 23:10 KST — v3 study: 학습 완료 ratio_100/lead14_seed42/physics_only, best_epoch=9, val_RMSE=0.972486, epochs=21; Ablation Validation only; Test disabled.

- 2026-10-10 23:20 KST — v3 study: 학습 완료 ratio_100/lead7_seed123/physics_only, best_epoch=13, val_RMSE=0.817875, epochs=25; Ablation Validation only; Test disabled.

- 2026-10-10 23:23 KST — v3 study: 학습 완료 ratio_100/lead14_seed7/physics_only, best_epoch=6, val_RMSE=0.990033, epochs=18; Ablation Validation only; Test disabled.

- 2026-10-10 23:45 KST — v3 study: 학습 완료 ratio_100/lead14_seed123/baseline, best_epoch=11, val_RMSE=0.974374, epochs=23; Ablation Validation only; Test disabled.

- 2026-10-10 23:50 KST — v3 study: 학습 완료 ratio_100/lead14_seed123/physics_only, best_epoch=11, val_RMSE=0.968271, epochs=23; Ablation Validation only; Test disabled.

- 2026-10-10 23:51 KST — v3 study: Fixedλ=.01 all48 Validation-best checkpoints frozen; Test inference gate PASS.

- 2026-10-10 23:54 KST — 사용자 결과 확인: 고정λ=.01 48개 학습 완료 및 Validation-best48 SHA256 재검증 PASS. 모든 run patience12 정상종료(13~27 epoch,6214~12906 updates), 부정상 early exit 없음. Validation 8조건중6개/24 seed쌍중19개 개선. Test평가 6/48 완료로 진행 중, 최종결과와 구분하여 validation_interim_summary.csv / validation_interim_report.md / validation_interim_audit.json 저장. 학습·설정 변경 없음.

- 2026-10-11 00:11 KST — Fixedλ=.01資料量/Lead48条件完了:新規39,再利用9; Test平均改善5/8条件。90日4000回bootstrap/30・180日感度/気候値診断/MHW/5図/8RQ報告作成。既存Testの探索的再現評価。outputs/v3_physics_ratio_lead_lambda001_20261009/report.md。

- 2026-10-11 00:11 KST — v3 study: Fixedλ=.01 ratio/lead all48training/evaluation/report complete.

- 2026-10-11 00:12 KST — 최종 결과 확인: 48평가 완료/분석완료/무결성PASS. 48개 metrics.json과 최종8조건집계 독립 대조 PASS. Test5/8조건 및14/24seed쌍 개선; 90일CI는10%L7개선/50%L14악화만0제외,30/180일도유지. Lead7 10%−100%+1.140%p(CI+.311~+2.002), 전체자료량 단조추세없음. MHW CSI/POD 개선3조건 모두 FAR악화. final_summary_ko.md 및 final_results_review.json 저장. 신규학습/설정변경없음.

- 2026-10-11 00:58 KST — v3 study: Adaptive pilot initialized: new outputs/v3_adaptive_physics_pilot_20261011; two cases50%L14/10%L7,3seeds;12checkpoint reuse strict PASS; source/physics audit recorded. New files only; native Fixed unchanged. GPU1/2/3 per latest attachment, max4 but plan2 training each subject to smoke memory. No Test/8-condition expansion.

- 2026-10-11 01:03 KST — Adaptive pilot 새 전용 scripts9개 작성 및 SHA고정; 기존src/Fixedtrainer 해시 불변. EMA .9, rho .25,100update,lambda[.0001,.03],initial .01 사전고정. GPU1/2/3 단계실행 감독PID652655: 관측residual6→gradient6→preflight6→조건통과시에만Adaptive학습6→Train/Validation평가6→보고. 실제 Test호출없음, 전체확장없음. 진단/사전검사 GPU당1,본학습계획2/상한4 및메모리검사. 초기가중치checkpoint 없음은N/A; gradient32batch 반복샘플 명시.

- 2026-10-11 01:05 KST — GPU1/2/3 다른사용자 점유로 신규GPU admission 대기. 실행전(모든attempt0) 운영계획 보완: native residual진단은 검증된 Fixed best Training fields의we_raw/target_index/파일SHA를 재사용하여 CPU2thread 순차실행; 원래 물리함수 재사용. protocol/newdiaghash만 실행전 갱신하고 기존버전 pre_execution_cpu_amendment에 복사보존. Gradient/GPU smoke/학습은 GPU가용성대기, 다른사용자 방해 없음.

- 2026-10-11 01:09 KST — v3 study: Adaptive phase residual gate PASS; outputs /ceph_data/yeonwoo/MHW_subset/outputs/v3_adaptive_physics_pilot_20261011.

- 2026-10-11 01:09 KST — GPU1/2/3 타사용자 고부하 지속. 독립적으로 진행 가능한 Fixed best/last gradient진단도 CPU fp32/2thread/단일worker로 실행하도록 변경(32배치, optimizer없음). GPU smoke와 실제학습은 자원대기 유지. gradient 실행전 및 adaptive checkpoint생성전 protocol/newdiaghash 갱신; 이전파일 cpu_gradient_amendment 보존. 기존 Fixed 코드/원본 checkpoint 불변. 진단device CSV에기록하여 GPU기반 Adaptive진단과 미세연산차이 한계 명시.

- 2026-10-11 01:15 KST — Adaptive pilot reporting correction: scripts/analyze_adaptive_pilot.py compares both Fixed/Adaptive best gradient ratios with actual checkpoint warmup factors, preserves nominal/base columns and CPU/GPU precision caveat; reads existing Fixed best-epoch physics metrics for residual comparison. Prior reporting source/operator hashes copied to outputs/v3_adaptive_physics_pilot_20261011/reporting_amendment; operator hash refreshed. AST and synthetic warmup-ratio regression PASS (reporting_regression.json). No objective, protocol, native source, checkpoint or running diagnostic changed.

- 2026-10-11 01:16 KST — Adaptive pilot progress report saved: outputs/v3_adaptive_physics_pilot_20261011/progress_report.md. 6/6 residual checks PASS; native learned-we mean residual RMSE 50%L14=.233831,10%L7=.252615 K/day, daily-observed=.409517/.404834; no clipping. CPU gradient diagnosis running; GPU1/2/3 occupied by another user, smoke/train pending. Source/operator hashes and new scripts AST rechecked PASS. No new Test inference.

- 2026-10-11 01:16 KST — Adaptive Fixed Gradient diagnosis first condition PASS,50%L14seed42: best32batch ratio median .066384,cosine mean .091937,14/32conflicts; last32batch .101384,.233247,6/32conflicts. No optimizer step/parameter changes. Detailed files diagnostics/0/gradient_*.csv and progress_report.md. Other5conditions continue; no aggregate causal conclusion.

- 2026-10-11 01:48 KST — v3 study: Adaptive phase gradient gate PASS; outputs /ceph_data/yeonwoo/MHW_subset/outputs/v3_adaptive_physics_pilot_20261011.

- 2026-10-11 01:49 KST — 사용자 요청으로 Adaptive GPU 단계 연기: 본인 UID/start_ticks로 확인한 실행 관리자 PID662021만 중지, 활성 worker 없음 확인. Residual6/6 및 best/last Gradient384batch 완료 보존. GPU smoke/신규 Adaptive 학습0, 다른 사용자 프로세스 미접촉. user_pause.json 및 queue_state.json에 사용자 보류 기록; 이후 GPU가 비어도 자동 실행되지 않음.

- 2026-10-11 02:01 KST — CPU-only analysis: audit PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:03 KST — CPU-only 분석 시작: 새 scripts/cpu_diagnostics_common.py,cpu_artifact_audit.py,cpu_physics_diagnostics.py,cpu_offline_adaptive.py,cpu_mhw_statistics.py 및 outputs/v3_cpu_diagnostics_20261011 생성. CUDA_VISIBLE_DEVICES 비움, CPU2thread/주소공간16GiB/단일worker, 기존GPUqueue 중지유지. 저장된144예측/w_e 재사용; native RHS 누적 및오차관계, 기존5일MHW기준 재현, rho후보 오프라인, centered bootstrap/Holm 설계. 기존모델·전처리·checkpoint 변경 및신규추론/backward/학습 없음. AST PASS; 실행검증 진행.

- 2026-10-11 02:04 KST — scripts/cpu_climatology_diagnostics.py 추가: 기존12climatology/24manifest의 DOY coverage, roughness, singleton-zero, 원자료·anomaly분산/분할분포 검증. 기존전수 z-score스캔은 v2/v3통계·climatology 동일성 확인 후 재사용하고7anomaly분포를 재계산 대조; static3채널 포함. 재학습/기후값수정 없음, AST PASS.

- 2026-10-11 02:07 KST — scripts/cpu_diagnostics_report.py 추가:7항목종합,matched7/14날짜 비교,기존6조건daily결과회귀,보호입력최종해시검사,후속적분loss/신뢰도loss 설계문서. code_provenance.json 기록 및새CPU스크립트AST PASS. 통계결과에따른현재실험설정변경 없음.

- 2026-10-11 02:09 KST — CPU 통계 실행전 검증 추가: cpu_mhw_statistics.py에서 신규 weighted bootstrap4000개가 기존 saved index-bootstrap 원시재표본과 일치하는지 검사하고 알려진 contingency4/2/1/8의CSI/POD/FAR를 확인. AST PASS,code_provenance 갱신.

- 2026-10-11 02:10 KST — CPU-only 후속분석 순차실행기 및그림코드 추가: physics완료 후offlineλ→MHW통계→climatology→보고→그림을 한worker씩 실행,각phase실패시 중단. GPUmanager호출없음,CUDA환경빈값상속. 누적skill/climatologycoverage/λ후보3그림PNG·PDF생성계획. AST PASS 및code_provenance갱신.

- 2026-10-11 02:12 KST — CPU 보고코드 실행전 보완: MHW 새FN/회복FN/새FP/제거FP 3seed집계와 누적skill/잔차-상대오차부호/Holm보정의 수치기반요약 추가. 학습/진단계산 불변, AST PASS 및코드해시갱신.

- 2026-10-11 02:13 KST — CPU-only analysis: physics PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:13 KST — CPU-only analysis: adaptive_offline PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:13 KST — CPU-only analysis: mhw_statistics PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:14 KST — CPU-only analysis: climatology PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:14 KST — CPU-only analysis: report PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:14 KST — CPU-only analysis: figures PASS; outputs=/ceph_data/yeonwoo/MHW_subset/outputs/v3_cpu_diagnostics_20261011; CUDA disabled; no model inference/backward/training.

- 2026-10-11 02:16 KST — CPU7항목최종완료/PASS:24조건72물리분할,144저장예측,380보호입력해시불변; 신규GPU/추론/backward/학습0, GPUqueue보류유지. Test8조건모두누적physics RMSE>zeroRHS;50%L14큰residual상위10% SST−1.645%;10%L7MHW미탐지증가/비MHW오차감소. rho.25 offline50%L14상한clamp69.8%. 10%singletonDOY28.23%,Train/Val/Test anomaly-z std1.036/1.856/1.841.90일SST Holm유의0/8,10%L7 MHW CSI/POD악화24비교Holm p=.0060. 원본SST지표/기존6case residual/4000bootstrap 회귀통과,메모리최대2735.9MiB. report_ko.md,summary_ko.md,3그림PNG/PDF,final_review.json저장. 모든Test해석탐색적/고정3seed조건부,모델성능개선판정아님.

- 2026-10-11 09:21 KST — 사용자 GPU재개지시로 Adaptive 파일럿 재개: GPU1/2/3 모두빈상태 확인. 기존rho=.25/EMA=.9/100updates/lambda[.0001,.03]/초기.01 유지; 50%L14·10%L7×3seed 신규6개, GPU사전검사6PASS후학습,Validation만평가. CPU진단결과로설정변경하지않음. native/operator/manifests/기존12best+last 해시검증PASS. 이전운영파일 resume_20261011_092120에복사보존, 감독PID1188804, preflight단계부터재개. GPU당smoke1개/학습계획2개,메모리admission/전체상한4개.

- 2026-10-11 09:24 KST — v3 study: Adaptive phase preflight gate PASS; outputs /ceph_data/yeonwoo/MHW_subset/outputs/v3_adaptive_physics_pilot_20261011.

- 2026-10-11 09:25 KST — v3 study: Adaptive pilot start ratio_50/lead14_seed42/adaptive; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-11 09:25 KST — v3 study: Adaptive pilot start ratio_50/lead14_seed7/adaptive; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-11 09:26 KST — v3 study: Adaptive pilot start ratio_50/lead14_seed123/adaptive; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-11 09:26 KST — v3 study: Adaptive pilot start ratio_10/lead7_seed42/adaptive; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-11 09:26 KST — v3 study: Adaptive pilot start ratio_10/lead7_seed7/adaptive; physics=0.01, auxiliary=0.0, identical heads; Validation only.

- 2026-10-11 09:27 KST — v3 study: Adaptive pilot start ratio_10/lead7_seed123/adaptive; physics=0.01, auxiliary=0.0, identical heads; Validation only.

## 2026-10-11 09:30 KST — Adaptive GPU 재개 후 첫 epoch 검증

- 대상: `outputs/v3_adaptive_physics_pilot_20261011/`; GPU 사전검사 6/6 PASS 후 GPU 1·2·3 각 2개, 총 6개 학습 진행 확인. 모든 GPU 프로세스 이름 `yeonwoo`, 프로세스당 NVML 약 11.30 GiB.
- 50%·Lead14 및 10%·Lead7 × seed 42/7/123 모두 최소 478 updates와 첫 Validation 완료, epoch 2 이상 진행 중. 저장 history 수치 finite, best.pt/last.pt 생성 확인; queue 실패 없음.
- 근거: `resume_20261011_092120/launch_verification.json`. 사전검사 checkpoint 복원·gradient 일치 PASS. 프로토콜 rho=0.25 및 기존 학습 설정 유지; 이후 Train/Validation 평가와 보고서 자동 진행, 신규 Test 평가는 실행하지 않음. 이번 확인에서 학습·모델 코드는 변경하지 않음.

## 2026-10-11 09:47 KST — 사용자 요청으로 Adaptive 중단 및 CPU 물리식 감사 전환

- scheduler를 먼저 정지한 후 UID/start_ticks가 일치하는 supervisor 및 6개 학습 PID에 SIGTERM. 기존 체크포인트·결과 보존. 남은 대상 PID: [].
- 신규 결과 경로 `outputs/v4_cpu_physics_audit_20261011/`; `gpu_stop_record.json`에 중단 기록. 신규 학습/추론/backward 금지, 이후 GPU 실험은 사용자가 tmux 명령으로 직접 실행.

- 2026-10-11 09:50 KST — v4 CPU physics audit: 신규 prepare_audit.py/audit_common.py 작성; 보호 입력 380개 선검사, 원자료 chunk 캐시 및 해안/단위/MHW 기준 검사. 원본 수정 없음.

- 2026-10-11 09:53 KST — v4 CPU physics audit: provenance_checks.py 신규 작성: 백업 GLORYS builder로12개 날짜 Td/u/v/w/dTdz/vadv 정확 재현 및 기존 해안 차분 일치 확인.

- 2026-10-11 09:54 KST — v4 CPU physics audit: check_vertical_sign.py 신규 CPU 검사: wo 연속방정식 부호와 P1 +w*dT/dd의 좌표 일관성 확인; 원본 미수정.

- 2026-10-11 09:55 KST — v4 CPU physics audit: vertical_sensitivity.py 신규 작성: 좌표 부호 근거에 따른 +/- 수직항 누적 민감도만 오프라인 비교; P0/P1 원본 코드 수정 없음.

- 2026-10-11 10:00 KST — v4 CPU physics audit: 사용자 재개 요청 후 GPU 중단 유지 확인. 기존 CPU PID1227117 실행 중으로 중복 실행 안 함. 신규 static_verification.md 저장; 이전 자동 승인 검토 사용량 오류로 미실행된 문서 작성 재개. run_ablation.py/statistics_and_tables.py 신규 코드 진행·작성 이력 확인, 원본 미변경. 향후 GPU는 사용자 직접 tmux 실행만 제공.

- 2026-10-11 10:01 KST — v4 CPU physics audit: contrast_checks.py: SST−Tdの日별 급변/결측 및 기존 Test-only MHW event와 전체 달력 label 경계차이를 CPU로 확인.

- 2026-10-11 10:08 KST — v4 CPU physics audit: glorys_mixed_layer_check.py 신규 추가: SST−Td 음수 빈도가 높아 GLORYS thetao의 깊이적분 Tm 대비를 보충검증. 기존 후보순위 동결 보존, 원본 수정·GPU 실행 없음.

- 2026-10-11 10:08 KST — v4 CPU physics audit: glorys_daily_metrics.py 보충검사 신규 코드: GLORYS Tm 두 후보의 일별 RMSE 추가 및 누적 MSE 독립 교차검증.

- 2026-10-11 10:09 KST — v4 CPU physics audit: run_ablation.py 신규 코드로 72조건 CPU 물리항 분해 완료; 후보 순위 Train/Validation에서 Test 계산 전에 동결.

- 2026-10-11 10:10 KST — v4 CPU physics audit: statistics_and_tables.py 신규 작성: seed별 결과/평균·SD 분리, 날짜 block30/90/180일 paired bootstrap 2000회, 공간 전체 보존, 보호 입력380개 후검사.

- 2026-10-11 10:10 KST — v4 CPU physics audit: write_report.py 작성·실행: 전체 수치/seed별·집계값/부호 감사/7개 최종 질문을 report_ko.md·summary_ko.md 및 독립 PNG/PDF로 정리. GPU 자동실행 없음.

- 2026-10-11 10:11 KST — v4 CPU physics audit: 전체 CPU 감사 최종 검토 완료: report_ko.md/summary_ko.md·CSV·PNG/PDF·코드 SHA 저장. 380개 입력 보존. 추가 GLORYS Tm 검사와 시간블록 CI 완료. 차기 GPU는 사용자 tmux 직접 실행만 제공.

- 2026-10-11 10:12 KST — v4 CPU physics audit: 최종 문서 순서 정리: 사용자 요청의7개 최종 질문 답변을 보고서 마지막에 배치. final_review.py 재현 코드와 source SHA 갱신; next_gpu_tmux.md 세션 준비만 제공, 학습 실행 없음.

### 2026-10-11 10:12 KST — 신규 CPU 감사 코드 파일 시각 확인

아래 시각은 실제 파일 mtime(Asia/Seoul), 사후 추정 시각 아님. 모두 outputs/v4_cpu_physics_audit_20261011/ 내 신규 분석 코드; 원본 학습 코드/데이터는 미변경.

- 2026-10-11 09:49 KST — audit_common.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 09:53 KST — check_vertical_sign.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 10:01 KST — contrast_checks.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 10:12 KST — final_review.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 10:07 KST — glorys_daily_metrics.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 10:05 KST — glorys_mixed_layer_check.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 09:49 KST — prepare_audit.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 09:53 KST — provenance_checks.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 09:52 KST — run_ablation.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 09:55 KST — statistics_and_tables.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 09:54 KST — vertical_sensitivity.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.
- 2026-10-11 10:04 KST — write_report.py — CPU 전용 재현·물리항 분해·자료 출처·통계·보고서 검증 목적.

- 2026-10-11 10:23 KST — v5 CPU horizontal advection audit: 신규 outputs/v5_cpu_horizontal_advection_audit_20261011 prepare/common/operators 작성. 좌표·부호·해안 불변성·해석장·격자 수렴검사 및 깊이가중 해류 CPU 캐시.

- 2026-10-11 10:27 KST — v5 CPU horizontal advection audit: validate_operators.py 신규 작성: u/v 양·음 방향,661해양 전체 스칼라 reference, 급격한 step, central2차/upwind1차 수렴검사 통과.

- 2026-10-11 10:31 KST — v5 CPU horizontal advection audit: depth_resolved.py 추가 CPU 감사: 깊이별u∇θ 평균, 평균유속×평균기울기, 평균유속×평균수온기울기 분리. 미래 GL 관측 기반 진단 후보, 학습용으로 채택/실행하지 않음.

- 2026-10-11 10:34 KST — v5 CPU horizontal advection audit: run_audit.py 신규 작성·완료:19가지 이류 계산/대조를72분할 CPU 검증; Train/Val 순위 Test 전에 동결.

- 2026-10-11 10:36 KST — v5 CPU horizontal advection audit: summarize.py:24조건×3분할×22후보 수치 집계, 동일 날짜 블록 30/90/180일 2,000회 CI 및 기존 입력380개·v4 재사용 입력 SHA 보존 검사 완료.

- 2026-10-11 10:39 KST — v5 CPU horizontal advection audit: make_report.py: CPU 수평 이류 감사 한국어 보고서·요약·Validation 그림 생성. 수치 구현 검사는 PASS, B가 C를 넘는 물리 적합성 기준은 미충족.

- 2026-10-11 10:39 KST — v5 CPU horizontal advection audit: finalize.py: 신규 분석 코드 문법·한국어 보고서·그림 및 결과 manifest 검증 완료. 22후보×72분할 1,584행 finite. 원본380개/v4입력18개 보존. Validation 최선 B_depth_uv는 A 대비 −0.680%, C 대비 +71.530%; B 8종 모두 C 개선0/24. GPU 미실행.

신규 감사 코드 실제 파일 mtime (KST); 대상은 outputs/v5_cpu_horizontal_advection_audit_20261011/:
- 2026-10-11 10:23 KST — common.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:29 KST — depth_resolved.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:39 KST — finalize.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:38 KST — make_report.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:23 KST — operators.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:23 KST — prepare.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:25 KST — run_audit.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:36 KST — summarize.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.
- 2026-10-11 10:27 KST — validate_operators.py: 수평 이류 CPU 진단·재현·회귀검사·통계·보고서 작성 목적.

- 2026-10-11 10:52 KST — outputs/glorys_source_inventory_20261011/inspect_sources.py 신규 작성·실행: 기존 서버 백업·Ceph NetCDF/처리 코드/압축파일 목록과 변수 메타데이터 조사, 2003 원격자 수온과 병합3D 직접 비교. 원본 미변경, 다운로드·GPU 미실행.

- 2026-10-11 10:54 KST — outputs/glorys_source_inventory_20261011/write_report.py 및 report_ko.md, netcdf_headers.txt 생성: 병합3D 전체기간 thetao/uo/vo와2003 원격자 thetao/mlotst 실재 확인, 원격자 유속·최초 재격자화 코드 미발견, 추가 자료 필요 범위 분리. inspect_sources.py 최초 구문 오류 수정 후 정상 완료. 다운로드·GPU 미실행.

- 2026-10-11 11:08 KST — v5 CPU advection root cause: common.py/preflight.py 신규 작성: 보호380개 사전SHA 및 재사용 입력SHA, 이전Adaptive7개 PID/start-tick 중단 확인; A–E CPU 진단 설정 저장.

- 2026-10-11 11:15 KST — v5 CPU advection root cause: run_de.py: D/E 원격자 대응·결측/깊이 검사 및 같은 영역의 이류 포함/제외 비교 완료. 2003 we=0 통제실험, 전체기간 원격자 유속 없음.

- 2026-10-11 11:17 KST — v5 CPU advection root cause: check_analysis.py: MSE 항등식·일별RMSE·구배 방향·부분수심 적분·해저 외삽금지·보충영역 이웃분류 회귀검사 PASS.

- 2026-10-11 11:23 KST — v5 CPU advection root cause: check_ml_support.py:2003 실제 legacy depth<=H 선택에 원격자 thetao 결측층이 포함되는 빈도 직접계수. 유속 원마스크를 수온마스크로 대체하지 않았으며 오류유속 비율로 해석하지 않음.

- 2026-10-11 11:27 KST — v5 CPU advection root cause: run_abc.py/statistics.py: A/B/C 계산 완료. 대표6조건 Train/Val 후24조건 확장, Test 분리; 저장we만 사용.

- 2026-10-11 11:27 KST — v5 CPU advection root cause: summarize.py: 72분할×2경로×8후보, 동일표본MSE 분해·기존P0 재현 및 보호380 SHA PASS. D 코드의 날짜별 표본수 공간축 집계 오류는 신규 코드에서 수정했고 완료된 E를 재사용함.

- 2026-10-11 11:29 KST — v5 CPU advection root cause: write_report.py: A–E 한국어 전체보고서·9개질문답변·최종결정·표/지도 저장. CPU만 실행, 기존P0/이류제외 통제비교 권고, 원인 미확정 한계 표시.

- 2026-10-11 11:30 KST — v5 CPU advection root cause: final_review.py: 필수13개 산출물·신규코드 구문·72분할·모든 이류 포함/제외 pair 표본수 일치·회귀검사·보호380SHA 검증 완료. 원본 물리식 수정, 다운로드, GPU, 신규추론 없음.

이번 신규 코드 파일 실제 mtime(KST), outputs/v5_cpu_advection_root_cause_20261011/:
- 2026-10-11 11:17 KST — check_analysis.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:23 KST — check_ml_support.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:08 KST — common.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:22 KST — final_review.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:08 KST — preflight.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:09 KST — run_abc.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:14 KST — run_de.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:09 KST — statistics.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:16 KST — summarize.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:29 KST — write_report.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.

- 2026-10-11 11:31 KST — v5 CPU advection root cause: write_report.py: A–E 한국어 전체보고서·9개질문답변·최종결정·표/지도 저장. CPU만 실행, 기존P0/이류제외 통제비교 권고, 원인 미확정 한계 표시.

- 2026-10-11 11:31 KST — v5 CPU advection root cause: final_review.py: 필수13개 산출물·신규코드 구문·72분할·모든 이류 포함/제외 pair 표본수 일치·회귀검사·보호380SHA 검증 완료. 원본 물리식 수정, 다운로드, GPU, 신규추론 없음.

이번 신규 코드 파일 실제 mtime(KST), outputs/v5_cpu_advection_root_cause_20261011/:
- 2026-10-11 11:17 KST — check_analysis.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:23 KST — check_ml_support.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:08 KST — common.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:22 KST — final_review.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:08 KST — preflight.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:09 KST — run_abc.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:14 KST — run_de.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:09 KST — statistics.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:16 KST — summarize.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.
- 2026-10-11 11:30 KST — write_report.py: A–E CPU 원인분리·동일표본 비교·보고서 검증 목적.

- 2026-10-11 12:56 KST — v6 flux/residual CPU audit: audit.py 신규 작성 및 실행: 보호 입력/기존 screen 결과 사전SHA, 전체 기간 해양 qnet 성분합·단위·부호 검사 통과. CPU optical 비교와 residual 관계 진단 시작. 원본 모델/자료 변경 없음.

- 2026-10-11 12:59 KST — v6 flux/residual CPU audit: report.py 신규 작성, 투과열 paired 비교·일별 residual 상관/구간/계절/해안 및 블록 재표집 구간을 보고서와 그림으로 저장. 해저 수심 및 최초 ERA5 누적 처리 미검증을 명시.

- 2026-10-11 12:59 KST — v6 flux/residual CPU audit: CPU penetration=on Lead별 24구간 및 residual 일별 진단 완료. 보호 파일 사후 SHA 일치. 결과: outputs/v6_cpu_flux_residual_audit_20261011.

- 2026-10-11 13:00 KST — v6 flux/residual CPU audit: audit.py 관계 진단에 abs_dh=|일별 혼합층 깊이 변화| 추가: 심화/얕아짐 상쇄 가능성 검사. 기존 저장 일별 residual만 재집계하며 관측장/광학 실험은 재계산하지 않음.

- 2026-10-11 13:01 KST — v6 flux/residual CPU audit: report.py 신규 작성, 투과열 paired 비교·일별 residual 상관/구간/계절/해안 및 블록 재표집 구간을 보고서와 그림으로 저장. 해저 수심 및 최초 ERA5 누적 처리 미검증을 명시.

- 2026-10-11 13:27 KST — v7 mixed-budget: 신규 v7 코드 작성 및 CPU 준비 시작. process title 설정을 환경변수 변경보다 먼저 수행하도록 수정; GPU 미실행.

- 2026-10-11 13:28 KST — v7 mixed-budget: prepare_mixed_budget_features.py: CPU 3D→2D feature cache completed; raw/v1/v2/v3 preserved; backward MLD proxy and oracle-forcing scope recorded.

- 2026-10-11 13:42 KST — v7 mixed-budget: v7 CPU 회귀/6개 subset 통계·입력·공통 구조·RHS 크기 검사 후 24회 수동 프로토콜 고정. GPU smoke 및 학습은 미실행.

- 2026-10-11 13:43 KST — v7 mixed-budget: v7 CPU 요약 작성; pending_only=True

- 2026-10-11 13:43 KST — v7 mixed-budget: v7 수동 실행 준비 완료: scripts/mixed_budget_common.py, mixed_budget_physics.py, prepare_mixed_budget_features.py, train_mixed_budget_gpu.py, validate_mixed_budget_training.py, run_mixed_budget_stage.py, run_mixed_budget_tmux.sh, summarize_mixed_budget_gpu.py, README_mixed_budget_gpu_ko.md 신규 작성. CPU 단위/6 subset/실제 HiLAM forward·autograd·validation(optimizer0) PASS, 380 보호 SHA PASS, CLI 및 bash 구문 확인. 24회 프로토콜 고정, GPU0/1/2 한 worker씩 단계별 실행. GPU smoke/학습/평가 모두 사용자 수동 실행 대기.

- 2026-10-11 13:45 KST — v7 mixed-budget: summarize_mixed_budget_gpu.py 공통 A/B residual 평가 경로를 실제2개 관측 endpoint를 임시 예측으로 넣어 CPU 확인; 유한값 PASS. 성능 결과로 사용하지 않음.

- 2026-10-11 13:51 KST — v7 mixed-budget: smoke 시작 B1_r10_lead7_seed42, GPU1, PID1517424, λ=0.1; oracle forcing, NOAA proxy, no adaptive.

- 2026-10-11 13:51 KST — v7 mixed-budget: smoke 시작 B0_r10_lead7_seed42, GPU0, PID1517396, λ=0.0; oracle forcing, NOAA proxy, no adaptive.

- 2026-10-11 13:51 KST — v7 mixed-budget: smoke 시작 A_r10_lead7_seed42, GPU2, PID1517436, λ=0.1; oracle forcing, NOAA proxy, no adaptive.

- 2026-10-11 13:57 KST — v7 mixed-budget: scripts/audit_mixed_budget_assumptions_cpu.py 및 outputs/v7_gpu_mixed_budget_comparison_20261011/assumption_audit/: 전체 순단파 단일지수 근사·NOAA/GLORYS 수온 차이·미래 forcing 경로 및 CPU 잔차 부호 반전 감사. 6개 통계/기후값, 실제 두 리드 train/val A/B 반사실 8건 PASS; oracle 감독과 과보정 가능성 명시. 모델/프로토콜/전처리 변경 및 GPU/학습 없음.

- 2026-10-11 14:12 KST — v8 training-only oracle96: scripts/oracle96/ 별도 96조건 실행기 생성. 원본/v7 보존, Train-only oracle·검증 forcing 제거·동일38240 updates·λ=.1·Kd=.06 고정. CPU 단위/24개 manifest 경계/양리드 실제 loss·gradient 검사 PASS. GPU는 0/1/2만, 1프로세스/GPU. 순차 gate 후 확대하도록 사용자 승인 반영.

- 2026-10-11 14:13 KST — v8 GPU 허용 목록을 사용자 최신 지시에 따라7/8/9로 변경. scripts/oracle96 실행기·GPU guard·문서·프로토콜 모두 동기화. 이전 사전검사/코드/프로토콜은 prelaunch_revision1에 보존. 실행 전 MHW 지표 문자열 비교 처리 및 patience CLI 기록 정합 수정. CPU 기존 단위/경계/gradient 검사는 PASS, 추가 추론 회귀 검사 후 GPU 실행 예정.

- 2026-10-11 14:13 KST — v8 training-only oracle96: GPU7/8/9 실행 전 추가 CPU 회귀 PASS: 미래forcing 제거/실제추가/NaN교란 예측·SST/MHW 동일, signed/normalized residual 평가 커널 finite. 학습 없음, CUDA 미초기화. 96조건 보고서/CSV는 미완료 상태로 생성.

- 2026-10-11 14:14 KST — v8 training-only oracle96: 사용자 직접 실행 승인으로 tmux v8_oracle96_gpu789 시작, pane PID1542305, GPU7/8/9 각1 worker 제한. CPU/추론 회귀 PASS 후 smoke부터 gated96조건 실행, 로그 /ceph_data/yeonwoo/MHW_subset/outputs/v8_training_only_oracle96_20261011/logs/controller_1791695661334665768.log.

- 2026-10-11 14:14 KST — v8 training-only oracle96: smoke: launch Baseline_r10_lead7_seed42 GPU7 PID1542414

- 2026-10-11 14:14 KST — v8 training-only oracle96: smoke: launch P0_r10_lead7_seed42 GPU8 PID1542429

- 2026-10-11 14:14 KST — v8 training-only oracle96: smoke: launch NoAdv_r10_lead7_seed42 GPU9 PID1542448

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke 시작 NoAdv_r10_lead7_seed42, GPU9, PID1542448, λ=0.1; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke 시작 Baseline_r10_lead7_seed42, GPU7, PID1542414, λ=0.0; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke 시작 P0_r10_lead7_seed42, GPU8, PID1542429, λ=0.1; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke 완료 P0_r10_lead7_seed42

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke 완료 NoAdv_r10_lead7_seed42

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke 완료 Baseline_r10_lead7_seed42

- 2026-10-11 14:15 KST — v8 training-only oracle96: smoke: launch Adv_r10_lead7_seed42 GPU7 PID1543537

- 2026-10-11 14:16 KST — v8 training-only oracle96: smoke 시작 Adv_r10_lead7_seed42, GPU7, PID1543537, λ=0.1; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:16 KST — v8 training-only oracle96: smoke 완료 Adv_r10_lead7_seed42

- 2026-10-11 14:16 KST — v8 training-only oracle96: representative: launch Baseline_r10_lead7_seed42 GPU7 PID1544497

- 2026-10-11 14:16 KST — v8 training-only oracle96: representative: launch P0_r10_lead7_seed42 GPU8 PID1544500

- 2026-10-11 14:16 KST — v8 training-only oracle96: representative: launch NoAdv_r10_lead7_seed42 GPU9 PID1544508

- 2026-10-11 14:16 KST — v8 training-only oracle96: train 시작 P0_r10_lead7_seed42, GPU8, PID1544500, λ=0.1; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:16 KST — v8 training-only oracle96: GPU smoke4조건 모두 PASS: forward/backward/유한 loss·gradient/체크포인트/forcing 제거·교란 SST·MHW 동일, 4모델 setup동일. GPU7 Baseline·8 P0·9 NoAdv 대표 학습 시작, Adv 대기. 이후 full Validation 불변성·38240update·동일shuffle gate 통과 시 잔여조건 자동 확대.

- 2026-10-11 14:16 KST — v8 training-only oracle96: train 시작 Baseline_r10_lead7_seed42, GPU7, PID1544497, λ=0.0; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:16 KST — v8 training-only oracle96: train 시작 NoAdv_r10_lead7_seed42, GPU9, PID1544508, λ=0.1; training-only oracle, NOAA proxy, fixed38240 updates, no adaptive.

- 2026-10-11 14:22 KST — v8 training-only oracle96: scripts/oracle96/resource_scheduler.py·resource_worker.py 추가: GPU6 대표 Adv 추가 및 이후7/8/9 각3개(총9개). 기존3학습PID 유지 인계, 과학 프로토콜·학습 코드 해시 불변. GPU별14GiB 신규 예약·2GiB 여유·외부PID 차단 단위검사6건 PASS. 초기PID 검사는 process title=yeonwoo라 argv 확인이 불가능하여 실제 PPID·시작tick·Ceph stdout 로그FD로 신원을 확인. resource_policy.json에 자원 정책 기록.

- 2026-10-11 14:23 KST — v8 training-only oracle96: 자원 관리자 인계 실행: 기존 manager PID1542307만 SIGSTOP, 학습PID1544497/1544500/1544508 계속실행 확인. 새 tmux v8_resource_4then9 panePID1551659, GPU6 Adv 추가 후 대표4검증, 이후7/8/9 각3개. 기존manager는 대표3학습 종료 후에만 정리하여 학습손실 방지.

- 2026-10-11 14:23 KST — v8 training-only oracle96: resource scheduler representative: Adv_r10_lead7_seed42 GPU6 PID1551806

- 2026-10-11 14:24 KST — v8 training-only oracle96: resource adapter train 시작 Adv_r10_lead7_seed42 GPU6 PID1551806; 과학 설정·seed·optimizer·학습 예산 불변

- 2026-10-11 14:25 KST — v8 training-only oracle96: 자원 인계 실제 확인 PASS: 기존 학습PID3개 유지·실행 지속, GPU6 Adv PID1551806 yeonwoo/UUID 확인. 현재6/7/8/9 각1개 대표4동시. 대표 gate 후6해제,7/8/9 각최대3개로 전환. 과학프로토콜·고정코드 SHA 불변.

- 2026-10-11 16:17 KST — v8 training-only oracle96: train 완료 Baseline_r10_lead7_seed42

- 2026-10-11 16:27 KST — v8 training-only oracle96: train 완료 NoAdv_r10_lead7_seed42

- 2026-10-11 16:37 KST — v8 training-only oracle96: train 완료 P0_r10_lead7_seed42

- 2026-10-11 16:47 KST — v8 training-only oracle96: resource adapter train 완료 Adv_r10_lead7_seed42

- 2026-10-11 16:47 KST — v8 training-only oracle96: resource scheduler 실패, 자동 확대/재시도 금지; 기존 관리자는 재개하지 않음.

- 2026-10-11 16:52 KST — v8 training-only oracle96: 대표4조건 결과 representative_results/report_ko.md·comparison_summary.csv 생성: 전부80epoch38240updates, NoAdv ValRMSE0.977975로Baseline대비8.17%개선(seed42한조건). 16:47 flock인계 오류로후속미시작 확인. shared_scheduler.py/shared_worker.py 신규추가: GPU7/8/9 각2개공유를 사용자 승인에따라 적용, allocator14GiB·headroom4GiB·실제예약·잠금해제대기 회귀PASS. 기존실패기록/체크포인트/과학프로토콜보존.

- 2026-10-11 16:53 KST — v8 training-only oracle96: 공유GPU 후속실험 시작: tmux v8_shared_gpu789_two_each, panePID1705942, GPU7/8/9 각최대2개. 대표4체크포인트해시/과학설정보존 및 lock검사 PASS. 현재다른작업 약14.5GB/GPU를 유지한채 여유메모리로우리작업실행. 실패시우리자식만중단, 자동재시도없음.

- 2026-10-11 16:56 KST — v8 training-only oracle96: shared scheduler remaining_10_lead7_seeds: Baseline_r10_lead7_seed7 GPU7 PID1709964

- 2026-10-11 16:56 KST — v8 training-only oracle96: shared scheduler remaining_10_lead7_seeds: P0_r10_lead7_seed7 GPU7 PID1709974

- 2026-10-11 16:56 KST — v8 training-only oracle96: shared scheduler remaining_10_lead7_seeds: NoAdv_r10_lead7_seed7 GPU8 PID1709992

- 2026-10-11 16:56 KST — v8 training-only oracle96: shared scheduler remaining_10_lead7_seeds: Adv_r10_lead7_seed7 GPU8 PID1709995

- 2026-10-11 16:56 KST — v8 training-only oracle96: shared scheduler remaining_10_lead7_seeds: Baseline_r10_lead7_seed123 GPU9 PID1709999

- 2026-10-11 16:56 KST — v8 training-only oracle96: shared scheduler remaining_10_lead7_seeds: P0_r10_lead7_seed123 GPU9 PID1710008

- 2026-10-11 16:59 KST — v8 training-only oracle96: 실제공유GPU 연결확인 PASS: GPU7 PID1709964/1709974, GPU8 PID1709992/1709995, GPU9 PID1709999/1710008, 모두yeonwoo. 기존타사용자프로세스유지. 6workerCUDA초기화완료·모델/데이터로딩진행, 각2개제한 및 다음2작업대기.

- 2026-10-11 17:00 KST — v8 training-only oracle96: shared GPU adapter train 시작 Baseline_r10_lead7_seed123 GPU9 PID1709999; 과학 설정·seed·optimizer·학습 예산 불변

- 2026-10-11 17:00 KST — v8 training-only oracle96: shared GPU adapter train 시작 Baseline_r10_lead7_seed7 GPU7 PID1709964; 과학 설정·seed·optimizer·학습 예산 불변

- 2026-10-11 17:00 KST — v8 training-only oracle96: shared GPU adapter train 시작 P0_r10_lead7_seed7 GPU7 PID1709974; 과학 설정·seed·optimizer·학습 예산 불변

- 2026-10-11 17:00 KST — v8 training-only oracle96: shared GPU adapter train 시작 P0_r10_lead7_seed123 GPU9 PID1710008; 과학 설정·seed·optimizer·학습 예산 불변

- 2026-10-11 17:00 KST — v8 training-only oracle96: shared GPU adapter train 시작 Adv_r10_lead7_seed7 GPU8 PID1709995; 과학 설정·seed·optimizer·학습 예산 불변

- 2026-10-11 17:00 KST — v8 training-only oracle96: shared GPU adapter train 시작 NoAdv_r10_lead7_seed7 GPU8 PID1709992; 과학 설정·seed·optimizer·학습 예산 불변


## 2026-10-11 17:09 KST — 대표 네 모델의 조기 Validation-best 로그 감사

- 대상: `outputs/v8_training_only_oracle96_20261011/early_best_audit_20261011/`에 CPU 전용 `analyze.py`, `report_ko.md`, epoch 320행 CSV, 모델 요약·checkpoint Validation 검증·Train/Validation 분포 CSV, 근거 SHA JSON 및 학습 곡선 PNG/PDF 3종을 추가했다.
- 이유: 동일 80 epoch(38,240 optimizer step 호출) 완료에도 best가 epoch1~4에 집중된 원인을 기존 기록만으로 분리하기 위함. 기존 학습 코드·설정·원본·체크포인트·결과는 수정하지 않았으며 실행 중 학습을 제어하지 않았다. 새 학습·GPU 호출·추론·원자료 로딩·checkpoint 역직렬화 없이 수행했다. 다중 분할 기존 분포 JSON에서는 Train/Validation 행만 사용하며 Test 지표를 분석·선택에 사용하지 않았다.
- 확인: 376개 Train 샘플의 24 batches/순회 대비 virtual epoch=478 updates, 따라서 best는 약20~80순회 후이다. epoch80 Val RMSE 악화율(best 기준): Baseline28.80%, P0 31.41%, NoAdv19.75%, Adv30.64%. Train loss는 전반적으로 감소하고 Val은 등락하면서 악화해 공통 과적합 양상. v3 SST anomaly Train/Val std 1.054/1.604의 분포 차이도 현재 동일 manifest·통계에 적용됨. LR 인과효과는 대조 실험 없이 확정 불가.
- 설정/검증: early stopping=false, patience81이며 중단 분기 없음. warmup5 이후 cosine, epoch80 LR=0이므로 epoch79/80 Val 동일. 저장된 best.pt 재로드 검증 PASS 및 full Val RMSE 차이 <3e-9 K; 별도 최종 평가 완료 증거는 아직 없음. frozen code40개 SHA 일치, epoch320행·Train/Validation 행 필터·산출물 생성 검증 및 대표 그림 육안 확인 완료.
- 권고: 현재 구조와 best checkpoint 유지; 현재 통제 실험은 중간 변경하지 않음. 후속 실험에서 공통 early stopping과 초기 평가 간격을 사전 설계하고 LR는 별도 Train/Validation 대조로 결정.

- 2026-10-11 17:34 KST — v8 training-only oracle96: shared scheduler 실패: 자동확대/재시도 금지


## 2026-10-11 17:34 KST — 사용자 요청에 따른 v8 실험 중지

- 실험 구조 변경을 위해 shared scheduler PID 1705944 및 학습 worker 6개 [1709964, 1709974, 1709992, 1709995, 1709999, 1710008]를 중지했다. PID 시작 시각·소유자·부모·lease를 확인한 이번 실험 프로세스만 신호를 보냈다.
- SIGINT로 자동 관리자를 중단해 후속 작업 자동 실행을 차단했다. 관련 프로세스 종료 및 GPU compute PID 소멸을 확인했다. 다른 사용자의 작업은 조작하지 않았다.
- 기존 best.pt/last.pt·로그·완료된 대표 4개 결과는 삭제·이동·덮어쓰지 않았다. 중단된 현재 epoch의 미저장 진행분은 포함되지 않으며, 최근 완료 epoch의 저장 파일을 보존한다.
- 상태 스냅샷/중지 검증: `outputs/v8_training_only_oracle96_20261011/user_stop_20261011_173438.json`. 사용자 재개 지시 전 새 실험/자동 재개 금지.

- 2026-10-11 17:52 KST — v8 actual-epoch oracle96: 新 Actual Epoch専用 scripts/oracle96_actual 追加: v8_actual_epoch_oracle96_20261011 별도96조건, early stopping OFF, actual80, LR/physics warmup5, old λ=.1 보존. CPU sampling/경계/실제loss gradient/80epoch resume/보호hash PASS. 기존 v8 및 Test 평가 미변경. GPU smoke 후 대표4개, 검증 PASS 후92개 자동확대 승인 반영.

- 2026-10-11 17:54 KST — v8 actual-epoch oracle96: Actual Epoch CPU 최종 준비: 기존 Validation/MHW reference 바이트 동일 사본, 신규 코드 SHA 동결, GPU7/8/9당2개 및 대기 worker 메모리 예약 검사 PASS. 기존 Oracle96 LR 서술은 실제 epoch 수식에 맞게 수정하고 physics warmup0→5 변경을 분리 명시. Test 자동 실행 없음.

- 2026-10-11 17:54 KST — Actual Epoch 신규 실행 관리자 tmux `v8_actual_oracle96_gpu789` 시작. GPU7/8/9 각 최대2개, 기존 관리자 종료 확인. CPU preflight PASS 후 smoke4→representative4→무결성 PASS 시 나머지92 자동확대. 기존 Virtual 체크포인트는 신규 학습에 재사용하지 않음. Test 자동 평가 없음. 출력 `/ceph_data/yeonwoo/MHW_subset/outputs/v8_actual_epoch_oracle96_20261011`.

- 2026-10-11 17:54 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead7_seed42 phase=smoke GPU7 PID1787721

- 2026-10-11 17:54 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead7_seed42 phase=smoke GPU8 PID1787722

- 2026-10-11 17:54 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead7_seed42 phase=smoke GPU9 PID1787723

- 2026-10-11 17:54 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead7_seed42 phase=smoke GPU7 PID1788067

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke start NoAdv_r10_lead7_seed42 GPU9 PID1787723, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke start Baseline_r10_lead7_seed42 GPU7 PID1787721, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke start P0_r10_lead7_seed42 GPU8 PID1787722, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke start Adv_r10_lead7_seed42 GPU7 PID1788067, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke complete NoAdv_r10_lead7_seed42

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke complete Baseline_r10_lead7_seed42

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke complete P0_r10_lead7_seed42

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: smoke complete Adv_r10_lead7_seed42

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead7_seed42 phase=train GPU7 PID1789321

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead7_seed42 phase=train GPU8 PID1789322

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead7_seed42 phase=train GPU9 PID1789323

- 2026-10-11 17:55 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead7_seed42 phase=train GPU7 PID1789751

- 2026-10-11 17:56 KST — v8 actual-epoch oracle96: train start P0_r10_lead7_seed42 GPU8 PID1789322, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:56 KST — v8 actual-epoch oracle96: train start NoAdv_r10_lead7_seed42 GPU9 PID1789323, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:56 KST — v8 actual-epoch oracle96: train start Baseline_r10_lead7_seed42 GPU7 PID1789321, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:56 KST — v8 actual-epoch oracle96: train start Adv_r10_lead7_seed42 GPU7 PID1789751, actual80 epochs, updates=1920, warmup5

- 2026-10-11 17:59 KST — Actual Epoch 대표4개 실행 검증 PASS: CPU 보호654파일·sampling24조건·양lead 실제physics gradient·80epoch synthetic resume PASS, GPU smoke4개 모두PASS. 실제 대표4개 모두epoch5 이상 진행; 매epoch376샘플 정확히1회/24updates, 최초5epoch 모델간shuffle일치, epoch5 physics λ=.1(Baseline0)/LR=.0002 및 post-warmup checkpoint 생성 확인. GPU7 Baseline+Adv, GPU8 P0, GPU9 NoAdv, 전worker명 yeonwoo 검증. 현재 학습 진행 중이며 대표완료/96완료로 표시하지 않음. 대표80epoch 및 checkpoint/invariance gate PASS 후92개 자동확대; Test 미실행. 근거 `outputs/v8_actual_epoch_oracle96_20261011/launch_verification.json`.

- 2026-10-11 18:23 KST — v8 actual-epoch oracle96: train complete NoAdv_r10_lead7_seed42

- 2026-10-11 18:24 KST — v8 actual-epoch oracle96: train complete P0_r10_lead7_seed42

- 2026-10-11 18:32 KST — v8 actual-epoch oracle96: train complete Baseline_r10_lead7_seed42

- 2026-10-11 18:33 KST — v8 actual-epoch oracle96: train complete Adv_r10_lead7_seed42

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead7_seed7 phase=train GPU7 PID1839131

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead7_seed7 phase=train GPU8 PID1839132

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead7_seed7 phase=train GPU9 PID1839136

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead7_seed7 phase=train GPU7 PID1839482

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead7_seed123 phase=train GPU8 PID1839484

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead7_seed123 phase=train GPU9 PID1839494

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: train start Baseline_r10_lead7_seed7 GPU7 PID1839131, actual80 epochs, updates=1920, warmup5

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: train start Baseline_r10_lead7_seed123 GPU8 PID1839484, actual80 epochs, updates=1920, warmup5

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: train start P0_r10_lead7_seed7 GPU8 PID1839132, actual80 epochs, updates=1920, warmup5

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: train start NoAdv_r10_lead7_seed7 GPU9 PID1839136, actual80 epochs, updates=1920, warmup5

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: train start Adv_r10_lead7_seed7 GPU7 PID1839482, actual80 epochs, updates=1920, warmup5

- 2026-10-11 18:34 KST — v8 actual-epoch oracle96: train start P0_r10_lead7_seed123 GPU9 PID1839494, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:17 KST — v8 actual-epoch oracle96: train complete Baseline_r10_lead7_seed7

- 2026-10-11 19:17 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead7_seed123 phase=train GPU7 PID1887546

- 2026-10-11 19:18 KST — v8 actual-epoch oracle96: train complete Baseline_r10_lead7_seed123

- 2026-10-11 19:18 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead7_seed123 phase=train GPU8 PID1888336

- 2026-10-11 19:18 KST — v8 actual-epoch oracle96: train start NoAdv_r10_lead7_seed123 GPU7 PID1887546, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:18 KST — v8 actual-epoch oracle96: train start Adv_r10_lead7_seed123 GPU8 PID1888336, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:20 KST — v8 actual-epoch oracle96: train complete NoAdv_r10_lead7_seed7

- 2026-10-11 19:20 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead14_seed42 phase=train GPU9 PID1890900

- 2026-10-11 19:20 KST — v8 actual-epoch oracle96: train complete P0_r10_lead7_seed7

- 2026-10-11 19:20 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead14_seed42 phase=train GPU8 PID1891111

- 2026-10-11 19:21 KST — v8 actual-epoch oracle96: train complete Adv_r10_lead7_seed7

- 2026-10-11 19:21 KST — v8 actual-epoch oracle96: train start Baseline_r10_lead14_seed42 GPU9 PID1890900, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:21 KST — v8 actual-epoch oracle96: train complete P0_r10_lead7_seed123

- 2026-10-11 19:21 KST — v8 actual-epoch oracle96: train start P0_r10_lead14_seed42 GPU8 PID1891111, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:22 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead14_seed42 phase=train GPU7 PID1893210

- 2026-10-11 19:22 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead14_seed42 phase=train GPU9 PID1893218

- 2026-10-11 19:23 KST — v8 actual-epoch oracle96: train start Adv_r10_lead14_seed42 GPU9 PID1893218, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:23 KST — v8 actual-epoch oracle96: train start NoAdv_r10_lead14_seed42 GPU7 PID1893210, actual80 epochs, updates=1920, warmup5

- 2026-10-11 19:58 KST — v8 actual-epoch oracle96: train complete Baseline_r10_lead14_seed42

- 2026-10-11 19:58 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead14_seed7 phase=train GPU9 PID1933906

- 2026-10-11 19:59 KST — v8 actual-epoch oracle96: train start Baseline_r10_lead14_seed7 GPU9 PID1933906, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:01 KST — v8 actual-epoch oracle96: train complete Adv_r10_lead7_seed123

- 2026-10-11 20:01 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead14_seed7 phase=train GPU8 PID1937202

- 2026-10-11 20:02 KST — v8 actual-epoch oracle96: train start P0_r10_lead14_seed7 GPU8 PID1937202, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:03 KST — v8 actual-epoch oracle96: train complete NoAdv_r10_lead7_seed123

- 2026-10-11 20:03 KST — v8 actual-epoch oracle96: train complete Adv_r10_lead14_seed42

- 2026-10-11 20:03 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead14_seed7 phase=train GPU7 PID1938668

- 2026-10-11 20:03 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead14_seed7 phase=train GPU9 PID1938670

- 2026-10-11 20:03 KST — v8 actual-epoch oracle96: train start NoAdv_r10_lead14_seed7 GPU7 PID1938668, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:03 KST — v8 actual-epoch oracle96: train start Adv_r10_lead14_seed7 GPU9 PID1938670, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:04 KST — v8 actual-epoch oracle96: train complete P0_r10_lead14_seed42

- 2026-10-11 20:04 KST — v8 actual-epoch oracle96: queue launch Baseline_r10_lead14_seed123 phase=train GPU8 PID1940316

- 2026-10-11 20:05 KST — v8 actual-epoch oracle96: train start Baseline_r10_lead14_seed123 GPU8 PID1940316, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:06 KST — v8 actual-epoch oracle96: train complete NoAdv_r10_lead14_seed42

- 2026-10-11 20:06 KST — v8 actual-epoch oracle96: queue launch P0_r10_lead14_seed123 phase=train GPU7 PID1941845

- 2026-10-11 20:06 KST — v8 actual-epoch oracle96: train start P0_r10_lead14_seed123 GPU7 PID1941845, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:35 KST — v8 actual-epoch oracle96: train complete Baseline_r10_lead14_seed7

- 2026-10-11 20:35 KST — v8 actual-epoch oracle96: queue launch NoAdv_r10_lead14_seed123 phase=train GPU9 PID1969912

- 2026-10-11 20:36 KST — v8 actual-epoch oracle96: train start NoAdv_r10_lead14_seed123 GPU9 PID1969912, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:40 KST — v8 actual-epoch oracle96: train complete NoAdv_r10_lead14_seed7

- 2026-10-11 20:40 KST — v8 actual-epoch oracle96: queue launch Adv_r10_lead14_seed123 phase=train GPU7 PID1974429

- 2026-10-11 20:41 KST — v8 actual-epoch oracle96: train start Adv_r10_lead14_seed123 GPU7 PID1974429, actual80 epochs, updates=1920, warmup5

- 2026-10-11 20:41 KST — v8 actual-epoch oracle96: train complete P0_r10_lead14_seed7

- 2026-10-11 20:41 KST — v8 actual-epoch oracle96: queue launch Baseline_r25_lead7_seed42 phase=train GPU8 PID1975793

- 2026-10-11 20:42 KST — v8 actual-epoch oracle96: train complete Baseline_r10_lead14_seed123

- 2026-10-11 20:42 KST — v8 actual-epoch oracle96: queue launch P0_r25_lead7_seed42 phase=train GPU8 PID1976140

- 2026-10-11 20:42 KST — v8 actual-epoch oracle96: train start Baseline_r25_lead7_seed42 GPU8 PID1975793, actual80 epochs, updates=4960, warmup5

- 2026-10-11 20:42 KST — v8 actual-epoch oracle96: train start P0_r25_lead7_seed42 GPU8 PID1976140, actual80 epochs, updates=4960, warmup5

- 2026-10-11 20:43 KST — v8 actual-epoch oracle96: train complete Adv_r10_lead14_seed7

- 2026-10-11 20:43 KST — v8 actual-epoch oracle96: queue launch NoAdv_r25_lead7_seed42 phase=train GPU9 PID1977625

- 2026-10-11 20:44 KST — v8 actual-epoch oracle96: train start NoAdv_r25_lead7_seed42 GPU9 PID1977625, actual80 epochs, updates=4960, warmup5

- 2026-10-11 20:44 KST — v8 actual-epoch oracle96: train complete P0_r10_lead14_seed123

- 2026-10-11 20:44 KST — v8 actual-epoch oracle96: queue launch Adv_r25_lead7_seed42 phase=train GPU7 PID1978498

- 2026-10-11 20:45 KST — v8 actual-epoch oracle96: train start Adv_r25_lead7_seed42 GPU7 PID1978498, actual80 epochs, updates=4960, warmup5

- 2026-10-11 21:13 KST — v8 actual-epoch oracle96: train complete NoAdv_r10_lead14_seed123

- 2026-10-11 21:14 KST — v8 actual-epoch oracle96: queue launch Baseline_r25_lead7_seed7 phase=train GPU9 PID2013746

- 2026-10-11 21:14 KST — v8 actual-epoch oracle96: train start Baseline_r25_lead7_seed7 GPU9 PID2013746, actual80 epochs, updates=4960, warmup5

- 2026-10-11 21:18 KST — v8 actual-epoch oracle96: train complete Adv_r10_lead14_seed123

- 2026-10-11 21:18 KST — v8 actual-epoch oracle96: queue launch P0_r25_lead7_seed7 phase=train GPU7 PID2020399

- 2026-10-11 21:18 KST — v8 actual-epoch oracle96: train start P0_r25_lead7_seed7 GPU7 PID2020399, actual80 epochs, updates=4960, warmup5

- 2026-10-11 21:28 KST — v8 actual-epoch oracle96: train complete Baseline_r25_lead7_seed42

- 2026-10-11 21:28 KST — v8 actual-epoch oracle96: queue launch NoAdv_r25_lead7_seed7 phase=train GPU8 PID2035845

- 2026-10-11 21:29 KST — v8 actual-epoch oracle96: train start NoAdv_r25_lead7_seed7 GPU8 PID2035845, actual80 epochs, updates=4960, warmup5

- 2026-10-11 21:32 KST — v8 actual-epoch oracle96: train complete P0_r25_lead7_seed42

- 2026-10-11 21:32 KST — v8 actual-epoch oracle96: queue launch Adv_r25_lead7_seed7 phase=train GPU8 PID2041714

- 2026-10-11 21:33 KST — v8 actual-epoch oracle96: train start Adv_r25_lead7_seed7 GPU8 PID2041714, actual80 epochs, updates=4960, warmup5

- 2026-10-11 21:33 KST — v8 actual-epoch oracle96: train complete NoAdv_r25_lead7_seed42

- 2026-10-11 21:33 KST — v8 actual-epoch oracle96: queue launch Baseline_r25_lead7_seed123 phase=train GPU9 PID2043364

- 2026-10-11 21:34 KST — v8 actual-epoch oracle96: train start Baseline_r25_lead7_seed123 GPU9 PID2043364, actual80 epochs, updates=4960, warmup5

- 2026-10-11 21:35 KST — v8 actual-epoch oracle96: train complete Adv_r25_lead7_seed42

- 2026-10-11 21:35 KST — v8 actual-epoch oracle96: queue launch P0_r25_lead7_seed123 phase=train GPU7 PID2046212

- 2026-10-11 21:36 KST — v8 actual-epoch oracle96: train start P0_r25_lead7_seed123 GPU7 PID2046212, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:02 KST — v8 actual-epoch oracle96: train complete Baseline_r25_lead7_seed7

- 2026-10-11 22:03 KST — v8 actual-epoch oracle96: queue launch NoAdv_r25_lead7_seed123 phase=train GPU9 PID2086897

- 2026-10-11 22:03 KST — v8 actual-epoch oracle96: train start NoAdv_r25_lead7_seed123 GPU9 PID2086897, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:11 KST — v8 actual-epoch oracle96: train complete P0_r25_lead7_seed7

- 2026-10-11 22:11 KST — v8 actual-epoch oracle96: queue launch Adv_r25_lead7_seed123 phase=train GPU7 PID2100239

- 2026-10-11 22:12 KST — v8 actual-epoch oracle96: train start Adv_r25_lead7_seed123 GPU7 PID2100239, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:22 KST — v8 actual-epoch oracle96: train complete Baseline_r25_lead7_seed123

- 2026-10-11 22:22 KST — v8 actual-epoch oracle96: queue launch Baseline_r25_lead14_seed42 phase=train GPU9 PID2116196

- 2026-10-11 22:22 KST — v8 actual-epoch oracle96: train complete NoAdv_r25_lead7_seed7

- 2026-10-11 22:22 KST — v8 actual-epoch oracle96: queue launch P0_r25_lead14_seed42 phase=train GPU8 PID2116536

- 2026-10-11 22:22 KST — v8 actual-epoch oracle96: train start Baseline_r25_lead14_seed42 GPU9 PID2116196, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:23 KST — v8 actual-epoch oracle96: train start P0_r25_lead14_seed42 GPU8 PID2116536, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:27 KST — v8 actual-epoch oracle96: train complete P0_r25_lead7_seed123

- 2026-10-11 22:27 KST — v8 actual-epoch oracle96: train complete Adv_r25_lead7_seed7

- 2026-10-11 22:28 KST — v8 actual-epoch oracle96: queue launch NoAdv_r25_lead14_seed42 phase=train GPU7 PID2124371

- 2026-10-11 22:28 KST — v8 actual-epoch oracle96: queue launch Adv_r25_lead14_seed42 phase=train GPU8 PID2124990

- 2026-10-11 22:28 KST — v8 actual-epoch oracle96: train start NoAdv_r25_lead14_seed42 GPU7 PID2124371, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:28 KST — v8 actual-epoch oracle96: train start Adv_r25_lead14_seed42 GPU8 PID2124990, actual80 epochs, updates=4960, warmup5

- 2026-10-11 22:53 KST — v8 actual-epoch oracle96: train complete NoAdv_r25_lead7_seed123

- 2026-10-11 22:53 KST — v8 actual-epoch oracle96: queue launch Baseline_r25_lead14_seed7 phase=train GPU9 PID2157868

- 2026-10-11 22:54 KST — v8 actual-epoch oracle96: train start Baseline_r25_lead14_seed7 GPU9 PID2157868, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:04 KST — v8 actual-epoch oracle96: train complete Adv_r25_lead7_seed123

- 2026-10-11 23:04 KST — v8 actual-epoch oracle96: queue launch P0_r25_lead14_seed7 phase=train GPU7 PID2170753

- 2026-10-11 23:05 KST — v8 actual-epoch oracle96: train start P0_r25_lead14_seed7 GPU7 PID2170753, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:11 KST — v8 actual-epoch oracle96: train complete Baseline_r25_lead14_seed42

- 2026-10-11 23:11 KST — v8 actual-epoch oracle96: queue launch NoAdv_r25_lead14_seed7 phase=train GPU9 PID2179138

- 2026-10-11 23:12 KST — v8 actual-epoch oracle96: train start NoAdv_r25_lead14_seed7 GPU9 PID2179138, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:18 KST — v8 actual-epoch oracle96: train complete NoAdv_r25_lead14_seed42

- 2026-10-11 23:18 KST — v8 actual-epoch oracle96: queue launch Adv_r25_lead14_seed7 phase=train GPU7 PID2187584

- 2026-10-11 23:19 KST — v8 actual-epoch oracle96: train complete P0_r25_lead14_seed42

- 2026-10-11 23:19 KST — v8 actual-epoch oracle96: queue launch Baseline_r25_lead14_seed123 phase=train GPU8 PID2187875

- 2026-10-11 23:19 KST — v8 actual-epoch oracle96: train start Adv_r25_lead14_seed7 GPU7 PID2187584, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:19 KST — v8 actual-epoch oracle96: train start Baseline_r25_lead14_seed123 GPU8 PID2187875, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:24 KST — v8 actual-epoch oracle96: train complete Adv_r25_lead14_seed42

- 2026-10-11 23:25 KST — v8 actual-epoch oracle96: queue launch P0_r25_lead14_seed123 phase=train GPU8 PID2195130

- 2026-10-11 23:26 KST — v8 actual-epoch oracle96: train start P0_r25_lead14_seed123 GPU8 PID2195130, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:41 KST — v8 actual-epoch oracle96: train complete Baseline_r25_lead14_seed7

- 2026-10-11 23:42 KST — v8 actual-epoch oracle96: queue launch NoAdv_r25_lead14_seed123 phase=train GPU9 PID2214124

- 2026-10-11 23:42 KST — v8 actual-epoch oracle96: train start NoAdv_r25_lead14_seed123 GPU9 PID2214124, actual80 epochs, updates=4960, warmup5

- 2026-10-11 23:57 KST — v8 actual-epoch oracle96: train complete P0_r25_lead14_seed7

- 2026-10-11 23:57 KST — v8 actual-epoch oracle96: queue launch Adv_r25_lead14_seed123 phase=train GPU7 PID2234163

- 2026-10-11 23:58 KST — v8 actual-epoch oracle96: train start Adv_r25_lead14_seed123 GPU7 PID2234163, actual80 epochs, updates=4960, warmup5

- 2026-10-12 00:03 KST — v8 actual-epoch oracle96: train complete NoAdv_r25_lead14_seed7

- 2026-10-12 00:03 KST — v8 actual-epoch oracle96: queue launch Baseline_r50_lead7_seed42 phase=train GPU9 PID2241544

- 2026-10-12 00:04 KST — v8 actual-epoch oracle96: train start Baseline_r50_lead7_seed42 GPU9 PID2241544, actual80 epochs, updates=9920, warmup5

- 2026-10-12 00:07 KST — v8 actual-epoch oracle96: train complete Baseline_r25_lead14_seed123

- 2026-10-12 00:07 KST — v8 actual-epoch oracle96: queue launch P0_r50_lead7_seed42 phase=train GPU8 PID2248471

- 2026-10-12 00:08 KST — v8 actual-epoch oracle96: train start P0_r50_lead7_seed42 GPU8 PID2248471, actual80 epochs, updates=9920, warmup5

- 2026-10-12 00:13 KST — v8 actual-epoch oracle96: train complete Adv_r25_lead14_seed7

- 2026-10-12 00:13 KST — v8 actual-epoch oracle96: queue launch NoAdv_r50_lead7_seed42 phase=train GPU7 PID2257992

- 2026-10-12 00:14 KST — v8 actual-epoch oracle96: train start NoAdv_r50_lead7_seed42 GPU7 PID2257992, actual80 epochs, updates=9920, warmup5

- 2026-10-12 00:19 KST — v8 actual-epoch oracle96: train complete P0_r25_lead14_seed123

- 2026-10-12 00:19 KST — v8 actual-epoch oracle96: queue launch Adv_r50_lead7_seed42 phase=train GPU8 PID2266994

- 2026-10-12 00:20 KST — v8 actual-epoch oracle96: train start Adv_r50_lead7_seed42 GPU8 PID2266994, actual80 epochs, updates=9920, warmup5

- 2026-10-12 00:34 KST — v8 actual-epoch oracle96: train complete NoAdv_r25_lead14_seed123

- 2026-10-12 00:34 KST — v8 actual-epoch oracle96: queue launch Baseline_r50_lead7_seed7 phase=train GPU9 PID2290913

- 2026-10-12 00:34 KST — v8 actual-epoch oracle96: train start Baseline_r50_lead7_seed7 GPU9 PID2290913, actual80 epochs, updates=9920, warmup5

- 2026-10-12 00:48 KST — v8 actual-epoch oracle96: train complete Adv_r25_lead14_seed123

- 2026-10-12 00:48 KST — v8 actual-epoch oracle96: queue launch P0_r50_lead7_seed7 phase=train GPU7 PID2314079

- 2026-10-12 00:49 KST — v8 actual-epoch oracle96: train start P0_r50_lead7_seed7 GPU7 PID2314079, actual80 epochs, updates=9920, warmup5

- 2026-10-12 01:10 KST — v8 actual-epoch oracle96: train complete Baseline_r50_lead7_seed42

- 2026-10-12 01:11 KST — v8 actual-epoch oracle96: queue launch NoAdv_r50_lead7_seed7 phase=train GPU9 PID2348648

- 2026-10-12 01:11 KST — v8 actual-epoch oracle96: train start NoAdv_r50_lead7_seed7 GPU9 PID2348648, actual80 epochs, updates=9920, warmup5

- 2026-10-12 01:22 KST — v8 actual-epoch oracle96: train complete P0_r50_lead7_seed42

- 2026-10-12 01:23 KST — v8 actual-epoch oracle96: queue launch Adv_r50_lead7_seed7 phase=train GPU8 PID2364385

- 2026-10-12 01:23 KST — v8 actual-epoch oracle96: train start Adv_r50_lead7_seed7 GPU8 PID2364385, actual80 epochs, updates=9920, warmup5

- 2026-10-12 01:25 KST — v8 actual-epoch oracle96: train complete NoAdv_r50_lead7_seed42

- 2026-10-12 01:25 KST — v8 actual-epoch oracle96: queue launch Baseline_r50_lead7_seed123 phase=train GPU7 PID2367571

- 2026-10-12 01:26 KST — v8 actual-epoch oracle96: train start Baseline_r50_lead7_seed123 GPU7 PID2367571, actual80 epochs, updates=9920, warmup5

- 2026-10-12 01:35 KST — v8 actual-epoch oracle96: train complete Adv_r50_lead7_seed42

- 2026-10-12 01:35 KST — v8 actual-epoch oracle96: queue launch P0_r50_lead7_seed123 phase=train GPU8 PID2379431

- 2026-10-12 01:36 KST — v8 actual-epoch oracle96: train start P0_r50_lead7_seed123 GPU8 PID2379431, actual80 epochs, updates=9920, warmup5

- 2026-10-12 01:40 KST — v8 actual-epoch oracle96: train complete Baseline_r50_lead7_seed7

- 2026-10-12 01:40 KST — v8 actual-epoch oracle96: queue launch NoAdv_r50_lead7_seed123 phase=train GPU9 PID2385714

- 2026-10-12 01:41 KST — v8 actual-epoch oracle96: train start NoAdv_r50_lead7_seed123 GPU9 PID2385714, actual80 epochs, updates=9920, warmup5

- 2026-10-12 02:01 KST — v8 actual-epoch oracle96: train complete P0_r50_lead7_seed7

- 2026-10-12 02:01 KST — v8 actual-epoch oracle96: queue launch Adv_r50_lead7_seed123 phase=train GPU7 PID2411179

- 2026-10-12 02:02 KST — v8 actual-epoch oracle96: train start Adv_r50_lead7_seed123 GPU7 PID2411179, actual80 epochs, updates=9920, warmup5

- 2026-10-12 02:22 KST — v8 actual-epoch oracle96: train complete NoAdv_r50_lead7_seed7

- 2026-10-12 02:22 KST — v8 actual-epoch oracle96: queue launch Baseline_r50_lead14_seed42 phase=train GPU9 PID2436437

- 2026-10-12 02:23 KST — v8 actual-epoch oracle96: train start Baseline_r50_lead14_seed42 GPU9 PID2436437, actual80 epochs, updates=9920, warmup5

- 2026-10-12 02:27 KST — v8 actual-epoch oracle96: train complete Baseline_r50_lead7_seed123

- 2026-10-12 02:27 KST — v8 actual-epoch oracle96: queue launch P0_r50_lead14_seed42 phase=train GPU7 PID2443475

- 2026-10-12 02:28 KST — v8 actual-epoch oracle96: train start P0_r50_lead14_seed42 GPU7 PID2443475, actual80 epochs, updates=9920, warmup5

- 2026-10-12 02:41 KST — v8 actual-epoch oracle96: train complete Adv_r50_lead7_seed7

- 2026-10-12 02:42 KST — v8 actual-epoch oracle96: queue launch NoAdv_r50_lead14_seed42 phase=train GPU8 PID2462807

- 2026-10-12 02:42 KST — v8 actual-epoch oracle96: train start NoAdv_r50_lead14_seed42 GPU8 PID2462807, actual80 epochs, updates=9920, warmup5

- 2026-10-12 02:52 KST — v8 actual-epoch oracle96: train complete NoAdv_r50_lead7_seed123

- 2026-10-12 02:52 KST — v8 actual-epoch oracle96: queue launch Adv_r50_lead14_seed42 phase=train GPU9 PID2476175

- 2026-10-12 02:53 KST — v8 actual-epoch oracle96: train start Adv_r50_lead14_seed42 GPU9 PID2476175, actual80 epochs, updates=9920, warmup5

- 2026-10-12 02:53 KST — v8 actual-epoch oracle96: train complete P0_r50_lead7_seed123

- 2026-10-12 02:53 KST — v8 actual-epoch oracle96: queue launch Baseline_r50_lead14_seed7 phase=train GPU8 PID2477895

- 2026-10-12 02:54 KST — v8 actual-epoch oracle96: train start Baseline_r50_lead14_seed7 GPU8 PID2477895, actual80 epochs, updates=9920, warmup5

- 2026-10-12 03:14 KST — v8 actual-epoch oracle96: train complete Adv_r50_lead7_seed123

- 2026-10-12 03:14 KST — v8 actual-epoch oracle96: queue launch P0_r50_lead14_seed7 phase=train GPU7 PID2508518

- 2026-10-12 03:15 KST — v8 actual-epoch oracle96: train start P0_r50_lead14_seed7 GPU7 PID2508518, actual80 epochs, updates=9920, warmup5

- 2026-10-12 03:27 KST — v8 actual-epoch oracle96: train complete Baseline_r50_lead14_seed42

- 2026-10-12 03:27 KST — v8 actual-epoch oracle96: queue launch NoAdv_r50_lead14_seed7 phase=train GPU9 PID2524520

- 2026-10-12 03:28 KST — v8 actual-epoch oracle96: train start NoAdv_r50_lead14_seed7 GPU9 PID2524520, actual80 epochs, updates=9920, warmup5

- 2026-10-12 03:45 KST — v8 actual-epoch oracle96: train complete P0_r50_lead14_seed42

- 2026-10-12 03:45 KST — v8 actual-epoch oracle96: queue launch Adv_r50_lead14_seed7 phase=train GPU7 PID2546055

- 2026-10-12 03:45 KST — v8 actual-epoch oracle96: train start Adv_r50_lead14_seed7 GPU7 PID2546055, actual80 epochs, updates=9920, warmup5

- 2026-10-12 03:55 KST — v8 actual-epoch oracle96: train complete NoAdv_r50_lead14_seed42

- 2026-10-12 03:55 KST — v8 actual-epoch oracle96: queue launch Baseline_r50_lead14_seed123 phase=train GPU8 PID2558599

- 2026-10-12 03:55 KST — v8 actual-epoch oracle96: train start Baseline_r50_lead14_seed123 GPU8 PID2558599, actual80 epochs, updates=9920, warmup5

- 2026-10-12 03:57 KST — v8 actual-epoch oracle96: train complete Baseline_r50_lead14_seed7

- 2026-10-12 03:57 KST — v8 actual-epoch oracle96: queue launch P0_r50_lead14_seed123 phase=train GPU8 PID2561311

- 2026-10-12 03:58 KST — v8 actual-epoch oracle96: train start P0_r50_lead14_seed123 GPU8 PID2561311, actual80 epochs, updates=9920, warmup5

- 2026-10-12 04:10 KST — v8 actual-epoch oracle96: train complete Adv_r50_lead14_seed42

- 2026-10-12 04:10 KST — v8 actual-epoch oracle96: queue launch NoAdv_r50_lead14_seed123 phase=train GPU9 PID2579019

- 2026-10-12 04:11 KST — v8 actual-epoch oracle96: train start NoAdv_r50_lead14_seed123 GPU9 PID2579019, actual80 epochs, updates=9920, warmup5

- 2026-10-12 04:31 KST — v8 actual-epoch oracle96: train complete P0_r50_lead14_seed7

- 2026-10-12 04:31 KST — v8 actual-epoch oracle96: queue launch Adv_r50_lead14_seed123 phase=train GPU7 PID2606471

- 2026-10-12 04:32 KST — v8 actual-epoch oracle96: train start Adv_r50_lead14_seed123 GPU7 PID2606471, actual80 epochs, updates=9920, warmup5

- 2026-10-12 04:44 KST — v8 actual-epoch oracle96: train complete NoAdv_r50_lead14_seed7

- 2026-10-12 04:44 KST — v8 actual-epoch oracle96: queue launch Baseline_r100_lead7_seed42 phase=train GPU9 PID2622906

- 2026-10-12 04:45 KST — v8 actual-epoch oracle96: train start Baseline_r100_lead7_seed42 GPU9 PID2622906, actual80 epochs, updates=20000, warmup5

- 2026-10-12 04:57 KST — v8 actual-epoch oracle96: train complete Baseline_r50_lead14_seed123

- 2026-10-12 04:57 KST — v8 actual-epoch oracle96: queue launch P0_r100_lead7_seed42 phase=train GPU8 PID2638737

- 2026-10-12 04:58 KST — v8 actual-epoch oracle96: train start P0_r100_lead7_seed42 GPU8 PID2638737, actual80 epochs, updates=20000, warmup5

- 2026-10-12 05:00 KST — v8 actual-epoch oracle96: train complete Adv_r50_lead14_seed7

- 2026-10-12 05:00 KST — v8 actual-epoch oracle96: queue launch NoAdv_r100_lead7_seed42 phase=train GPU7 PID2643017

- 2026-10-12 05:01 KST — v8 actual-epoch oracle96: train start NoAdv_r100_lead7_seed42 GPU7 PID2643017, actual80 epochs, updates=20000, warmup5

- 2026-10-12 05:10 KST — v8 actual-epoch oracle96: train complete P0_r50_lead14_seed123

- 2026-10-12 05:10 KST — v8 actual-epoch oracle96: queue launch Adv_r100_lead7_seed42 phase=train GPU8 PID2655104

- 2026-10-12 05:11 KST — v8 actual-epoch oracle96: train start Adv_r100_lead7_seed42 GPU8 PID2655104, actual80 epochs, updates=20000, warmup5

- 2026-10-12 05:23 KST — v8 actual-epoch oracle96: train complete NoAdv_r50_lead14_seed123

- 2026-10-12 05:23 KST — v8 actual-epoch oracle96: queue launch Baseline_r100_lead7_seed7 phase=train GPU9 PID2671357

- 2026-10-12 05:24 KST — v8 actual-epoch oracle96: train start Baseline_r100_lead7_seed7 GPU9 PID2671357, actual80 epochs, updates=20000, warmup5

- 2026-10-12 05:45 KST — v8 actual-epoch oracle96: train complete Adv_r50_lead14_seed123

- 2026-10-12 05:45 KST — v8 actual-epoch oracle96: queue launch P0_r100_lead7_seed7 phase=train GPU7 PID2699144

- 2026-10-12 05:46 KST — v8 actual-epoch oracle96: train start P0_r100_lead7_seed7 GPU7 PID2699144, actual80 epochs, updates=20000, warmup5

- 2026-10-12 06:30 KST — v8 actual-epoch oracle96: train complete Baseline_r100_lead7_seed42

- 2026-10-12 06:30 KST — v8 actual-epoch oracle96: queue launch NoAdv_r100_lead7_seed7 phase=train GPU9 PID2751035

- 2026-10-12 06:31 KST — v8 actual-epoch oracle96: train start NoAdv_r100_lead7_seed7 GPU9 PID2751035, actual80 epochs, updates=20000, warmup5

- 2026-10-12 06:57 KST — v8 actual-epoch oracle96: train complete NoAdv_r100_lead7_seed42

- 2026-10-12 06:57 KST — v8 actual-epoch oracle96: queue launch Adv_r100_lead7_seed7 phase=train GPU7 PID2783883

- 2026-10-12 06:58 KST — v8 actual-epoch oracle96: train start Adv_r100_lead7_seed7 GPU7 PID2783883, actual80 epochs, updates=20000, warmup5

- 2026-10-12 07:02 KST — v8 actual-epoch oracle96: train complete P0_r100_lead7_seed42

- 2026-10-12 07:02 KST — v8 actual-epoch oracle96: queue launch Baseline_r100_lead7_seed123 phase=train GPU8 PID2790373

- 2026-10-12 07:03 KST — v8 actual-epoch oracle96: train start Baseline_r100_lead7_seed123 GPU8 PID2790373, actual80 epochs, updates=20000, warmup5

- 2026-10-12 07:10 KST — v8 actual-epoch oracle96: train complete Baseline_r100_lead7_seed7

- 2026-10-12 07:10 KST — v8 actual-epoch oracle96: queue launch P0_r100_lead7_seed123 phase=train GPU9 PID2800796

- 2026-10-12 07:11 KST — v8 actual-epoch oracle96: train start P0_r100_lead7_seed123 GPU9 PID2800796, actual80 epochs, updates=20000, warmup5

- 2026-10-12 07:15 KST — v8 actual-epoch oracle96: train complete Adv_r100_lead7_seed42

- 2026-10-12 07:15 KST — v8 actual-epoch oracle96: queue launch NoAdv_r100_lead7_seed123 phase=train GPU8 PID2806407

- 2026-10-12 07:15 KST — v8 actual-epoch oracle96: train start NoAdv_r100_lead7_seed123 GPU8 PID2806407, actual80 epochs, updates=20000, warmup5

- 2026-10-12 07:49 KST — v8 actual-epoch oracle96: train complete P0_r100_lead7_seed7

- 2026-10-12 07:49 KST — v8 actual-epoch oracle96: queue launch Adv_r100_lead7_seed123 phase=train GPU7 PID2849842

- 2026-10-12 07:50 KST — v8 actual-epoch oracle96: train start Adv_r100_lead7_seed123 GPU7 PID2849842, actual80 epochs, updates=20000, warmup5

- 2026-10-12 08:27 KST — v8 actual-epoch oracle96: train complete NoAdv_r100_lead7_seed7

- 2026-10-12 08:27 KST — v8 actual-epoch oracle96: queue launch Baseline_r100_lead14_seed42 phase=train GPU9 PID2907461

- 2026-10-12 08:28 KST — v8 actual-epoch oracle96: train start Baseline_r100_lead14_seed42 GPU9 PID2907461, actual80 epochs, updates=20000, warmup5

- 2026-10-12 08:40 KST — v8 actual-epoch oracle96: train complete Baseline_r100_lead7_seed123

- 2026-10-12 08:40 KST — v8 actual-epoch oracle96: queue launch P0_r100_lead14_seed42 phase=train GPU8 PID2929860

- 2026-10-12 08:41 KST — v8 actual-epoch oracle96: train start P0_r100_lead14_seed42 GPU8 PID2929860, actual80 epochs, updates=20000, warmup5

- 2026-10-12 09:00 KST — v8 actual-epoch oracle96: train complete NoAdv_r100_lead7_seed123

- 2026-10-12 09:00 KST — v8 actual-epoch oracle96: train complete Adv_r100_lead7_seed7

- 2026-10-12 09:00 KST — v8 actual-epoch oracle96: queue launch NoAdv_r100_lead14_seed42 phase=train GPU8 PID2963853

- 2026-10-12 09:00 KST — v8 actual-epoch oracle96: queue launch Adv_r100_lead14_seed42 phase=train GPU7 PID2964098

- 2026-10-12 09:00 KST — v8 actual-epoch oracle96: train start NoAdv_r100_lead14_seed42 GPU8 PID2963853, actual80 epochs, updates=20000, warmup5

- 2026-10-12 09:00 KST — v8 actual-epoch oracle96: train start Adv_r100_lead14_seed42 GPU7 PID2964098, actual80 epochs, updates=20000, warmup5

- 2026-10-12 09:09 KST — v8 actual-epoch oracle96: train complete P0_r100_lead7_seed123

- 2026-10-12 09:09 KST — v8 actual-epoch oracle96: queue launch Baseline_r100_lead14_seed7 phase=train GPU9 PID2979430

- 2026-10-12 09:09 KST — v8 actual-epoch oracle96: train start Baseline_r100_lead14_seed7 GPU9 PID2979430, actual80 epochs, updates=20000, warmup5

- 2026-10-12 09:51 KST — v8 actual-epoch oracle96: train complete Adv_r100_lead7_seed123

- 2026-10-12 09:51 KST — v8 actual-epoch oracle96: queue launch P0_r100_lead14_seed7 phase=train GPU7 PID3049194

- 2026-10-12 09:52 KST — v8 actual-epoch oracle96: train start P0_r100_lead14_seed7 GPU7 PID3049194, actual80 epochs, updates=20000, warmup5

- 2026-10-12 10:13 KST — v8 actual-epoch oracle96: train complete Baseline_r100_lead14_seed42

- 2026-10-12 10:14 KST — v8 actual-epoch oracle96: queue launch NoAdv_r100_lead14_seed7 phase=train GPU9 PID3084607

- 2026-10-12 10:14 KST — v8 actual-epoch oracle96: train start NoAdv_r100_lead14_seed7 GPU9 PID3084607, actual80 epochs, updates=20000, warmup5

- 2026-10-12 10:42 KST — v8 actual-epoch oracle96: train complete P0_r100_lead14_seed42

- 2026-10-12 10:42 KST — v8 actual-epoch oracle96: queue launch Adv_r100_lead14_seed7 phase=train GPU8 PID3128979

- 2026-10-12 10:45 KST — v8 actual-epoch oracle96: train start Adv_r100_lead14_seed7 GPU8 PID3128979, actual80 epochs, updates=20000, warmup5

- 2026-10-12 10:55 KST — v8 actual-epoch oracle96: train complete Baseline_r100_lead14_seed7

- 2026-10-12 10:55 KST — v8 actual-epoch oracle96: queue launch Baseline_r100_lead14_seed123 phase=train GPU9 PID3149292

- 2026-10-12 10:56 KST — v8 actual-epoch oracle96: train start Baseline_r100_lead14_seed123 GPU9 PID3149292, actual80 epochs, updates=20000, warmup5

- 2026-10-12 11:02 KST — v8 actual-epoch oracle96: train complete NoAdv_r100_lead14_seed42

- 2026-10-12 11:02 KST — v8 actual-epoch oracle96: queue launch P0_r100_lead14_seed123 phase=train GPU8 PID3159917

- 2026-10-12 11:03 KST — v8 actual-epoch oracle96: train start P0_r100_lead14_seed123 GPU8 PID3159917, actual80 epochs, updates=20000, warmup5

- 2026-10-12 11:07 KST — v8 actual-epoch oracle96: train complete Adv_r100_lead14_seed42

- 2026-10-12 11:07 KST — v8 actual-epoch oracle96: queue launch NoAdv_r100_lead14_seed123 phase=train GPU7 PID3167165

- 2026-10-12 11:08 KST — v8 actual-epoch oracle96: train start NoAdv_r100_lead14_seed123 GPU7 PID3167165, actual80 epochs, updates=20000, warmup5

- 2026-10-12 11:55 KST — v8 actual-epoch oracle96: train complete P0_r100_lead14_seed7

- 2026-10-12 11:55 KST — v8 actual-epoch oracle96: queue launch Adv_r100_lead14_seed123 phase=train GPU7 PID3240197

- 2026-10-12 11:58 KST — v8 actual-epoch oracle96: train start Adv_r100_lead14_seed123 GPU7 PID3240197, actual80 epochs, updates=20000, warmup5

- 2026-10-12 12:04 KST — v8 actual-epoch oracle96: train complete NoAdv_r100_lead14_seed7

- 2026-10-12 12:27 KST — v8 actual-epoch oracle96: train complete Baseline_r100_lead14_seed123

- 2026-10-12 12:54 KST — v8 actual-epoch oracle96: train complete Adv_r100_lead14_seed7

- 2026-10-12 13:00 KST — v8 actual-epoch oracle96: train complete NoAdv_r100_lead14_seed123

- 2026-10-12 13:06 KST — v8 actual-epoch oracle96: train complete P0_r100_lead14_seed123

- 2026-10-12 13:47 KST — v8 actual-epoch oracle96: train complete Adv_r100_lead14_seed123

- 2026-10-12 13:48 KST — v8 actual-epoch oracle96: Actual epoch 96 training/Validation complete; Test intentionally NOT_RUN; report.md generated

- 2026-10-12 14:23 KST — Actual Epoch96 완료 결과 CPU 재검토. `outputs/v8_actual_epoch_oracle96_20261011/results_review_20261012/`에 검토코드·한국어보고·32조건/96run CSV·검증JSON·전체비교PNG/PDF 추가. 모든96개80epochs 및checkpoint재로드192기록/샘플·스케줄·초기화·순서·forcing독립성 PASS. NoAdv10%Lead7 평균paired개선3.045%,3/3seed; P0일관우위없음, Adv모든ratio×lead평균악화. Best7–54/중앙22, warmup전0/96. Test·새학습·GPU계산 미실행, 기존결과보존.

- 2026-10-12 14:42 KST — v8 actual-epoch oracle96: scripts/oracle96_test/*.py 신규 생성: 고정 Validation-best 96개 Test SST/MHW 추론·미래 forcing 불변성 검사·별도 결과 집계. 기존 학습 코드/통계/체크포인트 보존. CPU 지표 회귀 및 96 체크포인트 SHA 검증 PASS. GPU7/8/9 각 최대2, 다른 작업 메모리 고려.

- 2026-10-12 14:43 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r10_lead7_seed42 GPU7 PID3581837; no training

- 2026-10-12 14:43 KST — v8 actual-epoch oracle96: Test inference launch P0_r10_lead7_seed42 GPU8 PID3581838; no training

- 2026-10-12 14:43 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r10_lead7_seed42 GPU9 PID3581839; no training

- 2026-10-12 14:43 KST — v8 actual-epoch oracle96: Test inference launch Adv_r10_lead7_seed42 GPU7 PID3581991; no training

- 2026-10-12 14:44 KST — v8 actual-epoch oracle96: Test tmux 시작: socket=/ceph_data/yeonwoo/test_20261012.sock, session=v8_actual_test_gpu789, controller PID3581827. 첫 4개 PID3581837/3581838/3581839/3581991가 GPU7/8/9에 yeonwoo 이름으로 표시됨을 NVML 확인. 최초 소켓은 UNIX 경로 길이 제한으로 생성되지 않아 짧은 Ceph 경로 사용.

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r10_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r10_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r10_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r10_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r10_lead7_seed7 GPU7 PID3587413; no training

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test inference launch P0_r10_lead7_seed7 GPU8 PID3587414; no training

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r10_lead7_seed7 GPU9 PID3587416; no training

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test inference launch Adv_r10_lead7_seed7 GPU7 PID3587737; no training

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r10_lead7_seed123 GPU8 PID3587738; no training

- 2026-10-12 14:45 KST — v8 actual-epoch oracle96: Test inference launch P0_r10_lead7_seed123 GPU9 PID3587739; no training

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r10_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r10_lead7_seed123 GPU9 PID3590681; no training

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r10_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test inference launch Adv_r10_lead7_seed123 GPU9 PID3590837; no training

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r10_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r10_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r10_lead14_seed42 GPU7 PID3591379; no training

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test inference launch P0_r10_lead14_seed42 GPU8 PID3591381; no training

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r10_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r10_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r10_lead14_seed42 GPU8 PID3591667; no training

- 2026-10-12 14:47 KST — v8 actual-epoch oracle96: Test inference launch Adv_r10_lead14_seed42 GPU7 PID3591842; no training

- 2026-10-12 14:48 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r10_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:48 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r10_lead14_seed7 GPU9 PID3594394; no training

- 2026-10-12 14:48 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r10_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:48 KST — v8 actual-epoch oracle96: Test inference launch P0_r10_lead14_seed7 GPU9 PID3594641; no training

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r10_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r10_lead14_seed7 GPU7 PID3595836; no training

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r10_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test inference launch Adv_r10_lead14_seed7 GPU8 PID3595999; no training

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r10_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r10_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r10_lead14_seed123 GPU8 PID3596146; no training

- 2026-10-12 14:49 KST — v8 actual-epoch oracle96: Test inference launch P0_r10_lead14_seed123 GPU7 PID3596393; no training

- 2026-10-12 14:50 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r10_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:50 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r10_lead14_seed123 GPU9 PID3598120; no training

- 2026-10-12 14:50 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r10_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:50 KST — v8 actual-epoch oracle96: Test inference launch Adv_r10_lead14_seed123 GPU9 PID3598398; no training

- 2026-10-12 14:50 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r10_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:50 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r10_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r25_lead7_seed42 GPU7 PID3600315; no training

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test inference launch P0_r25_lead7_seed42 GPU8 PID3600316; no training

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r10_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r25_lead7_seed42 GPU8 PID3600628; no training

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r10_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test inference launch Adv_r25_lead7_seed42 GPU7 PID3600799; no training

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r10_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r25_lead7_seed7 GPU9 PID3601666; no training

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r10_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:51 KST — v8 actual-epoch oracle96: Test inference launch P0_r25_lead7_seed7 GPU9 PID3602134; no training

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r25_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r25_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r25_lead7_seed7 GPU7 PID3604589; no training

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test inference launch Adv_r25_lead7_seed7 GPU8 PID3604590; no training

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r25_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r25_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r25_lead7_seed123 GPU8 PID3604984; no training

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test inference launch P0_r25_lead7_seed123 GPU7 PID3605191; no training

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r25_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:52 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r25_lead7_seed123 GPU9 PID3605392; no training

- 2026-10-12 14:53 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r25_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:53 KST — v8 actual-epoch oracle96: Test inference launch Adv_r25_lead7_seed123 GPU9 PID3605784; no training

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r25_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r25_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r25_lead14_seed42 GPU9 PID3608946; no training

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r25_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test inference launch P0_r25_lead14_seed42 GPU7 PID3609176; no training

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r25_lead14_seed42 GPU8 PID3609178; no training

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r25_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test inference launch Adv_r25_lead14_seed42 GPU9 PID3609399; no training

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r25_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r25_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r25_lead14_seed7 GPU7 PID3609674; no training

- 2026-10-12 14:54 KST — v8 actual-epoch oracle96: Test inference launch P0_r25_lead14_seed7 GPU8 PID3609676; no training

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r25_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r25_lead14_seed7 GPU9 PID3612469; no training

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r25_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r25_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test inference launch Adv_r25_lead14_seed7 GPU7 PID3612749; no training

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r25_lead14_seed123 GPU9 PID3612750; no training

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r25_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:55 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r25_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:56 KST — v8 actual-epoch oracle96: Test inference launch P0_r25_lead14_seed123 GPU7 PID3613163; no training

- 2026-10-12 14:56 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r25_lead14_seed123 GPU8 PID3613164; no training

- 2026-10-12 14:56 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r25_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:56 KST — v8 actual-epoch oracle96: Test inference launch Adv_r25_lead14_seed123 GPU8 PID3613573; no training

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r25_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r25_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r50_lead7_seed42 GPU7 PID3616117; no training

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test inference launch P0_r50_lead7_seed42 GPU9 PID3616118; no training

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r25_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r25_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r50_lead7_seed42 GPU8 PID3616449; no training

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test inference launch Adv_r50_lead7_seed42 GPU9 PID3616450; no training

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r25_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r50_lead7_seed7 GPU7 PID3616702; no training

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r25_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 14:57 KST — v8 actual-epoch oracle96: Test inference launch P0_r50_lead7_seed7 GPU8 PID3617034; no training

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r50_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r50_lead7_seed7 GPU7 PID3619675; no training

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r50_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test inference launch Adv_r50_lead7_seed7 GPU9 PID3619845; no training

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r50_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r50_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r50_lead7_seed123 GPU8 PID3620070; no training

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r50_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test inference launch P0_r50_lead7_seed123 GPU7 PID3620228; no training

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r50_lead7_seed123 GPU9 PID3620229; no training

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r50_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 14:58 KST — v8 actual-epoch oracle96: Test inference launch Adv_r50_lead7_seed123 GPU8 PID3620550; no training

- 2026-10-12 14:59 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r50_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r50_lead14_seed42 GPU7 PID3623179; no training

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r50_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r50_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test inference launch P0_r50_lead14_seed42 GPU8 PID3623408; no training

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r50_lead14_seed42 GPU9 PID3623409; no training

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r50_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test inference launch Adv_r50_lead14_seed42 GPU7 PID3623690; no training

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r50_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r50_lead14_seed7 GPU9 PID3623848; no training

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r50_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:00 KST — v8 actual-epoch oracle96: Test inference launch P0_r50_lead14_seed7 GPU8 PID3624189; no training

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r50_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r50_lead14_seed7 GPU7 PID3626868; no training

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r50_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r50_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r50_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test inference launch Adv_r50_lead14_seed7 GPU7 PID3627180; no training

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r50_lead14_seed123 GPU8 PID3627181; no training

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test inference launch P0_r50_lead14_seed123 GPU9 PID3627182; no training

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r50_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r50_lead14_seed123 GPU9 PID3627557; no training

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r50_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:01 KST — v8 actual-epoch oracle96: Test inference launch Adv_r50_lead14_seed123 GPU8 PID3627726; no training

- 2026-10-12 15:02 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r50_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:02 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r100_lead7_seed42 GPU7 PID3630190; no training

- 2026-10-12 15:02 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r50_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:02 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r50_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:02 KST — v8 actual-epoch oracle96: Test inference launch P0_r100_lead7_seed42 GPU7 PID3630683; no training

- 2026-10-12 15:02 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r100_lead7_seed42 GPU8 PID3630685; no training

- 2026-10-12 15:03 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r50_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:03 KST — v8 actual-epoch oracle96: Test inference launch Adv_r100_lead7_seed42 GPU9 PID3630886; no training

- 2026-10-12 15:03 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r50_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:03 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r50_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:03 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r100_lead7_seed7 GPU8 PID3631162; no training

- 2026-10-12 15:03 KST — v8 actual-epoch oracle96: Test inference launch P0_r100_lead7_seed7 GPU9 PID3631163; no training

- 2026-10-12 15:04 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r100_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:04 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r100_lead7_seed7 GPU7 PID3633243; no training

- 2026-10-12 15:04 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r100_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:04 KST — v8 actual-epoch oracle96: Test inference launch Adv_r100_lead7_seed7 GPU7 PID3633508; no training

- 2026-10-12 15:04 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r100_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r100_lead7_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r100_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r100_lead7_seed123 GPU9 PID3634918; no training

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test inference launch P0_r100_lead7_seed123 GPU8 PID3634919; no training

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r100_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r100_lead7_seed123 GPU9 PID3634997; no training

- 2026-10-12 15:05 KST — v8 actual-epoch oracle96: Test inference launch Adv_r100_lead7_seed123 GPU8 PID3635140; no training

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r100_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r100_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r100_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r100_lead7_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r100_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r100_lead14_seed42 GPU7 PID3638732; no training

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test inference launch P0_r100_lead14_seed42 GPU9 PID3638733; no training

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r100_lead14_seed42 GPU8 PID3638734; no training

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test inference launch Adv_r100_lead14_seed42 GPU7 PID3638844; no training

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r100_lead7_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r100_lead14_seed7 GPU9 PID3638851; no training

- 2026-10-12 15:07 KST — v8 actual-epoch oracle96: Test inference launch P0_r100_lead14_seed7 GPU8 PID3638929; no training

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r100_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r100_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r100_lead14_seed7 GPU8 PID3642382; no training

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r100_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r100_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test inference launch Adv_r100_lead14_seed7 GPU9 PID3642606; no training

- 2026-10-12 15:08 KST — v8 actual-epoch oracle96: Test inference launch Baseline_r100_lead14_seed123 GPU8 PID3642607; no training

- 2026-10-12 15:09 KST — v8 actual-epoch oracle96: Test inference launch P0_r100_lead14_seed123 GPU9 PID3642812; no training

- 2026-10-12 15:09 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r100_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:09 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r100_lead14_seed42, overall Validation-best only; no training updates

- 2026-10-12 15:09 KST — v8 actual-epoch oracle96: Test inference launch NoAdv_r100_lead14_seed123 GPU7 PID3643176; no training

- 2026-10-12 15:09 KST — v8 actual-epoch oracle96: Test inference launch Adv_r100_lead14_seed123 GPU7 PID3643377; no training

- 2026-10-12 15:10 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r100_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:10 KST — v8 actual-epoch oracle96: Test evaluation complete Baseline_r100_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:10 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r100_lead14_seed7, overall Validation-best only; no training updates

- 2026-10-12 15:10 KST — v8 actual-epoch oracle96: Test evaluation complete P0_r100_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:10 KST — v8 actual-epoch oracle96: Test evaluation complete NoAdv_r100_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:10 KST — v8 actual-epoch oracle96: Test evaluation complete Adv_r100_lead14_seed123, overall Validation-best only; no training updates

- 2026-10-12 15:11 KST — v8 actual-epoch oracle96: Test 96/96 SST/MHW 평가 및 forcing 불변성 검사 완료; 별도 report_ko.md/comparison_summary.csv 생성.

- 2026-10-15 01:44 KST — Auxiliary24 preflight: scripts/oracle96_auxiliary24/audit.py 및 별도 outputs/v8_actual_epoch_auxiliary24_20261015 생성. 기존 λaux=0.0001 정의 감사와 실제 Train 3 batch GPU5 gradient/초기값/sampling/LR/체크포인트 검증. Rres 마지막257개 parameter만0으로 하면 SST예측 동일·auxloss/공유gradient=0 확인. 첨부 중단 조건에 따라 24개 본학습 미시작, Test 미실행. 기존 코드/데이터/실험 결과 보존.

- 2026-10-15 01:44 KST — Auxiliary24: scripts/oracle96_auxiliary24/report_audit.py 및 요청 산출물 작성. 대표 실제 batch에서 초기 gradient PASS, head-only zero 퇴화해 배제 FAIL; 24개 계획을 BLOCKED_PREFLIGHT로 기록하고 본학습/Test 미실행. 수치/코드근거/NOT_RUN 범위 및 이전 실험의 정규화 효과 가능성을 구분하여 report_ko.md에 명시.

- 2026-10-15 01:53 KST — Auxiliary24: zero-head 최소해를 사용자 지시로 비차단 정규화 한계로 재분류; 원 감사 original_preflight/ 복사 보존. aux_common/aux_worker/prepare_training/queue/summarize 신규 추가. 원 모델·손실 그대로 λaux=.0001, GPU5/6/7. 24×80epoch CPU sampling/LR PASS; GPU 비간섭 smoke 대기. Test 금지.

- 2026-10-15 01:54 KST — Auxiliary24: 본 학습 전 초기tensor 비교의 RNG guard 보강: Python/NumPy/CPU/CUDA를 모두 복원. 이전 protocol SHA snapshot 보존 후 소스해시 갱신. CPU24×80 sampling/LR검사 유지; 아직 training0/24.

- 2026-10-15 01:54 KST — Auxiliary24: launch smoke Auxiliary_r10_lead7_seed42 GPU5 PID377619

- 2026-10-15 01:55 KST — Auxiliary24: smoke start Auxiliary_r10_lead7_seed42 GPU5 PID377619 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:55 KST — Auxiliary24: Auxiliary24 manager interrupted/failed; evidence preserved

- 2026-10-15 01:55 KST — Auxiliary24: GPU smoke에서 cuDNN RNN eval-mode backward 제한 확인. 본학습0/24. 진단만 train-mode forward 후 원래 모든 mode/state/RNG/.grad 복원하도록 수정; 모델/손실/가중치 불변. 실패 로그·protocol snapshot 보존 후 동일 smoke 재시도.

- 2026-10-15 01:55 KST — Auxiliary24: launch smoke Auxiliary_r10_lead7_seed42 GPU5 PID379344

- 2026-10-15 01:56 KST — Auxiliary24: tmux detached queue aux24_gpu567, Ceph socket=/ceph_data/yeonwoo/aux24_20261015.sock, controller PID379319. GPU5/6/7 최대2개/장; smoke PASS 후 자동24개. Test 실행 없음.

- 2026-10-15 01:56 KST — Auxiliary24: smoke start Auxiliary_r10_lead7_seed42 GPU5 PID379344 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:56 KST — Auxiliary24: smoke complete Auxiliary_r10_lead7_seed42

- 2026-10-15 01:57 KST — Auxiliary24: launch train Auxiliary_r10_lead7_seed42 GPU5 PID380467

- 2026-10-15 01:57 KST — Auxiliary24: launch train Auxiliary_r10_lead7_seed7 GPU6 PID380468

- 2026-10-15 01:57 KST — Auxiliary24: launch train Auxiliary_r10_lead7_seed123 GPU7 PID380469

- 2026-10-15 01:57 KST — Auxiliary24: launch train Auxiliary_r10_lead14_seed42 GPU5 PID380814

- 2026-10-15 01:57 KST — Auxiliary24: launch train Auxiliary_r10_lead14_seed7 GPU6 PID380815

- 2026-10-15 01:57 KST — Auxiliary24: launch train Auxiliary_r10_lead14_seed123 GPU7 PID380816

- 2026-10-15 01:57 KST — Auxiliary24: train start Auxiliary_r10_lead7_seed123 GPU7 PID380469 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:57 KST — Auxiliary24: train start Auxiliary_r10_lead7_seed7 GPU6 PID380468 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:57 KST — Auxiliary24: train start Auxiliary_r10_lead7_seed42 GPU5 PID380467 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:57 KST — Auxiliary24: train start Auxiliary_r10_lead14_seed7 GPU6 PID380815 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:57 KST — Auxiliary24: train start Auxiliary_r10_lead14_seed123 GPU7 PID380816 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 01:58 KST — Auxiliary24: train start Auxiliary_r10_lead14_seed42 GPU5 PID380814 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 02:30 KST — Auxiliary24: train complete Auxiliary_r10_lead7_seed42

- 2026-10-15 02:30 KST — Auxiliary24: train complete Auxiliary_r10_lead7_seed123

- 2026-10-15 02:30 KST — Auxiliary24: launch train Auxiliary_r25_lead7_seed42 GPU5 PID408103

- 2026-10-15 02:30 KST — Auxiliary24: launch train Auxiliary_r25_lead7_seed7 GPU7 PID408285

- 2026-10-15 02:30 KST — Auxiliary24: train complete Auxiliary_r10_lead14_seed123

- 2026-10-15 02:30 KST — Auxiliary24: launch train Auxiliary_r25_lead7_seed123 GPU7 PID408488

- 2026-10-15 02:30 KST — Auxiliary24: train complete Auxiliary_r10_lead7_seed7

- 2026-10-15 02:30 KST — Auxiliary24: train complete Auxiliary_r10_lead14_seed42

- 2026-10-15 02:30 KST — Auxiliary24: launch train Auxiliary_r25_lead14_seed42 GPU6 PID408763

- 2026-10-15 02:31 KST — Auxiliary24: train start Auxiliary_r25_lead7_seed42 GPU5 PID408103 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 02:31 KST — Auxiliary24: launch train Auxiliary_r25_lead14_seed7 GPU5 PID408910

- 2026-10-15 02:31 KST — Auxiliary24: train start Auxiliary_r25_lead7_seed7 GPU7 PID408285 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 02:31 KST — Auxiliary24: train complete Auxiliary_r10_lead14_seed7

- 2026-10-15 02:31 KST — Auxiliary24: launch train Auxiliary_r25_lead14_seed123 GPU6 PID409075

- 2026-10-15 02:31 KST — Auxiliary24: train start Auxiliary_r25_lead7_seed123 GPU7 PID408488 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 02:31 KST — Auxiliary24: train start Auxiliary_r25_lead14_seed42 GPU6 PID408763 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 02:31 KST — Auxiliary24: train start Auxiliary_r25_lead14_seed7 GPU5 PID408910 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 02:31 KST — Auxiliary24: train start Auxiliary_r25_lead14_seed123 GPU6 PID409075 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 03:16 KST — Auxiliary24: train complete Auxiliary_r25_lead7_seed42

- 2026-10-15 03:16 KST — Auxiliary24: launch train Auxiliary_r50_lead7_seed42 GPU5 PID442172

- 2026-10-15 03:16 KST — Auxiliary24: train complete Auxiliary_r25_lead14_seed42

- 2026-10-15 03:17 KST — Auxiliary24: train start Auxiliary_r50_lead7_seed42 GPU5 PID442172 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 03:17 KST — Auxiliary24: launch train Auxiliary_r50_lead7_seed7 GPU6 PID442559

- 2026-10-15 03:17 KST — Auxiliary24: train complete Auxiliary_r25_lead14_seed7

- 2026-10-15 03:17 KST — Auxiliary24: launch train Auxiliary_r50_lead7_seed123 GPU5 PID442686

- 2026-10-15 03:17 KST — Auxiliary24: train complete Auxiliary_r25_lead7_seed7

- 2026-10-15 03:17 KST — Auxiliary24: train complete Auxiliary_r25_lead14_seed123

- 2026-10-15 03:17 KST — Auxiliary24: launch train Auxiliary_r50_lead14_seed42 GPU6 PID442883

- 2026-10-15 03:17 KST — Auxiliary24: launch train Auxiliary_r50_lead14_seed7 GPU7 PID442884

- 2026-10-15 03:17 KST — Auxiliary24: train complete Auxiliary_r25_lead7_seed123

- 2026-10-15 03:17 KST — Auxiliary24: train start Auxiliary_r50_lead7_seed7 GPU6 PID442559 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 03:17 KST — Auxiliary24: launch train Auxiliary_r50_lead14_seed123 GPU7 PID443159

- 2026-10-15 03:18 KST — Auxiliary24: train start Auxiliary_r50_lead7_seed123 GPU5 PID442686 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 03:18 KST — Auxiliary24: train start Auxiliary_r50_lead14_seed42 GPU6 PID442883 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 03:18 KST — Auxiliary24: train start Auxiliary_r50_lead14_seed7 GPU7 PID442884 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 03:18 KST — Auxiliary24: train start Auxiliary_r50_lead14_seed123 GPU7 PID443159 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 04:16 KST — Auxiliary24: train complete Auxiliary_r50_lead7_seed42

- 2026-10-15 04:16 KST — Auxiliary24: launch train Auxiliary_r100_lead7_seed42 GPU5 PID470399

- 2026-10-15 04:17 KST — Auxiliary24: train start Auxiliary_r100_lead7_seed42 GPU5 PID470399 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 04:17 KST — Auxiliary24: train complete Auxiliary_r50_lead7_seed123

- 2026-10-15 04:17 KST — Auxiliary24: launch train Auxiliary_r100_lead7_seed7 GPU5 PID470816

- 2026-10-15 04:18 KST — Auxiliary24: train start Auxiliary_r100_lead7_seed7 GPU5 PID470816 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 04:20 KST — Auxiliary24: train complete Auxiliary_r50_lead7_seed7

- 2026-10-15 04:20 KST — Auxiliary24: launch train Auxiliary_r100_lead7_seed123 GPU6 PID472080

- 2026-10-15 04:20 KST — Auxiliary24: train complete Auxiliary_r50_lead14_seed42

- 2026-10-15 04:20 KST — Auxiliary24: launch train Auxiliary_r100_lead14_seed42 GPU6 PID472449

- 2026-10-15 04:21 KST — Auxiliary24: train start Auxiliary_r100_lead7_seed123 GPU6 PID472080 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 04:21 KST — Auxiliary24: train complete Auxiliary_r50_lead14_seed7

- 2026-10-15 04:21 KST — Auxiliary24: launch train Auxiliary_r100_lead14_seed7 GPU7 PID472749

- 2026-10-15 04:21 KST — Auxiliary24: train start Auxiliary_r100_lead14_seed42 GPU6 PID472449 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 04:22 KST — Auxiliary24: train start Auxiliary_r100_lead14_seed7 GPU7 PID472749 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 04:22 KST — Auxiliary24: train complete Auxiliary_r50_lead14_seed123

- 2026-10-15 04:22 KST — Auxiliary24: launch train Auxiliary_r100_lead14_seed123 GPU7 PID473469

- 2026-10-15 04:23 KST — Auxiliary24: train start Auxiliary_r100_lead14_seed123 GPU7 PID473469 λaux=.0001 physicsOFF TestOFF

- 2026-10-15 06:02 KST — Auxiliary24: train complete Auxiliary_r100_lead7_seed7

- 2026-10-15 06:02 KST — Auxiliary24: train complete Auxiliary_r100_lead7_seed42

- 2026-10-15 06:04 KST — Auxiliary24: train complete Auxiliary_r100_lead7_seed123

- 2026-10-15 06:06 KST — Auxiliary24: train complete Auxiliary_r100_lead14_seed42

- 2026-10-15 06:09 KST — Auxiliary24: train complete Auxiliary_r100_lead14_seed7

- 2026-10-15 06:10 KST — Auxiliary24: train complete Auxiliary_r100_lead14_seed123

- 2026-10-15 06:10 KST — Auxiliary24: 24/24 Auxiliary-only 80actual epochs 완료, Validation-only 보고서 집계. Test NOT_RUN.

- 2026-10-15 13:42 KST — 최종 논문 감사: outputs/paper_final_verification_20261015에 격리된 감사/추론 코드 생성. 기존 Auxiliary24 Validation-best SHA를 고정하고 GPU 5/6/7 각각 1개씩 Test 추론(미래 forcing 제거/원본/NaN 불변성 포함) 시작. 학습·모델·손실 변경 없음.

- 2026-10-15 13:52 KST — 최종 감사 신규 코드(audit_cpu.py/check_rhs.py/reanalyze.py/provenance.py): 기존 모델 코드 변경 없이 120개 로그·해시, 88개 차분 테스트, 24개 RHS 재계산, 고정 예측 SST/MHW 재검산 및 30/90/180일 블록 bootstrap 구현. 감사 스크립트의 상대 climatology 경로·archived import·독립 테스트 인수·CPU process-title 초기화 문제를 수정; 원본 학습 경로 수정 없음. 현재 120개 프로토콜 및 53개 실행 소스 해시 일치, 88개 수치 테스트 통과. 후속 수치·보고서는 같은 출력 폴더에 보존.

- 2026-10-15 14:01 KST — 최종 논문 감사 완료: outputs/paper_final_verification_20261015에 필수10개 보고서,120개 Test 재검산,24개 Auxiliary 고정 Test 추론,6쌍×8조건의30/90/180일·2000회블록CI 및그림 저장. GPU5/6/7각1개,process name yeonwoo 확인,학습/optimizer update0. 120프로토콜·53소스해시·654보호파일·88수치검사·24RHS재계산PASS. NoAdv가Adv보다RMSE22/24우위,Auxiliary는Baseline대비Val16/24·Test15/24개선. 현120개PASS WITH LIMITATIONS/확인된재학습대상0,구WSDMP1–P3실행mapping UNVERIFIED. 문서렌더러에서선택의존성없이Markdown표생성, P0의K/s잔차후86400변환문구를실제연산순서에맞춤. 기존과학코드/데이터/결과교체없음.


### 2026-10-16 18:36 KST — Oracle120 논문 최종 분석·Figure/Table 생성
- 대상: /ceph_data/yeonwoo/MHW_subset/outputs/paper_analysis_20261016
- 변경: 기존120개 예측·평가/검증자료를 이용한 Table1–6, 별도Validation/seed표, Main1–5 및 S1–S6 (PNG350dpi/PDF/SVG), plot CSV, 대응통계, 주장근거/한계/재현성 보고서 생성.
- 이유: 사용자 첨부 최종분석 프롬프트 이행. 기존 데이터/결과/체크포인트 수정·삭제·이동 없음. 모델 학습·튜닝·추론·WSDM/Oracle120 감사 전체 재실행 없음.
- 검증: 120개 파일 완전성 및 dates/targets/mask/MHWlabel 정렬, 144개 seed대응/48개 조건대응, 기존90일 CI재사용 확인. NoAdv RMSE 22/24, CSI 19/24. 새 SW-Jeju 좌표상자와 observed-field 이류진단 범위 명시. output_validation.json 참조.

- 2026-10-16 18:37 KST — paper_analysis_20261016 그림 최종 점검: S1은 24개 실행이 공통으로 존재하는 epoch1/5/10/20/80만 중앙값/IQR 추세로 표시하고 불규칙 기록은 CSV에 보존. S5는 반복 위도 축 라벨을 줄여 colorbar 겹침 개선. 저장 plot CSV에서 재렌더링; 원본 예측·평가 변경 없음. 통합 캡션/README 및 manifest·코드해시 갱신.

- 2026-10-16 18:39 KST — 최종 해석 보강 및 tableS_noadv_adv 추가: SST/CSI 모두 NoAdv 우위를 양수로 통일, 기존 CSI CI 부호 반전/끝점 재정렬 검산. 7개 연구질문에 실제 수치와 비단조성/작은효과/계절 한계 명시. 기존 산출물 변경 없음; 새 분석 디렉터리 내 보고서·manifest 갱신.

- 2026-10-16 18:39 KST — 영역 선택 기술을 정확히 수정: SW-Jeju 좌표상자는 해당 새 영역의 공간오차 집계 전에 지정했음을 명시(전체 Test 지표를 전혀 보지 않았다는 의미가 아님). 최종 manifest의 11개 Figure/8개 Table 및 대응 데이터 경로 존재 확인.

- 2026-10-16 18:56 KST — subset 최소 보존 목록 확정: 최종120개 Validation-best 전부 보존 확인. 완료된 last/post-warmup 216개를 후보에 추가, 총1562개 47.236GiB 제안. 보존 scientific파일 존재/크기 검사 통과. 코드·데이터 삭제/이동 없음, 구체 목록 승인 대기.
