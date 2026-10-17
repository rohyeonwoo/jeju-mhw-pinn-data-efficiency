# Subset별 전처리 데이터셋 (2026-10-08)

사용자 요청에 따라 기존 Full 통계 공유 프로토콜과 별도로 subset-fit 경로를 추가했다.
학습은 실행하지 않는다. 기존 Full/48회 결과와 v2 블록 선택 JSON은 보존한다.

## 통계 정의

- 기존 seed 42/7/123, lead 7/14, history 30, 비율 10/25/50/100을 사용한다 (24개 manifest).
- Input: 선택된 sample의 30일 history 날짜 합집합에서 변수별 ocean-cell mean/std.
  중첩 history의 같은 날짜를 한 번만 센다. population std(ddof=0)를 사용한다.
- Target: 선택된 sample의 실제 target 날짜에서 SST mean/std.
- Physics: 기존 `compute_fixed_rhs_scale`를 선택된 sample에만 적용한다.
  observed endpoint 선형 경로, we=0, P0 RHS의 masked pooled mean-square (K/day)^2.
- Validation/test sample indices는 기존 `make_time_splits`와 동일하다.
  세 split 모두 해당 subset의 동일한 input/target 통계로 transform한다.
- Baseline/PINN은 같은 manifest를 사용한다. PINN은 manifest의 고정 RHS scale도 적용한다.
- 지리 좌표와 mask는 고정 공간 정보다. residual feature map은 사용하지 않는다.
- NetCDF의 기존 anomaly 입력 변수는 유지한다. 이 작업은 anomaly 생성 climatology까지
  subset으로 재추정하지 않으므로 전체 upstream preprocessing의 subset-only 보증은 아니다.

## 산출물과 사용

`data/subsets/preprocessing_subset_v1/index.json`에 24개 데이터셋과 검사 결과를 기록한다.
각 `ratio_{ratio}_lead{lead}_seed{seed}.json`은 원본 NetCDF SHA-256, 원본 subset SHA-256,
정확한 train/val/test 시작 인덱스, 통계 fitting 날짜 인덱스, 통계, RHS scale을 포함한다.
시간축을 잘라 이어붙이거나 원본 NetCDF를 24번 복제하지 않는다. manifest와 원본 NetCDF를
함께 읽으면 모델에 입력되는 정규화 dataset을 재현한다. 반환되는 train은 이미 subset이다.

```python
from mhw.subset_preprocessing import load_subset_bundle, apply_manifest_physics
bundle, metadata = load_subset_bundle(
    "data/subsets/preprocessing_subset_v1/ratio_10_lead7_seed42.json"
)
# bundle.train / bundle.val / bundle.test: 같은 subset 통계를 적용한 Dataset
# baseline과 PINN에 같은 bundle.stats['target']을 전달한다.
# PINN 실행기에서는 apply_manifest_physics(args, physics_args, metadata)를 호출한다.
bundle.ds.close()
```

`src/train_jeju_hilam.py`에 `--subset-manifest PATH`를 추가했다. 이 옵션은 Full 통계
계산 경로를 우회하고 subset train indices, subset stats 및 RHS scale을 사용한다.
history/lead 불일치, Full RHS scale override, P0 이외 budget은 오류 처리한다.
원본 source의 기본 Full 경로는 과거 재현성을 위해 유지했다. 새 scarcity 실험에는
반드시 manifest 경로를 사용한다. 기존 `_queue/*snapshot.py`는 과거 실행 기록이며
새 manifest를 자동으로 사용하지 않는다. 기존 fixed-step 48회 비교의 스케줄은 이 작업에서
변경하거나 재실행하지 않았다. 향후 실행기를 연결할 때 Full 기준 S와 subset train 길이를
혼동하지 않도록 스케줄도 명시해야 한다.

재구축 전용 명령 (학습 없음):

```bash
python -B scripts/build_subset_preprocessing.py
```

기존 출력 폴더를 덮어쓰지 않는다. 재실행은 별도 `--output-dir`를 지정한다.
생성이 완전히 끝나고 검증을 통과해야 `index.json`이 기록된다.

## 코드와 검증

- `src/mhw/subset_preprocessing.py`: subset-only fitting, 저장 통계 적용, raw RHS loader.
- `scripts/build_subset_preprocessing.py`: 24개 생성, 원본 split 보존과 dataset 적용 검증.
- `tests/test_subset_preprocessing.py`: 선택되지 않은 관측을 바꿔도 stats/RHS가 변하지 않는지,
  실제 fitting 날짜와 population statistics, 잘못된 인덱스 및 Full scale override 검사.
- 각 manifest는 train/val/test 시작·끝 sample의 모든 입력 채널 transform 및 target 역변환을 검사한다.
- 경량 physics loader의 raw tensor와 RHS scale을 기존 dataset/계산 함수와 대조한다.

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

기존 `compute_train_stats(..., train_indices=subset)`는 prefix slice를 사용하므로
새 subset 경로에서 호출하지 않는다. 새 함수 `fit_subset_stats`가 명시적인 날짜 집합으로
이 문제를 해결하며 Full 통계 계산 함수의 과거 동작은 보존한다.
