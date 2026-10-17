# 제주 혼합층 개선식 CPU 진단 — 실행 안내

## 파일
`experiment_mixed_layer_budget_cpu.py`: 기존 2D budget 및 3D GLORYS 파일을 읽기만 하는 독립 실행 스크립트입니다. GPU, torch, 체크포인트가 필요하지 않습니다.

## 1. 서버에 파일 배치
다운로드한 파일을 `~/mhw_subset/scripts/experiment_mixed_layer_budget_cpu.py`에 배치합니다. 기존 소스와 체크포인트는 수정하지 않습니다.

## 2. 데이터 경로 확인
다음은 예시이므로 **실제로 존재하는 파일 경로**로 수정해야 합니다.

```bash
cd ~/mhw_subset
find /ceph_data/yeonwoo/MHW_subset -type f -name '*.nc' | grep -E 'budget2d|glorys3d' | head -30
export BUDGET2D='/absolute/path/to/merged_..._budget2d.nc'
export GLORYS3D='/absolute/path/to/merged_..._glorys3d.nc'
```

## 3. 단위·좌표 확인

```bash
python - <<'PY'
import os, xarray as xr
for name, key in [('BUDGET2D','qnet'),('BUDGET2D','ssr'),('GLORYS3D','thetao')]:
    ds = xr.open_dataset(os.environ[name])
    print(name, key, ds[key].dims, ds[key].attrs)
    ds.close()
PY
```

`qnet`은 **W/m²**, `ssr`은 W/m² 또는 하루 누적 J/m²인지 확인합니다. 단위를 확인하지 않고 `--assume-qnet-wm2`나 `--ssr-units`를 사용하지 마세요.

## 4. 1차 실험: 투과열 미포함

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B scripts/experiment_mixed_layer_budget_cpu.py \
  --budget2d "$BUDGET2D" --glorys3d "$GLORYS3D" \
  --output /ceph_data/yeonwoo/MHW_subset/outputs/v6_cpu_mixed_layer_budget_screen_20261011 \
  --split val --history 30 --leads 7,14 --max-windows 24 \
  --kh-values 0,25,100,300 --penetration off
```

각 `Kh` 값에서 수평 이류 포함/제외 × 혼합층 심화 기반 연행 포함/제외를 비교합니다. 전부 CPU의 저장된 관측장만 이용합니다.

## 5. 2차 실험: 단파복사 투과항 포함

`ssr` 단위 검증 후 실행합니다. 메타데이터에 단위가 올바르게 기재돼 있다면 자동 판별합니다.

```bash
python -B scripts/experiment_mixed_layer_budget_cpu.py \
  --budget2d "$BUDGET2D" --glorys3d "$GLORYS3D" \
  --output /ceph_data/yeonwoo/MHW_subset/outputs/v6_cpu_mixed_layer_budget_optical_20261011 \
  --split val --history 30 --leads 7,14 --max-windows 24 \
  --kh-values 0,25,100,300 --penetration on --kd 0.06 --ssr-units auto
```

`Kd=0.06/m`은 **실측치가 아닌 민감도 가정**입니다. 결과가 민감하다면 `--kd 0.04`, `--kd 0.1` 등을 별도 출력 폴더로 비교하세요.

## 6. 분석 결과

- `summary.csv`: 목표별(GLORYS 혼합층 / NOAA SST), Lead별, 항별 조합의 누적 RMSE, bias, zero-change 기준, 이류 포함-제외 paired ΔMSE
- `window_metrics.csv`: 개별 시작일의 격자별 SSE/SAE 등 집계
- `term_diagnostics.csv`: 열플럭스, 수평 이류, 연행, 투과열, 라플라시안의 누적 크기
- `manifest.json`: 평가 설정과 해석 한계

**NOAA 비교는 GLORYS 혼합층 RHS와 NOAA SST 변화의 교차 비교로, 정확한 NOAA SST 보존식이 아닙니다.**

## 해석 제약

1. 기존 P0는 학습된 `w_e`와 NOAA SST 기반 경로를 사용합니다. 이 스크립트의 `entrainment_proxy=True`는 MLD 심화율과 GLORYS 하층 수온을 사용하므로 기존 P0를 재현하지 않습니다.
2. 모델의 예측 성능이 아니라 관측 경로에서 계산한 물리 열수지 적합성입니다.
3. 시간별 미래 GLORYS/MLD는 진단 목적으로 사용한 사후 관측값입니다. 추후 PINN 배포 시에는 과거값이나 별도의 예보 자료만 사용해야 합니다.
4. 수평 확산은 일정한 Kh를 가정한 마스크 기반 근사 차분, Qpen은 단일 지수 감쇠 근사, 연행은 MLD 심화율 proxy입니다.
5. 원래 v4/v5의 3-seed/24조건과 정확히 수치 비교하려면 그 실험에서 사용한 시작 인덱스, 마스크, 물리항 및 `w_e` 설정을 동일하게 맞춰야 합니다. 이 코드는 기본적으로 split에서 균일 샘플링한 시작일만 분석합니다. 동일 시작 인덱스 리스트가 있으면 `--starts-json /path/starts.json`을 사용하세요.
6. 최종 후보 선택은 Validation 기준으로 하고, Test는 고정된 후보의 최종 보고에만 사용합니다.
7. 입력 파일 보호 380개 해시 점검은 기존 진단 스크립트로 실행 전후에 별도 수행하십시오. 이 스크립트는 원본 입력을 읽기 전용으로 열고, 신규 출력 디렉터리 존재 시 실행을 중단합니다.
