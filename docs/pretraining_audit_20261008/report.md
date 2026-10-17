# 학습 전 최종 검증 — 2026-10-08 14:03 KST

**전체 판정: FAIL / 본 학습 보류.** subset별 normalization과 P0 RHS scale 적용은 통과했지만, 기존 thetao anomaly는 선택되지 않은 training 관측으로 만든 공통 climatology를 사용한다. sst_anom은 원본 생성 코드까지 확인되지 않았다.

본 학습·optimizer·backward·GPU 실행은 하지 않았다. 실제 main의 setup 문장을 config/optimizer 생성 전까지만 실행했으며, 모델 생성은 CPU zero-output probe로 대체했다. 24개 manifest × Baseline/PINN 2개 분기를 검사하고 실제 run_epoch를 optimizer=None으로 한 batch씩 총48회 평가했다. 원본 모델/학습/전처리 소스와 기존 manifest는 수정하지 않았다.

| 항목 | 판정 | 근거 |
|---|---|---|
| 1. anomaly 생성 방식/기간 확인 | FAIL (일부 미확인) | thetao 6개는 생성 코드와 수치 재현 PASS. sst는 메타데이터와 상위 파일 동일성까지만 확인 |
| 2. subset-only climatology | FAIL | thetao는 1993-01-01~2013-12-31의 공통7,670일; 선택되지 않은 training 관측 포함 |
| 3. subset input/target/RHS 적용 | PASS | 24/24 통계·RHS 재계산 일치, 실제 setup 및 loss 경로 검사 |
| 4. Baseline/PINN 동일 데이터/통계 | PASS | 같은 manifest의 indices/stats 및 입력·타깃·forcing tensor 일치 |
| 5. Full 통계 덮어쓰기 방지 | PASS (manifest 경로) | Full builder/RHS 재계산을 오류 함수로 막아도48개 setup 통과; 충돌 Full scale 주입 거부 |
| 6. Nested subset/원래 split 유지 | PASS | 24/24 원본 v2 인덱스 일치;6개 lead×seed nested chain;lead간 동일 starts;val/test 원래 split 일치 |

## 1–2. Anomaly provenance와 정보 범위

복구한 생성 코드: [/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py:95](/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py:95), [140–159행](/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py:140).

- `ntrain = int(round(nt * args.train_ratio))`, default `train_ratio=0.7`.
- 실제 nt=10,957, ntrain=7,670, 기간 **1993-01-01~2013-12-31**.
- 깊이별/격자별로 pandas `dayofyear`가 같은 날짜의 raw thetao 평균을 계산한다. 별도의 leap-day 정렬은 하지 않는다.
- 빈 DOY는 ±3 이내 유효 climatology, 이후 전체 reference 평균으로 대체한다.
- `anom = raw - climatology[doy]`를 전체 기간에 적용한다.
- 이 기준은 dataset의 선행70%이며 lead별 `make_time_splits` 또는 subset start indices로 정해지지 않는다.
- 원본 3D NetCDF의6개 깊이에서 위 코드 블록을 그대로 재실행했다. **6/6 변수 전체 배열이 저장 anomaly와 정확히 같고 최대 오차0**이다. 상세: [anomaly_reconstruction.json](anomaly_reconstruction.json).
- lead7 validation target 시작2014-01-11, lead14 시작2014-01-13이므로 thetao climatology에 val/test target 날짜는0개다. 초기 validation 입력 이력은 train 기간과 겹친다. 'validation과 날짜가 전혀 겹치지 않는다'는 의미는 아니다.

대표 lead14/seed42에서 climatology의7,670일 중 선택된44일 sample footprint 밖의 날짜 수:

| 비율 | 선택 window 밖 climatology 날짜 |
|---|---:|
| 10% | 6,950 |
| 25% | 5,780 |
| 50% | 3,890 |
| 100% | 20 |

`sst_anom`:

- [netcdf_metadata.json:9](netcdf_metadata.json:9)의 global comment는 `1971-2000 OI.v2 SST`를 기준으로 명시한다. 변수 자체도 NOAA anomaly로 표기되어 있다(21–31행).
- 현재 파일의 sst_anom은 상위 glorys3d 파일의 전체 배열과 정확히 같다. 전처리 코드는 원래 변수를 보존한다([161–164행](/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py:161)).
- 실제 NOAA anomaly 생성·보간 및 최초 merge 코드는 조사한 소스에서 찾지 못했다. 따라서 정확한 산출 알고리즘과 기준 기간을 코드로 완전히 입증한 것으로 처리하지 않는다.
- 메타데이터대로라면 이 기준은 subset별 fit이 아니고1971–2000의 외부 역사 정보를 사용한다. 기간상 현 val/test target은 포함하지 않지만, 이것을 원본 코드 확인을 대신하는 무누출 보증으로 표현하지 않는다.

현재 anomaly가 그대로 입력되는 근거: [src/mhw/jeju_hilam_dataset.py:16–36](../../src/mhw/jeju_hilam_dataset.py:16), [src/mhw/subset_preprocessing.py:67–71](../../src/mhw/subset_preprocessing.py:67). 새 함수는 기존 anomaly에 mean/std를 fit할 뿐 climatology를 재작성하지 않는다.

## 3–5. 실행 경로

[train_jeju_hilam.py:606–620](../../src/train_jeju_hilam.py:606): manifest가 있으면 `load_subset_bundle` 및 `apply_manifest_physics`를 사용하고 Full builder는 else 분기다.

[subset_preprocessing.py:135–178](../../src/mhw/subset_preprocessing.py:135): manifest의 train start indices와 원래 val/test indices를 검사한 뒤 같은 stats dictionary를 세 Dataset에 적용한다. graph wrapper는 여기서 target mean/std를 새로 저장한다.

[datasets.py:260–278](../../src/mhw/datasets.py:260): input `(x-mean)/std` 적용. [295–296행](../../src/mhw/datasets.py:295): target scaling. [train_jeju_hilam.py:311–313](../../src/train_jeju_hilam.py:311): normalized target MSE 및 동일 target stats로 raw prediction 역변환.

[subset_preprocessing.py:184–196](../../src/mhw/subset_preprocessing.py:184): manifest RHS scale을 args/physics_args에 같이 설정하며 다른 CLI scale은 거부한다. [train_jeju_hilam.py:641–655](../../src/train_jeju_hilam.py:641): target stats를 bundle에서 가져오고, RHS scale이 None일 때만 재계산한다. 정상 manifest 분기에서는 이 조건이 거짓이다.

[train_hybrid_jeju_forecast.py:394–407](../../src/train_hybrid_jeju_forecast.py:394): fixed scale 우선. [447–483행](../../src/train_hybrid_jeju_forecast.py:447): clip 이후 sqrt(scale)로 나누고 제곱. [1118–1131행](../../src/train_hybrid_jeju_forecast.py:1118): auxiliary residual penalty에 같은 scale 적용.

체크포인트 복원은 [train_jeju_hilam.py:788–793](../../src/train_jeju_hilam.py:788)에서 model_state만 읽으며 Full stats로 덮어쓰지 않는다. 평가용 ACC climatology는 [490–509행](../../src/train_jeju_hilam.py:490)에서 별도로 계산하지만 학습 input/target/RHS 통계를 바꾸지 않는다.

PASS 범위는 `--subset-manifest`와 검증한 기본 P0 설정이다. 옵션을 생략하거나 기존 `_queue` snapshot을 쓰면 legacy Full 경로가 남아 있다. 임의의 추가 normalization 옵션으로 바꾼 실행까지 검증한 것은 아니다.

실행 검증: [audit.py:73–164](audit.py:73), 결과: [runtime_verification.json](runtime_verification.json). raw 데이터는 캐시했으며 manifest별 통계·인덱스는 실제 코드로 각각 구성했다. 원본 disk loader도 별도로 검사했다.

## 6. 인덱스

[subset_preprocessing.py:140–148](../../src/mhw/subset_preprocessing.py:140)의 검사와 원본v2 JSON 대조를 모두 통과했다. 각 seed/lead에 대해 10% ⊂25% ⊂50% ⊂100%이고, 같은 seed/비율의 lead7/14 start indices가 같다.

샘플 수는376/987/1,974/3,995. val/test는 lead7 1,638/1,639, lead14 1,637/1,638로 보존된다.100% block subset은 Full train과 다르다.

## 수정 방안 (이번 감사에서 적용하지 않음)

1. 각 subset·seed의 unique input-history 날짜만으로 SST 및6개 raw thetao 깊이별 climatology를 fit한다. 원본 raw thetao가 들어 있는3D NetCDF가 존재함을 확인했다.
2. 저장한 subset climatology를 train/val/test 모두에 적용한다. DOY/leap-day 및 빈 계절일 fallback을 명시하고 fallback도 선택된 관측만 사용한다.
3. 재작성한 anomaly를 대상으로 input mean/std를 다시 fit하고, loader가 기존 anomaly 대신 이를 사용하도록 연결한 새 버전의24개 manifest를 만든다. 단순히 현재 anomaly를 다시 z-score하는 것으로 해결되지 않는다.
4. climatology 자체·원본수온 파일 해시·정확한 reference 인덱스를 저장하고, 선택되지 않은 raw 관측을 바꿔도 선택된 sample의 feature가 바뀌지 않는 통합 검증을 추가한다.
5. 원본 NOAA climatology를 유지하려면 외부 고정 기후 사전정보를 허용하는 실험으로 정의해야 한다. 엄밀한 subset-only 조건과 같은 실험으로 표시하지 않는다.

## 추가 실행 전 규칙 점검

직접 `train_jeju_hilam.py`를 실행하는 경로에는 `yeonwoo` process title 설정이 없다. [591–594행](../../src/train_jeju_hilam.py:591)에서 seed/device 설정에 들어가며, `set_seed`는 [100–105행](../../src/train_jeju_hilam.py:100)에서 CUDA를 조회/설정한다. 따라서 사용자 GPU process-name 규칙은 이 직접 경로가 현재 보장하지 않는다. CUDA 접근 전에 강제 title 설정 및 실제 GPU PID 표시 검증을 추가해야 한다.

또한 [53행](../../src/train_jeju_hilam.py:53)의 output-root 기본값은 `D:/MHW/...`이며 Ceph 내부 경로를 강제하지 않는다. Ceph 절대경로 검사와 model/ratio/lead/seed별 출력 분리를 실행기에 추가해야 한다. 이번 감사는 CUDA를 숨기고 모든 출력을 Ceph에 저장했으므로 GPU 실행 규칙 위반은 없었다.
