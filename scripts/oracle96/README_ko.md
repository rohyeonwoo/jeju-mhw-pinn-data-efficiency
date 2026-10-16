# Training-only Oracle Physics Supervision — 96조건

사용자 2026-10-11 실행 지시 적용. 실제 연구 기록은 `/ceph_data/yeonwoo/MHW_subset/docs/research_log.md`이다. `/home/yeonwoo/mhw_subset`은 현재 존재하지 않는다.

- GPU 7/8/9만 사용, GPU당 1 worker, CUDA 초기화 전 process title `yeonwoo`, PID/UUID 확인. 다른 사용자의 작업을 중단하지 않는다.
- 기존 src, raw/v1/v2/v3, v7 코드·결과·중단 상태를 보존한다. v7 features를 해시 검증 후 읽기 전용 재사용하며 전처리는 재생성하지 않는다.
- 10/25/50/100 × Lead7/14 × seeds42/7/123 × Baseline/P0/NoAdv/Adv =96. 내부 라벨은 B0/B1/A/B이며 표의 model_label로 표시한다. 기존 실행은 재사용하지 않는다.
- Train 경계는 기존 chronological split의 첫 Validation target 전날. 각 Train 샘플의 target 및 t..t+lead-1 forcing이 그 안에 있는지 사전검사와 getitem에서 강제한다. split 경계에서 history가 겹치는 기존 sliding-window 설계는 유지한다.
- Train만 oracle forcing loss를 사용한다. Validation/Test loader는 forcing_mode=persist의 과거/current 원본 view에서 prediction에 필요한 필드만 반환한다. Validation selection에서 physics를 계산하지 않는다. 모델 업데이트가 없고 미래 forcing이 없어도 예측 가능하다.
- 학습된 best checkpoint마다 full Validation에서 forcing 없음/실제 oracle 추가/NaN 교란 3가지의 예측·SST/MHW 지표·모델 상태 동일성을 검사한다. Test는96학습 완료 후 같은 검사와 평가를 수행한다. 평가 기준은 기존 MHW 첫70% raw 날짜·DOY±5·90%·5일 지속을 유지한다.
- 네 모델은 동일한 native HiLAM 구조(we/rres heads 포함), 초기화, subset, shuffle, AdamW betas(.9,.95), weight decay.1, batch16, fp32, native LR warmup5/cosine, clip norm1. 동일80×478=38240 updates, 조기 종료 없음. 매478updates full Validation SST RMSE로 best 선택. 과거 patience12 실행과 달라 기존 Baseline/P0는 재사용하지 않는다.
- λphys=.1, warmup0, λrres=λtie=0, free residual 없음, adaptive 없음. Kd=.06 m^-1, rho1025, Cp3990, h_min5 및 실제 물리식은 v7 검증 식 그대로 유지한다. 공통 v3 subset-fitted mean-square RHS scale을 사용한다. residual K/day를 ±10 clip 후 sqrt(scale)로 나눈 MSE.
- NoAdv와 Adv는 수평 이류 항만 다르다. P0는 학습 we와 고정1K contrast이고 개선식은 backward MLD 성장 proxy·GLORYS Tb를 사용하므로 다항 차이이다.
- 전체 순단파에 단일지수를 적용한 근사이며 Kd는 제주 실측 최적값이 아니다. NOAA SST는 GLORYS 층평균의 대리변수(Validation 차이 RMSE≈1.185K). 기존 CPU 잔차 부호 반전은 가열 과보정 가능성을 뜻하나 원인을 확정하지 않는다.
- SST-label efficiency를 평가한다. PINN의 추가 ERA5/GLORYS 감독을 포함한 total-information efficiency는 입증하지 않는다. 필요시 history-only 물리손실은 후속 ablation으로 제안하며 이번에 실행하지 않는다.
- 사후 물리 residual은 saved predictions에 별도로 적용하며 raw/cumulative signed residual과 term mean/RMS를 기록한다. CPU 열수지 적합성과 GPU 예측성능을 구분한다. Test 반복 확인 이력으로 탐색적 결과이며 Test에 맞춰 상수/λ를 수정하지 않는다.

진행: CPU → smoke4 → 대표4 → 10%Lead7의 나머지8 → 나머지84 → 평가192 model-splits → 한국어 보고. 각 gate는 유한 loss/gradient, checkpoint roundtrip, 동일 data/setup/updates, forcing 불변성을 요구한다. 실패/폭주/누출/데이터 불일치 시 전체 관리자 중단, 자동 재시도 없음. 성능이 가설에 불리하다는 이유로 중단하거나 숨기지 않는다.

CPU 준비: `CUDA_VISIBLE_DEVICES='' /ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B scripts/oracle96/prepare.py`

실행은 프로젝트 Ceph 디렉터리에서 전용 Ceph tmux socket으로 `scripts/oracle96/run_all.py`를 시작한다. `outputs/v8_training_only_oracle96_20261011/training_status.json`, `logs/`, 각 run의 history/status/config/best.pt/last.pt 및 forcing_invariance 파일에 실제 상태가 남는다. 기존 관리자는 재개하지 않는다.
