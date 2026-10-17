# Auxiliary-only 정의 감사

이전 정의는 명확하다. `scripts/train_aux_selection.py:5-7`은 `scripts/train_v3_ablation.py:16-34`의 setup을 사용한다. 과거 Validation 선택 결과 `outputs/aux_weight_selection_20261009/selection.json`의 선택 가중치는 **0.0001**이며 재탐색하지 않았다. 과거 초기 후보10을 재사용하지 않는다.

실제 목적함수:

`L = MSE(pred_norm, target_norm) + 0.0001 * mean_ocean[(86400 * Rres)^2] / S_RHS`

`Rres = 1e-4 * tanh(rres_head(shared_grid_features))`, `S_RHS`는 기존 v3 subset manifest의 고정 `mean(RHS^2)` 단위 (K/day)^2. 이는 RMS 자체가 아니라 제곱 평균이다. `Rres`는 K/s scale을 갖는 head 출력이며 day 변환 후 제곱한다. Rres target은 전 격자0이고 별도 관측/물리 target은 없다.

- Physics residual weight=0, 계산 경로 생략. `zero_rres_physics=True`; Rres는 SST 출력이나 물리 잔차에 더해지지 않는다.
- lambda_we_tie=0; 다른 보조항 없음. Auxiliary warmup 없음: 매 batch 동일 λaux. 이전 physics warmup 설정은 λphysics=0에서 비활성이다.
- 동일 native HiLAM predictor + we_head + rres_head. shared 14,887,681, we_head 66,049, rres_head 66,049, 전체15,019,779개 parameter. Oracle96 Baseline에도 같은 heads가 있으므로 추가 parameter 수0.
- Auxiliary에서 rres_head가 활성화되고 shared backbone에 gradient가 전달된다. SST output_head와 we_head에는 auxiliary gradient가 없다. SST output_head는 data loss로 학습된다.

코드 근거:
- `src/train_jeju_hilam.py:264-299`: 활성 손실만 구성, λphysics=0이면 물리손실 함수 호출 생략.
- `src/train_hybrid_jeju_forecast.py:1081-1117`: Rres penalty와 day 변환.
- `src/train_hybrid_jeju_forecast.py:803-809`: 고정 RHS scale 경로는 예측 수온과 무관한 detached 분모를 반환.
- `src/mhw/models/jeju_hilam.py:219-234,326-345`: SST와 Rres의 별도 출력 head 및 tanh scale.

퇴화해의 수학적 근거: Rres 마지막 linear layer의 W=0,b=0이면 shared representation과 SST 예측에 관계없이 Rres=0, Laux=0, ∇shared Laux=0이다. 이 조작은 감사용 메모리 모델에서만 수행하고 복구했다. 기존 source/checkpoint에는 적용하지 않았다.

이는 초기 gradient가 없다는 뜻이나, 과거 실험의 모든 정규화 효과가 없다는 증거는 아니다. 학습 초기에 shared representation을 바꿀 수 있으나, SST 학습을 개선하거나 보조 target을 학습하지 않고 head만0으로 보내 손실을 완전히 제거할 수 있다는 구조적 한계다. 실제 학습이 반드시 이 경로로 수렴했다고 단정하지 않는다.
