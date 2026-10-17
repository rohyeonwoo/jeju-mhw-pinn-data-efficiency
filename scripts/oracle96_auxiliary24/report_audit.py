"""Document a blocked preflight; never fabricate training or Validation results."""
import json,csv,hashlib,fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/v8_actual_epoch_auxiliary24_20261015';T=R/'outputs/v8_actual_epoch_oracle96_20261011';A=R/'outputs/aux_weight_selection_20261009'
def main():
 g=json.loads((O/'gradient_audit.json').read_text());assert g['status']=='BLOCKED_TRIVIAL_ZERO_SOLUTION'
 now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
 definition='''# Auxiliary-only 정의 감사

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
'''
 (O/'auxiliary_definition_audit.md').write_text(definition)
 lines=['# 실제 batch Gradient 감사',f'시각: {now}', '',g['scope'], '', '|Batch|Data loss|Aux loss|공유 Data gradient norm|공유 Aux gradient norm|가중 Aux/Data 비|Rres head gradient norm|cosine|','|---|---|---|---|---|---|---|---|']
 for b in g['initial_batches']:lines.append('|'+ '|'.join(str(b[k]) if k=='batch' else f'{b[k]:.8g}' for k in ['batch','data_loss','auxiliary_loss','shared_data_grad','shared_aux_grad','weighted_ratio','rres_head_grad','cosine'])+'|')
 lines+=['',f'Physics 함수 호출: {g["physics_calls"]}, Auxiliary 호출: {g["auxiliary_calls"]} (초기3 + 반사실3). 초기 gradient는 finite/nonzero이며 **초기 신호 존재 검사 PASS**.','', '|반사실 Batch|Aux loss|공유 Aux gradient|Rres head gradient|SST 예측 동일|Data loss 동일|','|---|---|---|---|---|---|']
 for b in g['zero_head_batches']:lines.append(f'|{b["batch"]}|{b["auxiliary_loss"]}|{b["shared_aux_grad"]}|{b["rres_head_grad"]}|{b["SST_prediction_bitwise_unchanged"]}|{b["data_loss_unchanged"]}|')
 lines+=['',f'마지막 Rres 층의 {g["zeroed_parameter_count"]}개 weight/bias만0으로 하면 정확한 zero-loss 해가 존재한다. **퇴화해 배제 검사 FAIL**. optimizer step은0회. 검사 후 원래 모델 전체 state hash 복구 확인.',f'GPU5 프로세스 이름 yeonwoo 확인; 최대 reserved {g["peak_reserved_mib"]:.1f} MiB.']
 (O/'gradient_audit.md').write_text('\n'.join(lines)+'\n')
 protocol=f'''# Auxiliary-only 24개 계획 — 실행 보류

작성: {now}

기존 Oracle96 원본: `{T}`. 신규 경로: `{O}`.
10/25/50/100% × lead7/14 × seed42/7/123. 기존 96개 재학습 없음. 계획 조합과 Baseline 대응 경로는 run_manifest.csv.

유지할 프로토콜: 80 actual epochs, epoch당 각 선택 Train window 정확히1회, batch16, 기존 v3 manifest/통계/climatology, AdamW betas(.9,.95)/weight decay0.1, gradient clipping norm1, peakLR0.0002/initialLR0.00002, LR warmup5actual epochs 후 native cosine, 매 epoch 전체 Validation SST RMSE, 전체80epoch 중 최저 checkpoint, early stopping OFF. Physics weight0/warmup 비활성, Aux weight0.0001/warmup 없음. Test 자동 실행 금지. 새 hyperparameter tuning 없음.

GPU: 첨부 마지막 지시의 **5·6·7**을 적용(본문7·8·9 대체), GPU당 우리 학습 작업 최대2. 다른 사용자 프로세스를 종료하지 않는다. 감사는GPU5 1개만 사용했고 본 학습 큐는 시작하지 않았다.

차단 조건: 기존 Auxiliary 정의에 predictor와 무관하게 head만0으로 보내는 정확한 퇴화해가 확인됨. 사용자가 요구한 사전검사 조건에 따라 자동24개 확장을 수행하지 않는다. 손실을 임의로 변경하지 않았다.
'''
 (O/'protocol.md').write_text(protocol)
 checks=[('Auxiliary 정의/가중치','PASS','기존 선택 λaux=0.0001, fixed RHS-normalized zero-target Rres'),('Physics 비활성','PASS','실제6호출에서 physics 함수0회; 호출 시 예외를 내도록 검사'),('초기 shared/head gradient','PASS','10% lead7 seed42, 실제 Train3batch finite/nonzero'),('head-only zero 퇴화해 배제','FAIL','마지막257개 parameter만0 → auxloss/sharedgrad0, SST 동일'),('초기 Predictor 동일','PASS (대표1조건)','독립 재구성 Baseline과 state tensors torch.equal; 저장 Oracle initial SHA와 일치'),('v3 통계/분할/입력 fingerprint','PASS (대표1조건)','원 Oracle96 config와 비교; 다른23조건 실제 bundle 검사는 NOT_RUN'),('Actual Epoch sampling','PASS (대표1epoch)','376samples, 24updates, 원 Baseline epoch1 순서SHA 일치'),('80epoch LR schedule','PASS (대표1조건)','원 Baseline history 80개 LR와 대조'),('Early stopping OFF','계획 고정 / 학습 NOT_RUN','Oracle96 원 설정 유지; 정식 루프는 시작하지 않음'),('Checkpoint save/load','PASS (감사 전용)','initial_roundtrip.pt state tensor 비교; 학습 best checkpoint 아님'),('GPU 이름/자원','PASS (감사GPU5)','NVML yeonwoo 및 지정 UUID; 본24개 배정 NOT_RUN'),('보호 입력 무결성','PASS','protected_audit_inputs.json 원문SHA 전후 일치')]
 (O/'preflight.md').write_text('# Preflight: FAIL — 24개 본 학습 보류\n\n|검사|판정|범위/근거|\n|---|---|---|\n'+'\n'.join('|'+ '|'.join(r)+'|' for r in checks)+'\n\n검증한 대표조건을 전체24조건 PASS로 확대하지 않았다. 퇴화해 차단 조건 때문에 나머지 실행 준비/전체 검사는 진행하지 않는다.\n')
 (O/'checkpoint_audit.md').write_text('# Checkpoint audit\n\nPASS (감사 전용): checkpoints/audit_only/initial_roundtrip.pt 저장/재로드 state tensor 일치. 반사실 head 변경을 복원한 초기 모델만 저장했다. optimizer update0. 정식 best/last 학습 checkpoint는 생성하지 않았다. 기존 Oracle96 및 과거 Auxiliary checkpoint는 수정하지 않았다.\n')
 for filename,columns in [('epoch_metrics.csv',['model','ratio','lead','seed','epoch','train_loss','data_loss','auxiliary_loss','lr','validation_rmse','updates']),('validation_summary.csv',['model','ratio','lead','n_seeds','val_rmse_mean','val_rmse_sd']),('seed_improvements.csv',['ratio','lead','seed','baseline_rmse','auxiliary_rmse','improvement_pct'])]:
  with (O/filename).open('w') as f:csv.writer(f).writerow(columns)
 # Preserve exact observations from previous training, without implying the new protocol ran.
 historical=[]
 for seed in [42,7,123]:
  d=A/f'runs/lead7_seed{seed}/auxiliary_only_aux0.0001';h=json.loads((d/'history.json').read_text());s=json.loads((d/'status.json').read_text())
  historical.append(dict(seed=seed,epochs=s['epochs'],first_aux=h[0]['train_rres_l2'],last_aux=h[-1]['train_rres_l2'],source=str(d)))
 (O/'historical_auxiliary_loss.json').write_text(json.dumps(historical,indent=2)+'\n')
 report=f'''# Auxiliary24 최종 사전검사 보고

{now}

**FAIL — 요청된 사전검사 중 zero-head 퇴화해 배제 조건을 통과하지 못해 본 학습0/24, Test 미실행.**

정의는 명확하고 초기 학습 신호도 있다. 이전 Validation에서 선택한 weight0.0001을 그대로 적용했으며, 실제 Train3batch에서 shared predictor와 Rres head의 보조 gradient는 finite/nonzero였다. Physics loss 함수는 호출되지 않았다.

그러나 Rres 마지막 층257개 parameter만0으로 설정하면 SST 예측과 Data loss는 bitwise 그대로이고 보조손실 및 보조 gradient는 모두 정확히0이었다. 이는 새로운 관측 target이나 예측 정확도 개선 없이 보조 목적을 해결할 수 있는 구조다. 첨부의 “보조손실이 상수이거나 의미 없는0으로 수렴하도록만 구성되어 있지는 않은가” 검사를 충족했다고 표시할 수 없어 자동 확대를 중단했다.

초기 gradient가 없거나 상수 loss라는 판정은 아니다. 실제 과거 학습이 항상 zero-head로 수렴했다는 결론도 아니다. 이 보조 정규화가 초기 optimization에 영향을 줄 가능성은 남는다. 따라서 과거 성능 결과를 무효라고 단정하지 않는다. 다만 이를 독립적인 물리 감독이나 유의미한 auxiliary target 학습의 증거로 해석할 수 없다.

현재 Oracle96 Baseline에도 두 head가 이미 포함되어 있으므로 구조 추가 parameter0개이며 비교 대상은 주로 기존 head 정규화 활성화 효과다. 공유 parameter14,887,681, Rres head66,049, we head66,049. 본 학습 설정을 임의로 수정하거나 다른 보조 task를 만들지 않았다.

## 기존 보조손실 감소 (참고만)

|Seed|기존 종료 epoch|첫 epoch Aux loss|마지막 Aux loss|
|---|---|---|---|
'''
 for h in historical:report+=f'|{h["seed"]}|{h["epochs"]}|{h["first_aux"]:.6f}|{h["last_aux"]:.6f}|\n'
 report+='''
위 과거 실험은 478updates/virtual epoch와 early stopping을 사용했으므로 이번80actual epoch 결과로 대체하거나 직접 같은 예산이라고 해석하지 않는다. 손실 감소 자체는 shared representation 개선과 head 축소를 구별하지 못한다.

## 산출물

- auxiliary_definition_audit.md: 정확한 기존 정의/가중치/코드 위치
- gradient_audit.md 및 gradient_audit.json: 실제3batch 초기 및 반사실 수치
- protocol.md, preflight.md: 유지할 설정과 PASS/FAIL/NOT_RUN 범위
- run_manifest.csv: 24개 계획 조합, 모두 BLOCKED_PREFLIGHT
- run_state.json: 전체 본 학습 미시작 상태
- checkpoint_audit.md: 감사용 초기 모델 roundtrip
- epoch_metrics.csv / validation_summary.csv / seed_improvements.csv: 본 학습이 없어 header만 존재
- gpu_identity.json / gpu_resource_log.csv: 감사 자원 및 process name 확인

현재 결정은 요청된 중단 규칙 준수다. 기존 zero-target Rres를 의도적인 출력 축소 정규화로 평가하는 실험과, 의미 있는 보조 예측 task를 학습하는 실험은 연구 목적이 다르다. 후자를 원할 경우 보조 target/연결 구조를 별도로 정해야 하며 이번 작업에서 임의 변경하지 않았다.
'''
 (O/'report_ko.md').write_text(report)
 with (R/'docs/research_log.md').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(f'\n- {now} — Auxiliary24: scripts/oracle96_auxiliary24/report_audit.py 및 요청 산출물 작성. 대표 실제 batch에서 초기 gradient PASS, head-only zero 퇴화해 배제 FAIL; 24개 계획을 BLOCKED_PREFLIGHT로 기록하고 본학습/Test 미실행. 수치/코드근거/NOT_RUN 범위 및 이전 실험의 정규화 효과 가능성을 구분하여 report_ko.md에 명시.\n')
 print('Blocked preflight reports saved')
if __name__=='__main__':main()
