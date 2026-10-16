"""Assemble a blocked preflight report and non-executable experiment proposal."""
from pathlib import Path
import json,hashlib,csv,ast
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parents[1];O=R/'outputs/v2_final_preflight_20261008';V=R/'data/subsets/preprocessing_subset_v2'
def read(n):return json.loads((O/n).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 summary=read('input_summary.json');features=read('input_feature_split_details.json');locs=read('outlier_top_locations.json');coast=read('coastal_stencil.json');cpu=read('cpu_gradient_path.json');peak=read('previous_peak_29p942.json');scales=read('target_rhs_scales.json')
 assert len(summary)==24 and len(features)==24*3*19
 index=json.loads((V/'index.json').read_text());manifests={}
 for row in index['datasets']:
  f=V/row['manifest'];m=json.loads(f.read_text());assert sha(f)==row['sha256']
  old=json.loads((V.parent/'preprocessing_subset_v1'/f.name).read_text())
  for k in ('train_start_indices','val_start_indices','test_start_indices','input_time_indices','target_time_indices'):assert m[k]==old[k]
  assert sha(V/m['climatology_file'])==m['climatology_sha256']
  for source,h in m['source_sha256'].items():assert sha(R/source)==h
  original=json.loads(Path(m['subset_path']).read_text());assert m['train_start_indices']==original[str(m['subset_ratio'])]['start_indices']
  manifests[m['subset_ratio'],m['subset_seed'],m['lead']]=(f,m)
 for seed in (42,7,123):
  for lead in (7,14):
   sets=[set(manifests[ratio,seed,lead][1]['train_start_indices']) for ratio in (10,25,50,100)]
   assert all(a<b for a,b in zip(sets,sets[1:]))
 for file in ('audit_v2_input_outliers.py','audit_v2_training_path.py','report_v2_final_preflight.py'):ast.parse((R/'scripts'/file).read_text())
 # CSV preserves complete feature/split moments and threshold counts for direct review.
 flat=[]
 for r in features:
  v={k:r[k] for k in ('ratio','seed','lead','split','variable','normalization_mean','normalization_std','n','min','max','mean','std','empty_doy_n')}
  for t in ('5','10','20'):
   for key in ('counts','fractions','empty_doy_counts','empty_doy_fractions','observed_doy_fractions'):v[key+'_gt'+t]=r[key][t]
  flat.append(v)
 with (O/'input_feature_split_details.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 # Proposal, not runnable commands: a corrected manifest/scale is a hard prerequisite.
 jobs=[];root=R/'outputs/subset_coastal_pending'
 def job(ratio,seed,lead,arm,lp,lr,stage):
  f,m=manifests[ratio,seed,lead];tag=str(lp).replace('.','p')
  d=root/stage/f'ratio_{ratio}'/arm/f'lambda_{tag}'/f'lead{lead}_seed{seed}'/'attempt_001'
  jobs.append(dict(id=len(jobs),status='BLOCKED_NOT_LAUNCHABLE',ratio=ratio,seed=seed,lead=lead,arm=arm,lambda_physics=lp,lambda_rres=lr,physics_heads=(arm!='baseline'),predictor='JejuHiLAM hidden256 temporal256 processor3 direct',current_manifest_for_audit=str(f),current_manifest_sha256=sha(f),required_execution_manifest=None,required_stencil='ocean_neighbors',output_directory=str(d),log_path=str(d/'run.log'),checkpoint_path=str(d/'jeju_hilam_best.pt'),stage=stage))
 for ratio in (10,25,50,100):
  for lead in (7,14):
   for seed in (42,7,123):
    job(ratio,seed,lead,'baseline',0.,0.,'validation_candidates')
    for lp in (.01,.1,.3):job(ratio,seed,lead,'physics_only',lp,0.,'validation_candidates')
 for lead in (7,14):
  for seed in (42,7,123):
   job(10,seed,lead,'auxiliary_only',0.,10.,'ablation')
   job(10,seed,lead,'full_pinn','SELECTED_BY_VALIDATION',10.,'ablation')
 assert len(jobs)==108
 plan=dict(status='BLOCKED_PROPOSAL_ONLY',launch_authorized=False,required_before_preparation=['connect corrected stencil consistently in physics loss/RHS/diagnostics','derive new version of 24 RHS manifests without overwriting v2','disable inactive loss graph branches and retest','pass prescribed two-batch GPU smoke','finalize executable protocol and hashes'],base_unique_jobs_if_positive_lambda_fixed=60,experiment_A=48,experiment_B_total_conditions=24,experiment_B_reused_from_A=12,experiment_B_additional_jobs=12,lambda_candidates=[0.,.01,.1,.3],lambda_source='server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/train_physics_pilot.py:150 and summarize_physics_pilot.py:65-70',selection='Proposed per ratio/lead: minimum mean best validation RMSE over seeds 42,7,123; ties smaller lambda. lambda=0 is baseline reference; never choose on test metrics.',positive_candidate_jobs=72,baseline_jobs=24,auxiliary_jobs=6,full_jobs_max=6,maximum_actual_jobs=108,actual_count_rule='108 - 3*k, where k is the number of 10% lead groups selecting lambda=0 (k=0,1,2): full at lambda0 duplicates auxiliary-only. No extra seed42-only pilots are planned. Selected positive-lambda candidates reused without retraining.',reuse_existing_checkpoints=False,reuse_reason='Old Full/anomaly/stencil/RHS configuration differs. No eligible completed subset-only corrected runs were verified.',common=dict(batch_size=16,history=30,precision='fp32',lr=.0002,initial_lr=.00002,weight_decay=.1,virtual_epoch_steps=478,max_virtual_epochs=80,max_steps=38240,warmup_virtual_epochs=5,patience_validations=12,selection_metric='val_rmse',lambda_we_tie=0.,zero_rres_physics=True),gpu_plan='After blockers resolved: query occupancy, choose one idle GPU, one job at a time; profile the two-batch smoke before any concurrency increase. Set exact title yeonwoo before CUDA and verify own PID in nvidia-smi; do not terminate others.',restart_policy='Current trainer lacks full optimizer/RNG/sampler resume. Preserve failed attempt and restart same config from seed in a new attempt directory. Exact mid-run resume requires additional saved optimizer, RNG and cycling sampler state; do not claim resume from best checkpoint.',completed_reuse_policy='Require matching raw, manifest, climatology, split, source, model, loss/stencil, weights, protocol hashes plus explicit completed marker and verified fields/metrics. Filename existence alone is insufficient.',metrics=['SST RMSE per seed','MHW CSI/POD/FAR using unchanged evaluator','paired relative RMSE improvement=100*(baseline-model)/baseline','corrected physics residual diagnostics for all arms including baseline','pred_raw/target_raw/mask/target_index and raw date mapping for time-block bootstrap'],jobs=jobs)
 save('blocked_execution_plan.json',plan)
 gpu=read('gpu_smoke.json');assert gpu['status']=='NOT_RUN'
 verdict=dict(verdict='FAIL — 수정 필요',data='Finite input/target statistics; extreme tails recorded, no stability guarantee',training_path='FAIL: corrected coastal stencil absent; inactive terms still computed at zero weight',gpu_smoke='NOT_RUN: STEP2 failed',formal_training='NOT_STARTED',preparation='STOPPED per user instruction; proposal only',artifacts_preserved=True,source_manifest_hash_checks='24/24 PASS',coastal_rhs_ratio=coast['legacy_to_corrected_scale_ratio'],cpu_diagnostic='PASS on one actual model batch; not a GPU smoke',source_sha256={str(f.relative_to(R)):sha(f) for f in [R/'src/train_jeju_hilam.py',R/'src/train_hybrid_jeju_forecast.py',R/'src/mhw/subset_preprocessing.py',R/'src/mhw/subset_climatology.py',R/'src/mhw/models/jeju_hilam.py']})
 save('verdict.json',verdict)
 table=['| 비율 | Seed | Lead | Max abs z | abs(z)>5 (%) | abs(z)>10 (%) | abs(z)>20 (%) | 판정 |','|---|---|---|---:|---:|---:|---:|---|']
 for r in summary:table.append(f"| {r['ratio']}% | {r['seed']} | {r['lead']} | {r['max_abs_z']:.4f} | {100*r['fractions']['5']:.6f} | {100*r['fractions']['10']:.6f} | {100*r['fractions']['20']:.6f} | 유한값 검사 PASS / 안정성 미확정 |")
 train=[]
 for seed in (42,7,123):
  for split in ('train','validation','test'):
   rr=[r for r in features if r['ratio']==10 and r['seed']==seed and r['lead']==7 and r['split']==split];n=sum(r['n'] for r in rr)
   train.append(f"| {seed} | {split} | {max(max(abs(r['min']),abs(r['max'])) for r in rr):.4f} | {sum(r['counts']['5'] for r in rr):,} | {sum(r['counts']['10'] for r in rr):,} | {sum(r['counts']['20'] for r in rr):,} |")
 top=[]
 for seed in (42,7,123):
  v=max([r for r in locs if r['ratio']==10 and r['seed']==seed],key=lambda r:abs(r['z']))
  top.append(f"| {seed} | {v['variable']} | {v['date']} | {v['latitude']}, {v['longitude']} | {v['z']:.4f} | {'빈 bin' if v['empty_climatology_bin'] else '관측 bin'} |")
 timestamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
 report=f'''# Subset-only v2 최종 정밀 검증

{timestamp}. **FAIL — 수정 필요. GPU Smoke Test 및 정식 학습 미실행.**

원본 raw, v1/v2 manifest, climatology, subset 인덱스, 기존 결과 및 실행 계획을 변경하지 않았다. 새 분석은 이 폴더에만 저장했다. 기존 v2의 네 gate PASS는 전처리 누수/전달 검증 결과이며, 이번에 추가로 요구된 해안 차분 보정 적용까지 검증한 결과가 아니었다.

## A. 데이터 이상값 검사

분모는 각 조건의 실제 Train/Validation/Test 입력 텐서에 등장하는 19변수 × 해양격자 × history 날짜의 총 개수다. history 중복 횟수를 반영한다. 세 split을 합친 요약이며 split별/변수별 분포는 별도로 제공한다. 날짜가 원본에 있다는 이유만으로 미사용 날짜를 입력에 포함하지 않았다.

{chr(10).join(table)}

최소/최대/평균/std, abs(z)>5/10/20의 정확한 개수·비율, 빈 DOY/관측 DOY의 조건부 비율은 `input_feature_split_details.csv`와 JSON에 모두 있다. 해양 NaN/Inf=0, 사용 normalization std<=1e-6=0. 육지는 모델 노드에 포함되지 않는다. 검사 도구의 가중 통계는 실제 배열을 반복한 작은 기준 예제와 비교해 검증했다. 수치 기준 통과를 학습 무해성으로 해석하지 않는다.

### 10% split별 개수 (Lead 7)

| Seed | Split | Max abs z | abs(z)>5 개수 | abs(z)>10 개수 | abs(z)>20 개수 |
|---|---|---:|---:|---:|---:|
{chr(10).join(train)}

Lead 14 결과도 CSV에 있다. 모든 조건에서 abs(z)>20은 10%의 validation/test에만 있었다. seed7에 가장 집중된다. 10% train의 최대치는 seed별 11.565/13.879/12.627이며 abs(z)>20은 없다. 그러나 2-batch smoke도 전체 꼬리 분포의 안정성을 보장할 수 없으므로 후속 평가 시 분포 차이를 계속 보고해야 한다.

### 위치 및 빈 DOY와의 관계

| Seed | 변수 | 날짜 | 위도, 경도 | z | Climatology 지원 |
|---|---|---|---|---:|---|
{chr(10).join(top)}

빈 DOY는 seed42/7/123에 34/9/52개, 관측 1개인 DOY는 104/134/72개다. 10% anomaly의 abs(z)>20은 두 lead 모두 빈 bin에서 0개이며 관측이 있던 bin에만 나타났다. 따라서 이번 극값이 빈 날짜 보간 때문에 생겼다고 단정할 근거는 없다. 관측 1개 bin의 학습 anomaly가 0이 되고 train std가 줄어드는 희소성은 여전히 존재한다. 보간 bin보다 관측 bin에서 큰 꼬리가 나타났다는 연관성만 확인했으며 인과를 주장하지 않는다.

기존 29.942는 {peak['variable']}, {peak['date']}, 위경도 {peak['latitude']}/{peak['longitude']}의 z={peak['z']:.6f}였다. **그 날짜는 seed7/10%의 두 lead 모두 train/validation/test input-history에 포함되지 않는다.** 이전 QC는 전체 raw 날짜를 분석했기 때문이다. 실제 입력 최대치는 28.3553이다.

`outlier_coordinates_gt5_ratio*_seed*.npz`는 abs(z)>5인 모든 실제 입력 날짜/해양 노드 좌표와 z를 중복 없이 저장한다. `outlier_top_locations.json`은 변수·그룹별 top20과 split별 실제 반복 횟수를 저장한다. 여기서 raw_input은 표준화 전 입력 변수 값이며 anomaly 변수에서는 raw 수온 자체가 아닌 anomaly다.

Target mean 범위는 {min(x['target_mean'] for x in scales):.6f}–{max(x['target_mean'] for x in scales):.6f}, std는 {min(x['target_std'] for x in scales):.6f}–{max(x['target_std'] for x in scales):.6f}이다. 모두 유한/양의 std이며 raw target과 분모 경로는 정상이다. 저장 RHS mean-square는 {min(x['rhs_mse_kday2'] for x in scales):.6f}–{max(x['rhs_mse_kday2'] for x in scales):.6f} (K/day)^2로 유한하지만, 아래 해안 처리 문제 때문에 의도한 physics 설정에 맞는 정상 scale이라고 판정할 수 없다.

## B. 학습 경로 검증

프로젝트 루트: `/ceph_data/yeonwoo/MHW_subset`. 표의 파일 경로는 이 루트 기준이다.

| 항목 | PASS/FAIL | 코드 근거 | 비고 |
|---|---|---|---|
| 실제 v2 dispatch 및 새 anomaly | PASS | src/train_jeju_hilam.py:643-647; src/mhw/subset_climatology.py:82-132 | public loader 실행, raw/NPZ 해시 확인 |
| v1/Full 통계 덮어쓰기 방지 | PASS | src/mhw/subset_preprocessing.py:135-203; src/train_jeju_hilam.py:681-692 | manifest RHS 주입 후 재계산 분기 안 탐. 기존 48 CPU setup 검사와 소스 해시 동일 |
| 동일 manifest, nested 및 val/test 유지 | PASS | src/mhw/subset_preprocessing.py:135-167 | 24개 원본 인덱스·v1·nested·해시 독립 재검사 |
| Input/target/RHS 전달 | PASS | src/mhw/subset_preprocessing.py:152-167,187-203 | 값 전달 PASS이며 RHS 수식의 적합성은 별도 FAIL |
| 동일 predictor 및 초기화 | PASS (CPU seed42) | src/mhw/models/jeju_hilam.py:218-235,284-345 | 실제 hidden256 모델 두 개의 모든 shared parameter와 예측이 정확히 동일 |
| 배치 순서 매칭 | PASS (구현) | src/train_jeju_hilam.py:213-224 | loader 전용 generator seed+17. 실학습 전체 trace는 미실행 |
| 수정된 해안 공간차분 | **FAIL** | src/train_hybrid_jeju_forecast.py:588-601,697,770 | 해양 mask 없이 육지값을 stencil에 포함 |
| P0 daily loss 연결 | PASS (legacy 구현) | src/train_jeju_hilam.py:321-331 | daily 함수 연결은 맞으나 보정된 해안 연산이 아님 |
| Physics only의 Rres gradient 배제 | PASS (CPU) | src/train_jeju_hilam.py:343-348 | lambda_rres=0, zero_rres_physics=True에서 Rres head gradient 0 |
| Auxiliary only의 physics gradient 배제 | PASS (CPU) | 같은 위치 | lambda_physics=0, lambda_we_tie=0에서 we head gradient 0 |
| OFF 손실의 계산 자체 생략 | **FAIL** | src/train_jeju_hilam.py:321-348 | aux-only에서도 daily physics 함수 1회 호출. 가중치0은 finite일 때 기여0일 뿐, 0*NaN 보호 없음 |
| 보조 head 독립 생성/활성화 | 제한 | src/train_jeju_hilam.py:669; src/mhw/models/jeju_hilam.py:224-235 | head 두 개 생성은 physics_mode에 묶임. 손실 가중치는 독립. baseline은 head 없음, 다른 세 조건은 둘 다 있음 |
| 기존 실행 계획과 이번 A 일치 | **FAIL** | data/subsets/preprocessing_subset_v2/execution_plan.json; scripts/build_subset_preprocessing_v2.py:128-146 | 기존 A는 P0 lambda_rres=10, 이번 요청은 Physics only lambda_rres=0. 기존 72-run 계획 자동 재사용 금지 |
| Checkpoint GPU 저장/복원 | 미검증 | src/train_jeju_hilam.py:809-830 | GPU smoke 미실행이므로 PASS 아님 |

### 해안 차분 재현 결과와 영향

현재 `spatial_gradients_torch`는 해양 mask 인자가 없다. `nodes_to_grid_torch`는 육지에 0을 채우며, mask는 차분 계산 뒤 loss 집계에 적용된다. 이 방식은 해안의 해수-육지 차이를 수평 수온 gradient에 섞는다.

기존 보정 코드는 `/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py:34-61`에 있다. `ocean_neighbor_gradients`는 해양 양 이웃이 있으면 중앙차분, 한 이웃만 있으면 한쪽 차분을 쓴다. `ocean_neighbor_support`는 x/y 이웃이 모두 있는 660개 격자를 physics 집계에 사용한다. 온도 학습/평가의 661개 격자는 유지한다. 기존 runner의 연결은 같은 폴더 `train_physics_pilot.py:160,211`에 있고, 현재 진입점에는 없다. 백업 runner 전체를 실행하지 않고 순수 stencil 함수만 AST로 읽어 진단했다.

- 해양을 상수 20으로 놓고 육지를 0으로 채웠을 때: legacy는 **141개 해양 격자**에서 비영 gradient, 최대 {coast['native_constant_max_abs_gradient_K_per_m']:.9f} K/m. 보정 함수는0.
- 육지값만 100으로 바꾸면 legacy 해양 gradient가 변한다. 보정 결과는 정확히 불변.
- 동일 10%/seed42/lead7의 전체 선택 sample에서 RHS를 진단 재계산: legacy **{coast['manifest_rhs_scale']:.9f}**, 보정 **{coast['diagnostic_corrected_subset_rhs_scale']:.9f}** (K/day)^2, 비율 **{coast['legacy_to_corrected_scale_ratio']:.3f}**.
- residual 자체와 정규화 분모가 모두 달라지므로 scale 숫자만 대체하면 안 된다. 다른 23개도 같은 보정 경로로 각각 다시 계산해야 한다. 손실/gradient에 영향을 줄 수 있는 implementation inconsistency이며, 예측 성능 영향 크기는 이 진단만으로 알 수 없다.
- 기존 manifest는 수정하지 않았다. 새로운 버전으로 파생시켜 해안 연산·support mask·RHS를 일관되게 기록해야 한다. Input/target/climatology를 바꿔야 한다는 근거는 이번 검사에서 얻지 못했다.

### 실제 모델 CPU gradient 진단 (GPU smoke 아님)

실제 hidden256/processor3 모델과 v2 첫 train sample 1개를 사용했다. optimizer step은0이다. Data loss={cpu['components']['data']['loss']:.6f}, physics loss={cpu['components']['physics']['loss']:.6f}. Shared predictor gradient norm은 data={cpu['components']['data']['shared_gradient_norm']:.6f}, physics={cpu['components']['physics']['shared_gradient_norm']:.6f}, Rres={cpu['components']['rres']['shared_gradient_norm']:.6f}. 네 가중치 조합의 gradient는 finite였다. 이 결과는 legacy 경로에서 연결성을 확인한 것이며 GPU 메모리/후속 batch/해안 보정 후 안정성을 보증하지 않는다.

Rres penalty는 shared representation에 gradient를 전달한다. Physics only에서는 Rres head gradient0, Auxiliary only에서는 we head gradient0을 확인했다. 비활성 term은 계산까지 건너뛰는 분기가 필요하다. 실제 `run_epoch`에 aux-only 가중치를 적용해 daily physics 함수가 호출됨을 별도 확인했다.

## C. GPU Smoke Test

| 모델 | Forward | Backward | Loss finite | Gradient finite | 판정 |
|---|---|---|---|---|---|
| Baseline | 미실행 | 미실행 | 미검증 | 미검증 | BLOCKED |
| P0 Physics only | 미실행 | 미실행 | 미검증 | 미검증 | BLOCKED |

사용자가 STEP1~2 통과 후에만 smoke를 허용했으므로 해안 차분 FAIL 상태에서 실행하지 않았다. 초기 제한된 실행의 nvidia-smi는 실패했으나 후속 조회는 성공(returncode0)했고 RTX A6000 10개, 조회 당시 활성 GPU 프로세스0개였다. GPU 미실행 이유는 드라이버 불가가 아니라 **코드/scale 선행 조건 미충족**이다. 사용자의 GPU 작업/다른 프로세스를 시작하거나 종료하지 않았다. Peak GPU memory, CUDA forward/backward, clip norm1 전후, checkpoint round trip은 아직 검증하지 않았다.

## D. 정식 학습 계획 (수정 후 재검증을 위한 초안)

| 실험 | 모델 조건 | Seed | Lead | 기본 고유 실행 수 | 상태 |
|---|---|---|---|---:|---|
| A | 4비율 Baseline | 42/7/123 | 7/14 | 24 | BLOCKED |
| A | 4비율 P0 Physics only | 42/7/123 | 7/14 | 24 | BLOCKED |
| B 추가 | 10% Auxiliary only | 42/7/123 | 7/14 | 6 | BLOCKED |
| B 추가 | 10% Full PINN | 42/7/123 | 7/14 | 6 | BLOCKED |

B의 10% Baseline/Physics only 12개는 A와 모든 설정·manifest가 같을 때만 재사용한다. 따라서 양의 lambda 하나가 정해졌을 때 기본60회다. 현재 v2의 기존72개 계획을 수정하지 않고 별도 `blocked_execution_plan.json`에 이번 요구를 반영했다. **실행 스크립트 확정은 사용자 지시에 따라 중단했으며, 이 JSON에는 실행 명령을 넣지 않았다.**

기존 해안 보정 pilot의 후보는 0/0.01/0.1/0.3이며, 0은 matched baseline이었다. 근거: 백업 `train_physics_pilot.py:150`, `launch_physics_pilot_4plus4.py:37-43`, `summarize_physics_pilot.py:65-70`. 기존 선택 기준은 세 seed의 best validation RMSE 평균 최소, 동률이면 작은 lambda다. 기존 lambda pilot의 lambda0은 Rres penalty가 켜져 있던 별도 조건이므로 순수 baseline과 혼동하지 않는다.

새 초안은 각 ratio/lead별로 같은 세 seed 평균 validation RMSE를 비교한다. Test RMSE는 선택에 사용하지 않는다. 선택용 집계에는 validation 메타데이터만 입력하고 모든 후보를 보존한다. 선택 후 checkpoint를 다시 학습하지 않고 그대로 재사용한다. 후보72회(3양의lambda×24조건)+Baseline24회+Aux6회+Full6회로 **최대108회**다. 10%에서 lambda0이 선택된 lead가 k개라면 Full이 Aux와 같아 3k개를 재사용하여 **108−3k회(k=0/1/2)**다. 양의 효과가 없는 경우 baseline 선택을 그대로 보고한다. 이는 제안 프로토콜이며 해안 보정·RHS 재생성·Smoke 통과 전 확정하지 않는다.

모델은 기존 predictor를 유지한다. 현재 native 정의를 기준으로 baseline은 auxiliary head 없음, 나머지 세 조건은 head2개 있음으로 명시했다. CPU seed42에서는 shared predictor 초기화와 출력이 정확히 같았다. 향후 A/B 같은 arm은 head 구조·seed·배치순서까지 동일하게 고정해 재사용해야 한다. 모든 head를 네 arm에 동일하게 둘 경우에는 기존 계획과의 차이를 명시하고 새로운 실행 설정으로 검증해야 한다.

공통 계획: history30, batch16, fp32, lr0.0002(initial0.00002), weight_decay0.1, 478step마다 validation, 최대38,240step, warmup5 virtual epoch, val RMSE patience12. 조기종료 때문에 실제 step 수는 달라질 수 있다. lambda_rres는 baseline/physics-only 0, aux/full 10; lambda_we_tie=0; zero_rres_physics=True. RHS가 바뀌면 같은 lambda_rres의 유효 강도도 바뀌므로 다시 smoke/gradient 진단해야 한다.

GPU는 후속 조회에서 비어 있는 장치1개, 동시1개로 시작하고 두 batch smoke의 peak memory를 본 뒤 확대 여부를 정한다. GPU 사용 전에 exact title yeonwoo를 설정하고 nvidia-smi PID로 확인한다. 모든 tmp/cache/log/checkpoint/예측은 Ceph 내부에 저장한다.

각 조건의 현재 manifest 경로·모델 설정·seed/lead·lambda·출력/로그/checkpoint 계획 경로는 JSON의108개 job 항목에 있다. `required_execution_manifest=null`과 `BLOCKED_NOT_LAUNCHABLE`을 유지하므로 현재 잘못된 scale로 실행 가능한 계획이 아니다.

재시작: 현재 trainer는 model/stats/best epoch만 저장하고 optimizer/RNG/순환 sampler의 완전한 resume 상태를 저장하지 않는다(src/train_jeju_hilam.py:809-819). 실패 결과를 보존한 뒤 새 attempt 디렉터리에서 동일 seed/config로 처음부터 재시작한다. best checkpoint에서 정확히 이어지는 resume라고 주장하면 안 된다. 중간 재개를 지원하려면 그 상태 저장·복원과 별도 재현성 검사가 필요하다.

완료 재사용: raw/manifest/climatology/split/source/model/loss/stencil/weights/protocol 해시와 완료 표시, metrics/fields 유효성을 모두 확인한다. 기존 Full/v1/다른 해안 설정 checkpoint는 사용하지 않는다. 파일 존재만으로 완료 판정하지 않는다.

결과 구조 제안: `outputs/subset_coastal_pending/{{stage}}/ratio_{{r}}/{{arm}}/lambda_{{lambda}}/lead{{h}}_seed{{s}}/attempt_001/`에 config/provenance, run.log, best checkpoint, history.csv, metrics.json, test_fields.npz, completion.json. 기존 `collect_predictions`(src/train_jeju_hilam.py:418-467)는 pred_raw/target_raw/mask/target_index를 저장한다. 날짜 매핑을 같이 저장해 시간블록 bootstrap에 사용한다. SST RMSE, seed별 성능, paired 상대 개선율 `100*(RMSE_baseline-RMSE_model)/RMSE_baseline`, 모든 arm의 공통 corrected physics residual을 산출한다. native baseline에는 physics diagnostic이 없으므로 공통 사후 진단 경로가 필요하다. MHW CSI/POD/FAR는 기존 `src/summarize_jeju_hilam_mhw_skill.py:32-39` 평가기를 그대로 사용하고 입력 anomaly climatology로 임계값을 바꾸지 않는다.

## E. 최종 판정과 수정 순서

**FAIL — 수정 필요.** 누수 없는 v2 입력 통계는 확인됐지만, 의도한 해안 차분이 적용되지 않고 그에 맞는 RHS scale도 아니다. 현재 상태에서는 정식 학습 진행 가능이라고 보고할 수 없다.

1. 기존 v2/raw/결과를 그대로 보존한다.
2. 검증된 ocean-neighbor 차분과 support mask를 daily loss, RHS fitting, 공통 physics diagnostic에 일관되게 연결한다. 상수 해수 gradient0·육지값 불변·선형장 차분·고립 격자 처리를 회귀 검사한다.
3. 새 버전의24개 physics scale/manifest를 생성한다. anomaly/input/target 통계는 유지 가능 여부를 다시 확인하고, 변경된 소스/차분 정의 해시를 기록한다.
4. lambda0 손실 경로는 objective에서 계산을 생략하고 diagnostic은 no_grad로 분리한다. 비활성 함수 호출 방지·0*NaN 방지·활성 gradient 유지 검사를 추가한다.
5. Baseline/P0 Physics only 각각 최대2 GPU batch로 forward/backward/gradient clipping/checkpoint 저장·복원을 검증한다. 통과 후에만60회 기본 실험과 lambda 후보 계획/런처를 확정한다.

수정 방안을 먼저 보고하라는 지시에 따라 위 핵심 학습 코드/데이터 수정은 수행하지 않았다. 이번에 새로 쓴 감사 코드는 CPU 기준 예제, legacy/보정 차분 대조, 실제 모델 loss gradient, artifact 해시 검사로 검증했다. 원본 모델 소스·manifest·기존 실행 결과 변경 없음.
'''
 (O/'report.md').write_text(report)
 save('report_verification.json',dict(manifest_count=24,feature_split_rows=len(features),plan_jobs=108,all_planned_jobs_blocked=True,source_hashes_match_v2=True,nested_indices_preserved=True,report_generated_kst=timestamp))
 with (R/'docs/research_log.md').open('a') as f:f.write(f'\n\n## {timestamp} — v2 정밀 감사 최종 FAIL 보고\n\n- scripts/report_v2_final_preflight.py 추가, outputs/v2_final_preflight_20261008/의 보고서·CSV·verdict·blocked_execution_plan·29.942 위치 진단 생성.\n- 입력24조건/1,368변수-split 검사 및 전체 abs(z)>5 좌표 기록. 기존29.942는 실제 입력 미사용 날짜. 입력 최대28.355; abs(z)>20은10% held-out에만 존재.\n- 해안 보정 미연결 및118.28배 RHS 차이로 FAIL. zero-weight branch가 계산 자체를 끄지 않는 점, 기존72-run 계획의 Rres 가중치 불일치, 완전 resume 미지원도 기록했다.\n- 24개 manifest/climatology/source 해시·nested·원본 인덱스 보존 검사 PASS. GPU smoke·정식 학습·optimizer step 없음. 사용자 지시에 따라 정식 실행 준비를 중단하고60회 기본/최대108회 후보 포함 계획은 BLOCKED 초안으로만 저장했다.\n')
 print(json.dumps(verdict,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
