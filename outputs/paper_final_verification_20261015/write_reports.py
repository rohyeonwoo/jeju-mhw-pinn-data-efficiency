"""Render evidence-linked Korean audit reports; no scientific files modified."""
from pathlib import Path
import json, pandas as pd,numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';T=R/'outputs/v8_actual_epoch_oracle96_20261011';A=R/'outputs/v8_actual_epoch_auxiliary24_20261015'
J=lambda p:json.loads(Path(p).read_text());stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
def link(p,n=1,text=None):
 p=Path(p);p=p if p.is_absolute() else R/p
 return f'[{text or str(p.relative_to(R)) if p.is_relative_to(R) else text or p.name}]({p}:{n})'
def func(p,needle):
 p=Path(p);p=p if p.is_absolute() else R/p
 n=next(i for i,s in enumerate(p.read_text().splitlines(),1) if needle in s)
 return link(p,n,needle)
def artifact(n):return link(O/n)
def write(name,title,body): (O/name).write_text(f'# {title}\n\n작성: {stamp}. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.\n\n'+body.strip()+'\n')
def table(d,cols=None):
 if cols is not None:d=d[cols]
 fmt=lambda v: f'{v:.7g}' if isinstance(v,(float,np.floating)) else str(v)
 return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
audit=J(O/'audit_status.json');phy=J(O/'physics_numerical_checks.json');pr=pd.read_csv(O/'protocol120.csv');tests=pd.read_csv(O/'stencil_tests.csv');sc=pd.read_csv(O/'rhs_scale_recomputed.csv')
model='src/mhw/models/jeju_hilam.py';native='src/train_jeju_hilam.py';hybrid='src/train_hybrid_jeju_forecast.py';mixed='scripts/oracle96/mixed_budget_physics.py';worker='scripts/oracle96_actual/worker.py';aux='scripts/oracle96_auxiliary24/aux_worker.py';sub='src/mhw/subset_preprocessing.py'
write('model_definition_audit.md','모델 정의·손실·실행 코드 감사',f'''
**판정: 현재 Oracle120 PASS. WSDM P1–P3 실행 시점 연결은 UNVERIFIED.**

Oracle96 96개와 Auxiliary24 24개의 config/history/checkpoint audit를 직접 읽고, Validation-best 파일 SHA를 다시 계산했다. 실행 프로토콜의 소스 해시 기록 53개가 현재 파일과 모두 일치한다(중복 경로를 포함한 기록 수). 프로젝트에는 접근 가능한 Git 저장소가 없으며, 이 검증의 버전 근거는 Git commit이 아니라 저장된 SHA 및 config이다. {artifact('source_hashes.json')}, {artifact('protocol120.csv')}. 구형 manifest도 commit을 unknown으로 기록한다.

모두 SeaCast-inspired JejuHiLAM: 1층 GRU(256), 3단계 hierarchical graph, hidden256, processor3, MLP hidden1, direct SST output. 19개 동적 입력과 graph 정적 속성을 사용한다. ConvLSTM과 동일 모델이 아니다. shared 범주(SST head 포함) 14,887,681 + we head66,049 + Rres head66,049 = **15,019,779 parameters**. 전체 `requires_grad` 구조는 같지만 비활성 head는 gradient가 None이다. SST head도 모두 같다.

근거: {link(model,114)}, {link(model,167)}, {link(model,219)}, {link(model,326)}; 실제 공통 초기 tensor/hash 비교는 {func(aux,'def setup(')} 및 120개 config의 `shared_initial_sha256`, `parameters`, `tensor_fingerprints`.

| 모델 | we head 존재/목적함수 활성 | Rres head 존재/활성 | 실제 목적함수 | 물리 RHS |
|---|---|---|---|---|
| Baseline | 있음/OFF | 있음/OFF | D | 없음 |
| Auxiliary-only | 있음/OFF | 있음/ON | D+0.0001 U | 없음 |
| P0 | 있음/ON | 있음/OFF | D+λ(e)P0 | flux+수평 이류+학습 연행 |
| NoAdv | 있음/OFF | 있음/OFF | D+λ(e)P_A | flux−penetration+MLD 연행 proxy |
| Adv | 있음/OFF | 있음/OFF | D+λ(e)P_B | NoAdv+수평 이류 |

D = mean[(pred_norm−target_norm)²], target_mean/std는 subset 고정값. P = 유효 support 평균[(clip(r,−10,10)/sqrt(S))²]. U = 해양 평균[(86400 Rres)²]/S. S는 `(K/day)²` 단위의 고정 P0 RHS mean-square이며 **std가 아니다**. λ(e)=0.1 min(e/5,1); Aux는 0.0001 고정, 물리 warmup 없음. 모든 모델 entrainment-tie=0, adaptive/adversarial loss 없음.

근거: {func(native,'def compute_batch_losses(')}, {func(hybrid,'def compute_physics_loss(')}, {func(hybrid,'def rres_penalty(')}, {func(worker,'def arguments(')}, {func(aux,'def setup(')}.

Gradient: D→SST head→shared. P0→SST interpolation/derivative 및 we head→shared. NoAdv/Adv→예측 SST interpolation→shared; 두 auxiliary head는 사용하지 않는다. U→Rres head→shared이나 SST output head에는 U의 직접 경로가 없다. 초기 및 각 학습 epoch의 physics predictor gradient 기록이 양수인 검사와 Aux autograd 진단을 보존했다. `Physics only`와 이번 `Adv`는 head 수의 차이가 아닌 RHS의 차이다. 과거 `Full PINN`은 physics+Rres이고 **이번 Adv는 Full PINN(aux ON)이 아니다**.

NoAdv/Adv는 동일 모듈 `path(..., advection=False/True)`의 단일 분기로만 다르다. NoAdv에서도 항 진단을 위해 gradient/advection 숫자는 계산하지만 loss RHS에는 포함하지 않아 그 항의 gradient는 전달되지 않는다. 공통 mask와 RHS scale도 같다. {link(mixed,25)}, {link(mixed,30)}, {link(mixed,42)}.

P0와 Adv의 차이: 단파 투과의 유무, 학습 we×1K와 관측 MLD-deepening×(NOAA proxy−Tb), 기존 budget2d 산술 ML 유속과 3D 두께가중 ML 유속. 따라서 P0→Adv 결과를 '이류 추가 효과'로 해석하면 안 된다.

구형 WSDM 정의 매핑은 P0=original, P1=original+vertical, P2=original+diffusion, P3=original+vertical+diffusion. {link(hybrid,55)}. 현재 NoAdv/Adv에 P1–P3의 vertical/diffusion은 없다. 구형 원고·P1–P3 전용 실행 config/checkpoint/source-hash 연결을 확보하지 못했으므로 동일 실험이라는 주장은 **UNVERIFIED**. {artifact('legacy_inventory.json')} 및 experiment_protocol_audit.md 참고.
''')
write('horizontal_advection_audit.md','수평 이류 수식·단위·좌표 검증',f'''
**판정: 현재 균일 0.25° 격자의 실행 계산 PASS. 원격자→병합 최초 처리 및 불균일 격자 일반화는 한계.**

1. budget2d `u_ml`, `v_ml`: m s⁻¹, 각각 ML-averaged eastward/northward velocity. Adv의 3D `uo`, `vo`도 각각 eastward/northward m s⁻¹. 원래 파일의 실제 attrs를 저장했다: {artifact('dataset_metadata.json')}, {artifact('physics_numerical_checks.json')}.
2. tensor의 마지막 축=x=longitude, 뒤에서 두 번째=y=latitude. 실제 latitude30→37°, longitude123→131°, 간격 모두+0.25°. `stored_direction=decreasing` 속성은 실제 배열과 모순된다. 계산은 이 속성이 아니라 좌표값의 부호를 사용한다. 이 불일치를 실제 v 부호 오류로 바꾸어 해석하지 않는다.
3. dx=R cos(lat) gradient(lon_radians,axis=1), dy=R gradient(lat_radians,axis=0), R=6,371,000m. 부호 있는 간격을 유지하므로 좌표 감소 순서에서도 테스트가 통과한다. {func('src/mhw/datasets.py','def compute_grid_spacing_m(')}. degree를 그대로 미분 분모로 사용하는 활성 경로 없음.
4. D_xT,D_yT는 K/m. A_h=−u D_xT−v D_yT는 K/s. P0는 최종 residual/RHS에86400을 곱하고, Adv는 항 내부에서−86400(u D_xT+v D_yT)를 만든다. 서로 중복 변환하지 않는다. {link(hybrid,690)}, {link(hybrid,725)}, {link(mixed,28)}.
5. 최초 재격자화/일평균 코드·가중치·원격자 u/v 파일이 없어 그 단계의 벡터 정합성까지 PASS로 확대하지 않는다. 현재 3D/budget2d time/lat/lon 배열은 정확히 일치하고, feature builder가 순서를 검사한 뒤 uo/vo를 동일 depth 가중으로 투영한다. {link('scripts/prepare_mixed_budget_features.py',16)}, {link('scripts/prepare_mixed_budget_features.py',22)}, {link('outputs/glorys_source_inventory_20261011/report_ko.md') if (R/'outputs/glorys_source_inventory_20261011/report_ko.md').exists() else link('outputs/glorys_source_inventory_20261011/report.md')}.

수치 검증: FP32 선형장 T=20+2e−5 x−1e−5 y, u=.2,v=−.1에서 기대−0.432K/day, 최대 오차2.038e−6. FP64 최대4.385e−15. 상수장, u/v 반전, 실제 해안 mask, NaN/Inf 육지, 위도 보정 및 좌표 역순, 16개 이웃 조합 포함88/88 통과. {artifact('stencil_tests.csv')}.

비균일 좌표의 편측차분은 중심 좌표의 평균 간격을 사용한다. 0.25°/0.5° 양쪽 간격에서 한 이웃을 제거하면 진짜 선형장 미분보다33.333% 큰 사례가 재현된다. 이는 **일반 불균일 격자 차분으로서의 제한**이며, 모든 실제 좌표 간격이0.25°인120개 모델의 오류로 판정하지 않는다. F 테스트의 PASS는 명시된 구현 규칙 재현이고, 임의 격자에서의 수학적 정확성 PASS가 아니다.

논문 문장: “Horizontal temperature gradients are computed in metres using ocean-neighbour finite differences on the regular latitude–longitude grid, with the longitude spacing corrected by cos(latitude); the resulting advective tendency is expressed in K day⁻¹.”
''')
write('coastal_stencil_tests.md','해안 stencil CPU 수치 시험',f'''
**88/88 PASS; 운영 코드 변경 없음.** {artifact('audit_cpu.py')}, {artifact('stencil_tests.json')}.

실제 구현 {link('src/mhw/ocean_stencil.py',8)}, {link('src/mhw/ocean_stencil.py',15)}, {link('src/mhw/ocean_stencil.py',21)}:

| 이웃 | 미분 규칙 | 물리 손실 support |
|---|---|---|
| 양쪽 바다 | (T_hi−T_lo)/(2Δ) | 각 축에 최소1개 있으면 포함 |
| 한쪽 바다 | (T_hi−T)/Δ 또는(T−T_lo)/Δ | 반대 축도 최소1개 있으면 포함 |
| 해당 축 이웃 없음 | 해당 축 미분0 | 해당 셀 제외 |
| 고립 바다 | 두 축0 | 제외 |
| 바깥 경계 | 영역 밖을 이웃으로 보지 않음; 내부 쪽 편측 | 동일 support 규칙 |
| 육지 | arithmetic 전에 T를0으로 대체; 출력0 | 제외 |

실제 해양661개, physics support660개. 육지0/−100/1000/NaN/Inf를 바꿔도 해양 미분 차이0. 유효한 바다 자체의 NaN을 숨기는 규칙은 아니다. Native/current P0, scale, NoAdv/Adv는 같은 함수를 공유한다. NoAdv도 Adv와 support를 동일하게 유지하므로 'gradient가 필요 없어서 해안 support를 늘린 모델'이 아니다.

FP32 허용 오차는 사전 식 `86400 × eps(dtype) × 32 × max|T| / 23000`으로 정했다. 온도 차의 취소오차·분모·곱셈을 포함한 보수적 연산 bound이며, 실제 오차보다 느슨한 별도 성능 기준을 탐색하지 않았다. FP64도 같은 식. 지리 spacing이float32로 반환되는 테스트에는 rtol1e−7/atol1e−12 K/m. 상수장·부호반전·육지불변·과거 연산자 비교에는 exact equality.

{table(tests[tests.test.str.contains('constant|linear|geographic|nonuniform|archived|actual_mask')],['test','expected_min','expected_max','calculated_min','calculated_max','max_abs_error','max_rel_error','atol','rtol','status'])}

전체16개 해양 이웃 조합×float32/64의 계산 기대값과 support는 CSV에 각각 기록했다. 상대오차는 기대값0 부근에서 해석하지 않고 절대오차를 사용한다.

과거 보정 구현 {link('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py',46)}과 실제 mask의 유한장 결과는 bitwise 동일. current는 NaN/Inf 육지 arithmetic도 사전에 차단한다. 과거30회 pilot는 runtime patch로 loss와 scale 모두 보정됐다는 별도 hash/checkpoint 감사가 있다: {link('outputs/historical_coastal_audit_20261008/report.md')}.

반면 v2 원본 native 미분은 mask가 없었다: {link('outputs/v3_preflight_20261008/originals/src/train_hybrid_jeju_forecast.py',588)}. 이 오류를 이미 보정된 Oracle120에 다시 귀속하지 않는다. 비균일 격자의 F 규칙 재현 PASS는 비균일 격자 정확도 보장이 아니며 horizontal_advection_audit.md에 구체적인33.333% 편차를 남겼다.
''')
write('physics_equation_audit.md','물리 잔차·단위·정규화 감사',f'''
**현재120의 활성 물리 경로: PASS WITH LIMITATIONS. 모든 잔차는 clipping 전에 K/day이다.**

호출: {func(worker,'def run_training(')} → {func(native,'def run_epoch(')} → {func(native,'def compute_batch_losses(')} → {func('scripts/oracle96/train_mixed_budget_gpu.py','def daily(')} → P0 native daily_rollout 또는 A/B mixed loss → {func(hybrid,'def compute_physics_loss(')}. `worker.main`의 install로 실제 dispatch가 등록된다. 명목상 함수 이름만 보고 현재 모듈을 추정하지 않았다.

현재시점t=start+29, leadL=7/14, T_j=T_t+(j/L)(T̂_(t+L)−T_t), j=0,…,L−1. δT=(T̂−T_t)/L는1일 증가량이며 K/day tendency로 비교한다. forcing은 `[t:t+L]` 즉t,…,t+L−1에 대응한다. P0는 `(T_(j+1)−T_j)/86400` K/s를 먼저 만들고 전체 residual에86400을 곱한다. A/B는 increment를직접K/day로 사용한다. lead에 따른 이중86400 또는 누락 없음.

H_j=max(mlotst_j,5m), D는 해양 이웃 미분, ΔT_d=1K, S는 선택 Train의 P0 RHS mean-square.

**P0 코드 수식**

r_j = (T_(j+1)−T_j) − 86400[ qnet_j/(1025·3990·H_j) − u_ml,j D_xT_j − v_ml,j D_yT_j − ŵ_e·1/H_j ].

ŵ_e = 1e−5 softplus(head_we−1) m/s. 직접 관측된 연행속도가 아닌 학습 latent output. 7/14일의 모든 daily step에 동일 forecast head field를 사용한다. Rres, vertical, diffusion 모두0. P0 함수 안 observed target tendency를 만들지만 현재 fixed RHS normalization에서는 loss에 쓰이지 않는다. {link(hybrid,659)}, {link(hybrid,803)}, {link(model,340)}.

**NoAdv / Adv 코드 수식**

w_e,j = max(0, max(Hraw_j,5)−max(Hraw_(j−1),5))/86400 m/s.

F_j = 86400(qnet_j−ssr_j exp(−0.06 H_j))/(1025·3990·H_j).

E_j = −86400 w_e,j (T_j−Tb_j)/H_j.

A_j = −86400(u_j D_xT_j+v_j D_yT_j).

r_A,j = δT−(F_j+E_j), r_B,j = δT−(F_j+E_j+A_j).

Tb는H보다깊은첫유효thetao 중심층, u/v는midpoint-interface두께가중ML평균. P0의기존u_ml/v_ml와달리featurecache에서재계산. 미래NOAA정답은이식에입력되지않는다. 근거 {link(mixed,9)}, {link(mixed,25)}, {link(mixed,30)}, {link('scripts/prepare_mixed_budget_features.py',23)}.

| Quantity | 실제 입력 단위 | 연산/변환 | 최종 단위 |
|---|---|---|---|
| qnet,ssr | W m⁻² | qnet=ssr+str+slhf+sshf, 가열양수 | W m⁻² |
| rho,Cp,H | kg m⁻³,J kg⁻¹ K⁻¹,m | q/(rho Cp H) | K/s |
| flux,penetration | 위 열항 | ×86400; penetration 차감 | K/day |
| u,v,gradient | m/s,K/m | −86400(uTx+vTy) | K/day |
| P0 ŵ_e | m/s | −86400 ŵ_e·1K/H | K/day |
| A/B w_e | MLD차/86400 m/s | −86400 w_e(T_j−Tb)/H | K/day |
| SST increment | K/1day | direct 차이를L로 나눔 | K/day |
| r_j | 위 동일단위 항의 차 | clip[−10,10] | K/day |
| S | pooled mean(RHS²) | residual/sqrt(S) | S:(K/day)²; normalized r:무차원 |
| Rres | 1e−4 tanh(head), K/s | ×86400 후제곱/S | penalty 무차원 |

qnet 전기간 해양 조합검산: 최대 차이 {phy['qnet_component_max_abs_difference_wm2']:.7g} W/m², RMS {phy['qnet_component_rmse_wm2']:.7g} W/m² (float32합산 규모). qnet/ssr에W/m² attrs가 있고, ssr는누적에너지→평균flux변환기록이있다. 이를재차86400으로나누지않는다. 최초ERA5파일의시간평균처리 전체 provenance는별도한계이며, 이검사만으로 복원했다고하지않는다.

Clip은r_j의물리단위에서±10K/day, normalization및square이전이다. clipping이threshold밖의gradient를0으로만들수있으므로 rawRMS와clippednormalizedloss는구별한다. {link(hybrid,455)}.

**RHS scale 24개 재계산**

선택startwindows, 관측current/target를선형보간, we=0인P0resolvedRHS를 사용하여pooledmean(RHS_Kday²) 계산. `S`라는변수는standard deviation/RMS가아니며 sqrt(S)가RMS. inputmean/std=고유선택history날짜×해양; target=선택target날짜×해양. scale은타깃을쓰지만 Train만이며모델독립. NoAdv/Adv별자체RHSscale은없고모든arm이같은P0 S를쓴다. 물리학적으로항별상대크기를동일화한것은아니다.

24개모두양수finite·재계산일치. 허용rtol2e−6, atol1e−9는FP32배치 reduction차이이며최대절대차이 {sc.abs_error.max():.8g}. {artifact('rhs_scale_recomputed.csv')}; {func(sub,'def compute_subset_rhs_scale(')}, {func(hybrid,'def compute_fixed_rhs_scale(')}, {func(sub,'def apply_manifest_physics(')}. manifest적용이legacyv1/v2/다른stencil/상충CLIscale을거부한다. Full통계덮어쓰기없음은120개config의실제통계일치로도검증했다.

{table(sc)}

동일실제batch/예측에서loss RHS와scale RHS구현대조: RHS최대차 {phy['p0_rhs_runtime_max_error_kday']:.8g}K/day, residual최대차 {phy['p0_residual_runtime_max_error_kday']:.8g}K/day;float32허용2e−6내. Predictor field gradient norm {phy['prediction_gradient_norm']:.8g},finite. {artifact('physics_numerical_checks.json')}.

**미해결항 및 예전 확장식**

현재Rres는Auxiliary-only의0-targetpenalty로만사용; 모든물리RHS에서제외. 원논문의 `+10L_rres`를현재P0/Adv설명에사용하면틀린다. 기존P1vertical은precomputedvadv_kday/86400을K/sRHS에더함;생성시positive-down수심에서`+w*dT/dz`를86400배하고±2K/day clip. P2diffusion은Kh laplacian(T),Kh기본100m²/s. 두경로모두현재120개에서비활성이다. {link(hybrid,595)}, {link(hybrid,615)}, {link(hybrid,627)}, {link('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py',125)}. 현재laplacian은ocean-neighborLaplacian이아니고경계2차분0이며, 이를현재Adv에사용된것처럼해석하지않는다.

**Common Residual 비교 가능 범위**

native로그의`common_res_*`는이름과달리각모델자체RHS/학습we를사용한다. Baseline비활성값이나Aux의진단생략0을물리잔차0으로해석하면안된다. NoAdv자체r_A와Adv자체r_B의감소율도동일식비교가아니다. {link(native,180)}, {link(native,363)}, {func(aux,'class Monitor(')}.

이번재분석은모든120예측에공통A와공통B식각각을적용하고동일관측MLDwe/Tb/mask를사용했다. 따라서Baseline의존재하지않거나훈련되지않은head를물리관측처럼대입하지않는다. 과거공통평가는별도로zero-we/own-latent-we/shared-P0-we를구분했다: {link('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/final_clean_evaluate.py',179)}. zero-we는연행제외진단, shared-P0-we는P0에의존한진단이며독립관측열수지아니다. 원고의정확한감소율·대상표가없어그수치자체는UNVERIFIED. 공통식의rawresidual과예측오차를같이제시하고'진정한물리정합성증명'이라고하지않는다.

**논문에 유지해야 할 한계**: NOAA SST는GLORYSML평균의대리변수(기존ValidationRMSE약1.185K); 전체순단파에대한단일지수근사,Kd=.06은pilot가정; 투과항후CPU잔차평균부호반전은가열과보정가능성이지입증이아님; missingvertical/diffusion/mixing/assimilation항; 관측미래forcing은Training-onlyoracle감독. SST-label efficiency이며total-information efficiency가아니다. inferenceforcing불변성은추가감독정보가없다는증명이아니다.

권장영문: “Daily physics residuals are expressed in K day⁻¹ (the P0 residual is first formed in K s⁻¹ and multiplied by 86,400), clipped to [−10, 10] K day⁻¹, and normalized by the fixed P0 RHS RMS computed from each selected training subset.”
''')
write('experiment_protocol_audit.md','Oracle120 학습 프로토콜·전처리·역사적 비교',f'''
**Oracle120 120/120 PASS.** 소스해시53기록일치, bestcheckpoint120개SHA일치, 9,600epoch행재검사. {artifact('protocol120.csv')}, {artifact('audit_status.json')}, {artifact('manifest_checks.json')}.

| 항목 | 실제 확인 |
|---|---|
| 조건 | 4ratios×2leads×3seeds×5models=120 |
| 1epoch | 유한DataLoader의선택window정확히1회; drop_last=False |
| Train n | 10%376,25%987,50%1974,100%3995 |
| batch16, 80epochs | 각각24/62/124/250updates/epoch;1920/4960/9920/20000total |
| Validation | 매epoch, data-onlyRMSE, 모두80회 |
| selection | epoch1–80의최저ValRMSE;동점은먼저저장한epoch |
| best range | {pr.best_epoch.min()}–{pr.best_epoch.max()}, 중앙값{pr.best_epoch.median():g} |
| earlystop | OFF, loop range(1,81);patience81이나actualworker에서는earlystopbreak없음 |
| optimizer | AdamW,betas(.9,.95),wd.1,clipnorm1 |
| LR | initial2e−5,epoch1=5.6e−5,epoch5=2e−4,cosineepoch80=0 |
| physics weight | P0/NoAdv/Adv .1min(epoch/5,1);Aux .0001constant;Baseline0 |
| pairing | Train/Val/Test인덱스·manifest·stats·initialsharedSHA·sampleorder80회·LR80회일치 |

마지막epochLR=0이므로optimizer.step호출은기록되지만그epoch가파라미터학습을추가하는것은아니다. 모든군에공통인기존스케줄이며조기종료가아니다. '80번유효한nonzero-LRepoch'라고기술하지않는다. {link(native,230)}, {link(worker,42)}, {link(worker,69)}, {link(worker,111)}, {link(aux,98)}.

Fullmean/std나FullRHS를덮어쓴흔적없음.120config통계가각manifest와동일하다. v3는v2의input/target통계·climatology·indices를보존하고RHS만보정했다. v2climatology는선택history날짜만,366month/day윤년정렬bin,빈bin은같은subset의주기적선형보간;Fullfallback없음. 10%고유입력608일. input은climatologyfit와같은고유날짜, target통계는선택target날짜. 코드: {link('src/mhw/subset_climatology.py',19)}, {link('src/mhw/subset_climatology.py',26)}, {link('src/mhw/subset_climatology.py',95)}, {link(sub,59)}. 전기간raw를메모리에읽는것과전기간통계fit는구분한다.

선택Traintarget와forcing끝이첫Validationtarget보다이른지모든manifest에서검사. lead7Valtarget2014-01-11–2018-07-06,Test2018-07-07–2022-12-31;lead14Val2014-01-13–2018-07-07,Test2018-07-08–2022-12-31. slidinghistory가분할경계를걸칠수있으므로모든입력관측날짜가분할별완전분리라는주장은하지않는다. {link('scripts/oracle96/train_mixed_budget_gpu.py',16)}, {link('scripts/oracle96/train_mixed_budget_gpu.py',83)}.

100%는공통7639후보전체가아닌90일완전블록내44일window3995개. 10⊂25⊂50⊂100은6개seed/lead조합모두검사했고Val/Testindices도보존됨.

| 비교항목 | 기존WSDM/Full main 기록 | 해안보정30회/초기v3 | Oracle120 |
|---|---|---|---|
| 구조 | ConvLSTM및SeaCast-inspired모델군 | SeaCast-inspired;baselinehead정책시기별상이 | 모든군동일all-headJejuHiLAM |
| 물리 | P0–P3;Fullmain선정P0 | P0daily/P1daily/P0interval | P0/NoAdv/Adv/Aux별도 |
| seed/lead | main42/7/123,lead1/3/7/14 | 주로lead7 pilot | 42/7/123,lead7/14 |
| train | originalchronological70% | subsetwindows라도Full통계였던pilot;v3는subset통계 | selected3995max,subset-only |
| anomaly | 병합파일사전계산 | Full→subset-only변경 | v2month/day기후값보존 |
| epochs | 기존main저장행12–40등,earlystop서로상이 | 478update가상block,patience12 | 80actual,earlystopOFF |
| λrres | mainPINN10 | pilot주요P0rres0 | Aux만.0001 |
| 검증상태 | P0main일부metric확보;정확한WSDMP1–P3실행mapping미확보 | 보정30회별도checkpoint감사존재 | 120실행해시/config/로그직접확인 |

구형근거: {link('outputs/main_experiment_final/experiment_manifest_final.json')}, {link('src/run_main_experiment_final.ps1')}, {artifact('legacy_inventory.json')}, {link('outputs/historical_coastal_audit_20261008/report.md')}. oldFullP0 lead7seed123의S=9.7367229778와보정pilotFullS=.089963214는같은규모가아니다. 전자는해안오염legacy규모와일치하나그숫자만으로WSDMP1–P3실행소스까지확정하지않는다. 현재코드로구형실험을소급인증하지않는다. 기존원고의완전한실행mapping확보전동일통제실험처럼합치면안된다.
''')
write('mhw_evaluation_audit.md','SST·MHW 평가 정의와 기준 재검산',f'''
**PASS WITH LIMITATIONS. 기존정의를그대로검산했으며threshold를수정하지않았다.**

{artifact('mhw_reference_check.json')}: reference7670일,1993-01-01–2013-12-31(raw전체10957일의앞round(.7N)). threshold는각DOY±5일의해양SST90thpercentile,clim은동일모집단평균. 고정reference.npz를원자료로재계산하여clim/threshold모두NaN동일성을포함해bitwise일치. 동일SHA `ebf00ed893ac839e1593f81a631c4a51cb2e772840be04c0a40787476c24402e`를모든군에적용한다.

입력용subsetclimatology와별개다. 평가reference에는선택되지않은Train날짜가있지만Validation/Test날짜는없고,이는모든모델에공통인평가label의기준이다. normalization이나forward에이를적용하지않는다. 따라서strictsubset-only**학습통계**조건과모순되지않지만'모든연구단계의정보가subset만'이라는표현은피한다.

코드 {func('src/mhw/mhw_skill.py','def compute_climatology_threshold(')}, {func('src/mhw/mhw_skill.py','def day_of_year(')}, {func('src/mhw/mhw_skill.py','def enforce_min_duration(')}, {func('src/mhw/mhw_skill.py','def compute_mhw_skill_metrics(')}, {func('scripts/oracle96_test/metrics.py','def score(')}.

| 검증항목 | 실제 정의/결과 |
|---|---|
| threshold비교 | SST > threshold,동률은비사건 |
| 지속 | 각격자연속5일이상;4일/5일synthetictestPASS |
| gap/smoothing | 짧은gapmerge없음,31일thresholdsmoothing없음 |
| leapday | pandasdayofyear1..366그대로;2019-03-01=60,2020-02-29=60,2020-03-01=61 |
| inputclimatology와차이 | input은2000년month/day정렬;MHW는rawDOY,윤년후1일계절위상차가능 |
| mask | 동일661ocean;coast141,offshore520 |
| TP/FP/FN | 전체고정lead연속targetseries의격자×날짜합산 |
| POD | TP/(TP+FN) |
| FAR | FP/(TP+FP),false-positive-rate(FP/(FP+TN))와다름 |
| CSI | TP/(TP+FP+FN) |
| Precision | TP/(TP+FP) |
| F1 | 2TP/(2TP+FP+FN) |
| 0분모 | 코드정의0;일반적인결측처리와구분 |
| Testdates | lead7 1639일,lead14 1638일,각series1일간격 |

SST는raw°C오차(차이는K),해양격자동일가중RMSE/MAE/Bias이며면적cos(lat)가중평가가아니다. 5일사건은고정lead의여러발행일예측을이어붙인사후판정으로,단일발행시점의미래5일경보능력과동일하지않다. Test경계밖의사건연장일은주어지지않아경계censoring이존재한다. domain-areaevent(.15) 및onsetmatching은grid-day지표와별개다.

rawprediction→재계산숫자,저장dailySSE/TP/FP/FN/labels동일성검사는 {artifact('reanalyze.py')}의score및truth비교에있다. 모델·ratio·seed간같은lead의관측label을정확히비교한다. 결과는 {artifact('analysis_status.json')}, {artifact('test_results120.csv')}.

이정의는Hobday-style로기술가능하나'표준Hobday전절차(윤년정렬·smoothing·gapmerge)를완전히구현'했다고하면부정확하다. 정의를바꾸는민감도분석은저장예측으로가능하지만이번에는기존평가를조용히대체하지않았다. 현재비교계산오류가확인된것은아니므로재학습사유가아니다.
''')
print('Wrote six technical reports')
