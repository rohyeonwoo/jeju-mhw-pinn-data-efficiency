# 물리 잔차·단위·정규화 감사

작성: 2026-10-15 13:59 KST. 신규 학습·재튜닝·원본 코드/데이터/기존 결과 변경 없음.

**현재120의 활성 물리 경로: PASS WITH LIMITATIONS. 모든 잔차는 clipping 전에 K/day이다.**

호출: [def run_training(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_actual/worker.py:72) → [def run_epoch(](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:302) → [def compute_batch_losses(](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:264) → [def daily(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/train_mixed_budget_gpu.py:100) → P0 native daily_rollout 또는 A/B mixed loss → [def compute_physics_loss(](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:447). `worker.main`의 install로 실제 dispatch가 등록된다. 명목상 함수 이름만 보고 현재 모듈을 추정하지 않았다.

현재시점t=start+29, leadL=7/14, T_j=T_t+(j/L)(T̂_(t+L)−T_t), j=0,…,L−1. δT=(T̂−T_t)/L는1일 증가량이며 K/day tendency로 비교한다. forcing은 `[t:t+L]` 즉t,…,t+L−1에 대응한다. P0는 `(T_(j+1)−T_j)/86400` K/s를 먼저 만들고 전체 residual에86400을 곱한다. A/B는 increment를직접K/day로 사용한다. lead에 따른 이중86400 또는 누락 없음.

H_j=max(mlotst_j,5m), D는 해양 이웃 미분, ΔT_d=1K, S는 선택 Train의 P0 RHS mean-square.

**P0 코드 수식**

r_j = (T_(j+1)−T_j) − 86400[ qnet_j/(1025·3990·H_j) − u_ml,j D_xT_j − v_ml,j D_yT_j − ŵ_e·1/H_j ].

ŵ_e = 1e−5 softplus(head_we−1) m/s. 직접 관측된 연행속도가 아닌 학습 latent output. 7/14일의 모든 daily step에 동일 forecast head field를 사용한다. Rres, vertical, diffusion 모두0. P0 함수 안 observed target tendency를 만들지만 현재 fixed RHS normalization에서는 loss에 쓰이지 않는다. [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:659), [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:803), [src/mhw/models/jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/mhw/models/jeju_hilam.py:340).

**NoAdv / Adv 코드 수식**

w_e,j = max(0, max(Hraw_j,5)−max(Hraw_(j−1),5))/86400 m/s.

F_j = 86400(qnet_j−ssr_j exp(−0.06 H_j))/(1025·3990·H_j).

E_j = −86400 w_e,j (T_j−Tb_j)/H_j.

A_j = −86400(u_j D_xT_j+v_j D_yT_j).

r_A,j = δT−(F_j+E_j), r_B,j = δT−(F_j+E_j+A_j).

Tb는H보다깊은첫유효thetao 중심층, u/v는midpoint-interface두께가중ML평균. P0의기존u_ml/v_ml와달리featurecache에서재계산. 미래NOAA정답은이식에입력되지않는다. 근거 [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:9), [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:25), [scripts/oracle96/mixed_budget_physics.py](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96/mixed_budget_physics.py:30), [scripts/prepare_mixed_budget_features.py](/ceph_data/yeonwoo/MHW_subset/scripts/prepare_mixed_budget_features.py:23).

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

qnet 전기간 해양 조합검산: 최대 차이 9.155273e-05 W/m², RMS 8.366766e-06 W/m² (float32합산 규모). qnet/ssr에W/m² attrs가 있고, ssr는누적에너지→평균flux변환기록이있다. 이를재차86400으로나누지않는다. 최초ERA5파일의시간평균처리 전체 provenance는별도한계이며, 이검사만으로 복원했다고하지않는다.

Clip은r_j의물리단위에서±10K/day, normalization및square이전이다. clipping이threshold밖의gradient를0으로만들수있으므로 rawRMS와clippednormalizedloss는구별한다. [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:455).

**RHS scale 24개 재계산**

선택startwindows, 관측current/target를선형보간, we=0인P0resolvedRHS를 사용하여pooledmean(RHS_Kday²) 계산. `S`라는변수는standard deviation/RMS가아니며 sqrt(S)가RMS. inputmean/std=고유선택history날짜×해양; target=선택target날짜×해양. scale은타깃을쓰지만 Train만이며모델독립. NoAdv/Adv별자체RHSscale은없고모든arm이같은P0 S를쓴다. 물리학적으로항별상대크기를동일화한것은아니다.

24개모두양수finite·재계산일치. 허용rtol2e−6, atol1e−9는FP32배치 reduction차이이며최대절대차이 3.3730399e-09. [outputs/paper_final_verification_20261015/rhs_scale_recomputed.csv](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/rhs_scale_recomputed.csv:1); [def compute_subset_rhs_scale(](/ceph_data/yeonwoo/MHW_subset/src/mhw/subset_preprocessing.py:119), [def compute_fixed_rhs_scale(](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:812), [def apply_manifest_physics(](/ceph_data/yeonwoo/MHW_subset/src/mhw/subset_preprocessing.py:187). manifest적용이legacyv1/v2/다른stencil/상충CLIscale을거부한다. Full통계덮어쓰기없음은120개config의실제통계일치로도검증했다.

| ratio | seed | lead | stored | recomputed | abs_error | status |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 123 | 14 | 0.09047598 | 0.09047598 | 2.116417e-10 | PASS |
| 100 | 42 | 14 | 0.09047598 | 0.09047598 | 2.116417e-10 | PASS |
| 100 | 7 | 14 | 0.09047598 | 0.09047598 | 2.116417e-10 | PASS |
| 100 | 123 | 7 | 0.09060376 | 0.09060376 | 1.21694e-09 | PASS |
| 100 | 42 | 7 | 0.09060376 | 0.09060376 | 1.21694e-09 | PASS |
| 100 | 7 | 7 | 0.09060376 | 0.09060376 | 1.21694e-09 | PASS |
| 10 | 123 | 14 | 0.07059743 | 0.07059743 | 3.37304e-09 | PASS |
| 10 | 42 | 14 | 0.08414582 | 0.08414582 | 1.124347e-09 | PASS |
| 10 | 7 | 14 | 0.09232043 | 0.09232043 | 5.621733e-10 | PASS |
| 10 | 123 | 7 | 0.06741457 | 0.06741457 | 0 | PASS |
| 10 | 42 | 7 | 0.08493099 | 0.08493098 | 3.37304e-09 | PASS |
| 10 | 7 | 7 | 0.09229022 | 0.09229022 | 0 | PASS |
| 25 | 123 | 14 | 0.07969812 | 0.07969812 | 8.56645e-10 | PASS |
| 25 | 42 | 14 | 0.08332505 | 0.08332505 | 1.499129e-09 | PASS |
| 25 | 7 | 14 | 0.08827559 | 0.08827559 | 0 | PASS |
| 25 | 123 | 7 | 0.07934067 | 0.07934067 | 4.283225e-10 | PASS |
| 25 | 42 | 7 | 0.08276181 | 0.08276181 | 0 | PASS |
| 25 | 7 | 7 | 0.08980056 | 0.08980056 | 1.71329e-09 | PASS |
| 50 | 123 | 14 | 0.0835984 | 0.0835984 | 4.283225e-10 | PASS |
| 50 | 42 | 14 | 0.08892953 | 0.08892953 | 1.820371e-09 | PASS |
| 50 | 7 | 14 | 0.0878206 | 0.0878206 | 4.283225e-10 | PASS |
| 50 | 123 | 7 | 0.08432857 | 0.08432857 | 1.927451e-09 | PASS |
| 50 | 42 | 7 | 0.0894254 | 0.0894254 | 6.424838e-10 | PASS |
| 50 | 7 | 7 | 0.08861295 | 0.08861294 | 9.637257e-10 | PASS |

동일실제batch/예측에서loss RHS와scale RHS구현대조: RHS최대차 2.3841858e-07K/day, residual최대차 1.5199184e-06K/day;float32허용2e−6내. Predictor field gradient norm 0.12311725,finite. [outputs/paper_final_verification_20261015/physics_numerical_checks.json](/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015/physics_numerical_checks.json:1).

**미해결항 및 예전 확장식**

현재Rres는Auxiliary-only의0-targetpenalty로만사용; 모든물리RHS에서제외. 원논문의 `+10L_rres`를현재P0/Adv설명에사용하면틀린다. 기존P1vertical은precomputedvadv_kday/86400을K/sRHS에더함;생성시positive-down수심에서`+w*dT/dz`를86400배하고±2K/day clip. P2diffusion은Kh laplacian(T),Kh기본100m²/s. 두경로모두현재120개에서비활성이다. [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:595), [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:615), [src/train_hybrid_jeju_forecast.py](/ceph_data/yeonwoo/MHW_subset/src/train_hybrid_jeju_forecast.py:627), [preprocess_glorys3d_budget_terms.py](/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py:125). 현재laplacian은ocean-neighborLaplacian이아니고경계2차분0이며, 이를현재Adv에사용된것처럼해석하지않는다.

**Common Residual 비교 가능 범위**

native로그의`common_res_*`는이름과달리각모델자체RHS/학습we를사용한다. Baseline비활성값이나Aux의진단생략0을물리잔차0으로해석하면안된다. NoAdv자체r_A와Adv자체r_B의감소율도동일식비교가아니다. [src/train_jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:180), [src/train_jeju_hilam.py](/ceph_data/yeonwoo/MHW_subset/src/train_jeju_hilam.py:363), [class Monitor(](/ceph_data/yeonwoo/MHW_subset/scripts/oracle96_auxiliary24/aux_worker.py:32).

이번재분석은모든120예측에공통A와공통B식각각을적용하고동일관측MLDwe/Tb/mask를사용했다. 따라서Baseline의존재하지않거나훈련되지않은head를물리관측처럼대입하지않는다. 과거공통평가는별도로zero-we/own-latent-we/shared-P0-we를구분했다: [final_clean_evaluate.py](/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/final_clean_evaluate.py:179). zero-we는연행제외진단, shared-P0-we는P0에의존한진단이며독립관측열수지아니다. 원고의정확한감소율·대상표가없어그수치자체는UNVERIFIED. 공통식의rawresidual과예측오차를같이제시하고'진정한물리정합성증명'이라고하지않는다.

**논문에 유지해야 할 한계**: NOAA SST는GLORYSML평균의대리변수(기존ValidationRMSE약1.185K); 전체순단파에대한단일지수근사,Kd=.06은pilot가정; 투과항후CPU잔차평균부호반전은가열과보정가능성이지입증이아님; missingvertical/diffusion/mixing/assimilation항; 관측미래forcing은Training-onlyoracle감독. SST-label efficiency이며total-information efficiency가아니다. inferenceforcing불변성은추가감독정보가없다는증명이아니다.

권장영문: “Daily physics residuals are expressed in K day⁻¹ (the P0 residual is first formed in K s⁻¹ and multiplied by 86,400), clipped to [−10, 10] K day⁻¹, and normalized by the fixed P0 RHS RMS computed from each selected training subset.”
