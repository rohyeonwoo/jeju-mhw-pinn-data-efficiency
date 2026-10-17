import os
from pathlib import Path
O=Path('/ceph_data/yeonwoo/MHW_subset/outputs/paper_final_verification_20261015')
os.environ['MPLCONFIGDIR']=str(O/'runtime/MPLCONFIGDIR')
exec((O/'write_reports.py').read_text())
D=pd.read_csv(O/'test_results120.csv');P=pd.read_csv(O/'paired_seed_results.csv');S=pd.read_csv(O/'paired_summary.csv');CI=pd.read_csv(O/'block_bootstrap.csv');V=pd.read_csv(O/'protocol120.csv');DG=pd.read_csv(O/'aux_diagnostics_rechecked.csv')
assert len(D)==120 and J(O/'analysis_status.json')['status']=='COMPLETE'
# Compact publication tables; identical seed order in every pairing.
models=['Baseline','Auxiliary','P0','NoAdv','Adv'];rows=[]
for (ratio,lead),g in D.groupby(['ratio','lead']):
 row=dict(Ratio=ratio,Lead=lead)
 for name in models:
  x=g[g.model==name].rmse;row[name]=f'{x.mean():.5f} ± {x.std(ddof=1):.5f}'
 rows.append(row)
rmsetable=pd.DataFrame(rows)
vpair=[];auxrows=[]
for (ratio,lead),g in V.groupby(['ratio','lead']):
 b=g[g.model=='Baseline'].set_index('seed');a=g[g.model=='Auxiliary'].set_index('seed');delta=a.val_rmse-b.val_rmse
 vpair.append(dict(ratio=ratio,lead=lead,Baseline=b.val_rmse.mean(),Auxiliary=a.val_rmse.mean(),delta_K=delta.mean(),wins=int((delta<0).sum())))
for _,s in S[(S.reference=='Baseline')&(S.candidate=='Auxiliary')].iterrows():
 q=CI[(CI.reference=='Baseline')&(CI.candidate=='Auxiliary')&(CI.ratio==s.ratio)&(CI.lead==s.lead)&(CI.metric=='rmse')&(CI.block_days==90)].iloc[0];g=D[(D.ratio==s.ratio)&(D.lead==s.lead)];b=g[g.model=='Baseline'];a=g[g.model=='Auxiliary'];auxrows.append(dict(ratio=s.ratio,lead=s.lead,Baseline_RMSE=b.rmse.mean(),Auxiliary_RMSE=a.rmse.mean(),delta_K=s.delta_rmse_mean,improvement_pct=s.mean_paired_improvement_pct,wins=s.candidate_wins,CI95=f'[{q.ci_low:.6f}, {q.ci_high:.6f}]',Baseline_CSI=b.mhw_grid_csi.mean(),Auxiliary_CSI=a.mhw_grid_csi.mean(),delta_POD=a.mhw_grid_pod.mean()-b.mhw_grid_pod.mean(),delta_FAR=a.mhw_grid_far.mean()-b.mhw_grid_far.mean()))
auxrows=pd.DataFrame(auxrows);nval=sum(x['wins'] for x in vpair);ntest=int(auxrows.wins.sum())
med=DG[DG.epoch.isin([1,5,10,20,80])].groupby('epoch')[['rres_rms_kday','auxiliary_loss','weighted_auxiliary_loss','data_loss','weighted_aux_data_loss_ratio','shared_aux_gradient_norm','shared_data_gradient_norm','weighted_gradient_ratio','rres_last_layer_parameter_norm']].median().reset_index();med['weighted_gradient_percent']=med.weighted_gradient_ratio*100
preserve=J(O/'protected_integrity.json');assert not preserve['mismatches'];scale=pd.read_csv(O/'rhs_scale_recomputed.csv');assert (scale.status=='PASS').all()
write('auxiliary_ablation_audit.md','Auxiliary-only 기여 및 고정 Test 평가',f'''
**PASS WITH LIMITATIONS. 학습 오류는 확인되지 않았으며 효과는 조건별로 작거나 불균일하다.**

Data+0.0001×Rres penalty, physics OFF, we-tie OFF, 동일 모델 구조/초기화/v3 RHS/80 actual epochs 유지. 비활성 physics 함수 호출은 24개×80epoch 모두0이고, Rres 호출은 각 epoch의 모든 training batch와 일치했다. {artifact('protocol120.csv')}; {func(aux,'class Monitor(')}, {func(native,'def compute_batch_losses(')}.

Rres=1e−4 tanh(head), U=mean[(86400Rres)²]/S. 마지막 affine weight/bias를0으로 두면 Rres=0, U=0, 공유 Predictor로 가는 U gradient도0이 될 수 있다. 이는0-target 정규화의 예상 최소해이지 학습 실패가 아니다. 실제 마지막층 norm이0이 되었다는 주장과 구분한다. 기존 사전검사 FAIL을 수학적 성질로 재분류한 기록을 유지했다: {link('outputs/v8_actual_epoch_auxiliary24_20261015/preflight_reclassification.json')}.

247개 별도 train-batch autograd 진단의 model state·기존 grad·CPU/CUDA/Python/NumPy/loader RNG·mode 보존 flag가 모두 참이다. optimizer update는0이며 fixed first16 selected Train samples를 사용했다. Epoch1은 초기화 직후가 아니라 1epoch 학습 후이다. raw auxiliary gradient와 λ를 곱한 gradient를 구분한다. {func(aux,'def diagnostic(')}, {artifact('aux_diagnostics_rechecked.csv')}.

{table(med,['epoch','rres_rms_kday','auxiliary_loss','weighted_auxiliary_loss','data_loss','weighted_aux_data_loss_ratio','shared_aux_gradient_norm','shared_data_gradient_norm','weighted_gradient_percent','rres_last_layer_parameter_norm'])}

기존 1.759%→0.199%→0.041%는 각각1.758776%,0.198507%,0.040710%로 재현됐다. Rres RMS 및 auxiliary gradient가24개 모두 감소한다. 최종 gradient가 정확히0이라는 증거는 없으며, 고정 진단 batch에서의 감소이지 모든 배치/시점에서 사라진다는 증명은 아니다.

**Validation 검산: {nval}/24개 seed 조건에서 개선.**

{table(pd.DataFrame(vpair))}

**이번 고정 Test 추론: 24/24 완료.** 기존 Validation-best의SHA/epoch/ValRMSE를 실행 전 protocol.json에 고정했으며, 재학습·reselection·optimizer step 없이 GPU5/6/7 각1개로만 추론했다. 원래 Oracle96과 같은 score()·고정 MHW threshold를 사용하고 기존96결과를 덮어쓰지 않았다. 모두 forcing 제거/원본/NaN에서 prediction bitwise 동일, state unchanged, 대응 Baseline의 target/mask/date 동일. 근거 {artifact('auxiliary_test/protocol.json')}, {artifact('aux_test.py')}, 각 `auxiliary_test/runs/*/forcing_invariance_test.json`.

Test 개선 **{ntest}/24**, Validation16/24와 혼동하지 않는다. Δ=Aux−Baseline, 음수가 RMSE 개선; 개선율은 seed별 비율의 평균. 아래 CI는 고정3seed에 조건부인90일블록 구간이다.

{table(auxrows)}

25%·Lead14의 평균 개선약2.47%,3/3seed는 실질적인 차이가 관찰된 조건이다. 반면10%·Lead7은0/3개선이며,50/100%의 절대변화는 대체로 작다.100%·Lead7의 매우 작은 감소가 조건부 시간CI에서0을 제외하더라도 큰 실용적 효과나 seed 모집단 재현성을 의미하지 않는다. 별도 모델/손실 변경으로 좋은 수치를 선택하지 않았다. 기존 Test를 반복 열람한 연구이므로 이번 평가는 누락된 ablation Test의 탐색적 보완이지 독립 확증이 아니다.
''')
# Full comparison table includes all six pairs, no selection by sign.
primary=[]
for _,s in S.iterrows():
 row=dict(ratio=s.ratio,lead=s.lead,comparison=s.candidate+' − '+s.reference,delta_RMSE_K=s.delta_rmse_mean,seed_SD=s.delta_rmse_sd,improvement_pct=s.mean_paired_improvement_pct,wins=int(s.candidate_wins))
 for metric,short in [('rmse','RMSE'),('mhw_grid_csi','CSI'),('mhw_grid_pod','POD'),('mhw_grid_far','FAR')]:
  q=CI[(CI.reference==s.reference)&(CI.candidate==s.candidate)&(CI.ratio==s.ratio)&(CI.lead==s.lead)&(CI.metric==metric)&(CI.block_days==90)].iloc[0];row[short+'_delta_CI95']=f'{q.estimate:.5f} [{q.ci_low:.5f}, {q.ci_high:.5f}]'
 primary.append(row)
primary=pd.DataFrame(primary);primary.to_csv(O/'primary_comparisons.csv',index=False)
spatial=[]
for (ratio,lead),g in D.groupby(['ratio','lead']):
 a=g[g.model=='NoAdv'].set_index('seed');b=g[g.model=='Adv'].set_index('seed');row=dict(ratio=ratio,lead=lead)
 for k in ['rmse','DJF_rmse','MAM_rmse','JJA_rmse','SON_rmse','coast_rmse','offshore_rmse','historical_SW_Korea_box_rmse','historical_SW_Korea_JJA_rmse','mhw_grid_hits','mhw_grid_misses','mhw_grid_false_alarms']:row['NoAdv_minus_Adv_'+k]=float((a[k]-b[k]).mean())
 spatial.append(row)
spatial=pd.DataFrame(spatial);spatial.to_csv(O/'noadv_adv_spatial_seasonal.csv',index=False)
phystable=D.groupby('model')[['A_residual_rms_kday','B_residual_rms_kday','A_cumulative_mean_k','B_cumulative_mean_k','A_daily_rmse_correlation','B_daily_rmse_correlation']].mean().reset_index();noadvwins=J(O/'analysis_status.json')['NoAdv_vs_Adv_seed_wins'];pair=P[(P.reference=='NoAdv')&(P.candidate=='Adv')];csiwins=int((pair.delta_mhw_grid_csi<0).sum())
ciN=CI[(CI.reference=='NoAdv')&(CI.metric=='rmse')&(CI.block_days==90)];nexclude=int((ciN.ci_low>0).sum())
write('statistical_comparison.md','120개 고정 Test 예측 재분석과 통계',f'''
**PASS WITH LIMITATIONS.120개 모두 저장 예측에서 SST 및 MHW를 재계산하여 기존/신규 저장 지표와 일치했다.** 같은 lead의 target/date/mask/관측MHW labels가 모든ratio/seed/model에서 같다. NoAdv의 Adv 대비 SST 우위 **{noadvwins}/24**, CSI 우위 **{csiwins}/24**가 원본예측에서 재현됐다. {artifact('analysis_status.json')}, {artifact('test_results120.csv')}.

**SST RMSE(K), seed 평균±표본SD(ddof=1)**

{table(rmsetable)}

**통계 정의**: 각 seed에서 model pair를 같은 날짜로 대응시킨다. 날짜 단위로 전체661격자의SSE와TP/FP/FN을 합친 뒤 circular temporal block bootstrap을 수행했다. 주90일, 민감도30/180일, 각2000회, RNGseed=20261015+lead+block. 한 replicate는 원래기간길이만큼 block을 연결해 선택한다. 같은 날짜 index를 두모델·모든ratio·고정된3seed에 함께 사용한다. 각seed의재표본RMSE/CSI/POD/FAR를계산한뒤3seed평균차이의2.5/97.5percentile. 격자와날짜를독립관측으로취급하지않으며 seed를재표본하지않았다. MHW labels는원래연속series에서5일지속판정후 고정해 재표집하므로 block경계에서사건을인위적으로새로만들지않는다.

구간은 고정된학습모델3개와해당Test기간에조건부인시간적불확실성이다. 독립seed모집단의CI가아니며,24조건을독립24반복으로가정하지않는다. 다중비교보정을하지않은탐색적CI이고 일부유리한조건만골라확증주장할수없다.0포함은동등성입증이아니다.90일은기존연구의분석규칙을유지한것으로최적blocklength를추정한것은아니다.

모든6쌍×8조건의수치(Δ=candidate−reference; RMSE/FAR음수,CSI/POD양수가통상개선):

{table(primary[['ratio','lead','comparison','improvement_pct','wins','RMSE_delta_CI95','CSI_delta_CI95','POD_delta_CI95','FAR_delta_CI95']])}

전체144개seed쌍(6×24)의RMSE차이/개선율/TPFPFN차이는 {artifact('paired_seed_results.csv')}; meanSD/승리seed수는 {artifact('paired_summary.csv')}; 모든30/90/180일 구간576행은 {artifact('block_bootstrap.csv')}. 두모델의평균RMSE비율과seed별개선율평균은다르므로섞지않았다.

**NoAdv–Adv 공간·계절 진단**

아래Δ는위비교표와달리NoAdv−Adv이며음수면NoAdv RMSE가낮다. 전체·coast/offshore·DJF/MAM/JJA/SON은동일raw예측에서계산했다. 전체계절을합친NoAdv평균RMSE는8/8조건에서낮지만,90일조건부CI가0을제외한Adv−NoAdv는{nexclude}/8이다.

{table(spatial[['ratio','lead','NoAdv_minus_Adv_rmse','NoAdv_minus_Adv_DJF_rmse','NoAdv_minus_Adv_MAM_rmse','NoAdv_minus_Adv_JJA_rmse','NoAdv_minus_Adv_SON_rmse','NoAdv_minus_Adv_coast_rmse','NoAdv_minus_Adv_offshore_rmse']])}

지역 추가분석은 과거스크립트에정의된33.5–35°N,125.5–127.5°E box를그대로계산했다: {link('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/diagnose_sw_summer.py',43)}. 이box는코드주석도SW-Korea라고하며제주도의남서해역과동일한영역으로단정할수없다. 그래서결과열을`historical_SW_Korea_box`로명시했다. **별도로명확히정의된'제주남서부'ROI결론은UNVERIFIED**이고,Test결과를보고영역을재선택하지않았다. 지역box·JJA및TPFPFN의전체표는 {artifact('noadv_adv_spatial_seasonal.csv')}.

**동일수식으로재계산한 물리잔차와예측오차**

각모델의24조건평균을아래기술한다. A=관측MLD연행+투과, B=A+이류인공통진단식. 모든model에둘다적용하며학습head는쓰지않는다. 누적잔차평균은K,일별잔차RMS는K/day. 마지막두열은각실행내의날짜별해양RMSE와해양residualRMS Pearson상관의조건평균이며인과/독립표본검정이아니다.

{table(phystable)}

Adv는공통B잔차를더작게만들수있지만NoAdv가예측RMSE에서대부분낫다. 즉이자료/해상도/대리온도/손실가중치에서제약식에더잘맞는것과NOAASST예측을잘하는것은구분된다. 평균상관이작거나음수인것도잔차감소가예측오차감소를보장하지않음을보여줄뿐,물리법칙자체가틀렸다는증거는아니다. 현재잔차의T는예측값이므로양의누적평균을그대로관측가열부족의정량추정으로읽지않는다. 이전관측CPU실험의투과on/off부호반전과함께가열과보정가능성을한계로기록하되이번에Kd를재조정하지않았다.

분석코드 {artifact('reanalyze.py')}; 모든원본fields의SHA를test_results120.csv에기록했다. 모델별40개meanSD행과모든지표는 {artifact('comparison_summary.csv')}.
''')
# Status matrix and actionable retraining decision.
matrix=pd.DataFrame([
 ['수평 이류항 수식','PASS','−86400(uTx+vTy), 선형장/속도반전 통과','현재 Adv 이류 구현 사용 가능'],
 ['단위 및 부호','PASS WITH LIMITATIONS','미터·cos(lat)·86400 확인; upstream 원자료 처리 일부 미확인','K/day 서술 가능; 저장 방향 attrs 오류와 원자료 한계 명시'],
 ['해안 경계 계산','PASS WITH LIMITATIONS','88/88; 661 ocean/660 support; 비균일 편측 한계','실제 균일 격자에서 유효; 일반 불균일 격자로 확대 금지'],
 ['물리 RHS Scale','PASS','24/24 재계산 일치; subset P0 mean-square','공통 RMS normalization으로 정확히 기술'],
 ['모델별 손실 구성','PASS','전체 구조 동일; Aux만 Rres; NoAdv/Adv 이류만 차이','Adv를 physics+aux Full PINN으로 부르지 않음'],
 ['120개 학습 프로토콜','PASS','120/120; 9600epoch; 동일 pairing; best SHA','100%=3995, actual epoch, 마지막 LR0 명시'],
 ['SST 평가','PASS','120개 raw 예측 재검산 및 dates/targets 일치','고정 Validation-best의 탐색적 Test 비교'],
 ['MHW 평가','PASS WITH LIMITATIONS','동일고정threshold 재생성일치; rawDOY·5일','윤년/경계/고정lead사후사건 정의 한계'],
 ['Auxiliary 기여','PASS WITH LIMITATIONS',f'Val{nval}/24,Test{ntest}/24; gradient 점차약화','보편적 큰효과 주장불가;25%·L14조건 개선'],
 ['NoAdv–Adv 비교','PASS WITH LIMITATIONS',f'RMSE{noadvwins}/24,CSI{csiwins}/24;시간blockCI','해당조건에서 이류제외 예측우위;물리법칙 부정아님'],
 ['기존 WSDM과의 정합성','UNVERIFIED','P0–P3 매핑있으나 정확한실행source/checkpoint미확보','현재120과별도실험군;기존물리감소율확정금지']
],columns=['검증 항목','판정','근거','논문에 미치는 영향'])
matrix.to_csv(O/'verification_matrix.csv',index=False)
write('retraining_decision.md','재계산·재학습 필요 여부',f'''
**현재 Oracle120: PASS WITH LIMITATIONS — 확인된 재학습 대상0개.**

물리계산/전처리/optimizer/학습step/selection에영향을준새오류를확인하지못했다. 120실행프로토콜,53소스해시기록,120best체크포인트SHA,24subsetscale재계산,88stencil테스트가통과했다. 보호654파일해시도일치한다. {artifact('audit_status.json')}, {artifact('protected_integrity.json')}, {artifact('rhs_scale_recomputed.csv')}.

| 대상 | 판정 | 조치 |
|---|---|---|
| 현재Baseline/P0/NoAdv/Adv/Aux 전체120 | PASS WITH LIMITATIONS | 기존학습유지;이번고정Test및통계보고사용 |
| 기존96Test SST/MHW | PASS | 96/96숫자재현;교체/재학습불필요 |
| Auxiliary24 누락Test | PASS | 이번24개고정추론으로보완완료 |
| 서로다른식의native common_res를동일식처럼비교한주장 | FAIL — RECOMPUTE ONLY (그주장에한함) | 이번공통A/B진단으로대체해서기술;기존수치조용히덮어쓰지않음 |
| WSDM P1–P3 실제실행 및 원고특정감소율 | UNVERIFIED | 해당runmanifest/source/checkpoint와원고표매핑확보전보류 |
| 초기v2/legacy gradient | 과거보정필요경로,현재120과별개 | 현재v3는이미보정;legacy범위는실제실행증거로정해야함 |
| 비균일격자편측미분 일반화 | 현재실험외한계 | 현재균일0.25°재학습사유없음;격자변경시에새검증필요 |
| MHW윤년정렬/정의변경 | 선택적후처리민감도,현재비교오류아님 | 필요시새평가버전으로고정예측재계산;현재기준보존 |
| 명확한제주남서부ROI | UNVERIFIED | 역사적SW-Korea box와혼동금지;결과를보고영역선택하지않음 |

현재'FAIL—RETRAIN REQUIRED'로특정할run은없다. 이것은과거WSDM모든실험의무오류증명이아니다. 예를들어실행시점P2/P3가mask없는Laplacian을사용했거나legacyP0가육지0을미분한것으로확정된다면해당physics학습군의재학습이필요할수있다. 현재정확한P1–P3실행artifact연결이부족하므로추측으로모델·ratio·lead·seed를지정하지않는다. 어떤재학습도시작하지않았다.

최종사용범위:120개모델의SST/MHW성과와공통식잔차진단은한계와함께논문에사용할수있다. 'PINN은언제나저자료량에서우수', '전체정보량동일', 'Adv가완전한실측혼합층수온방정식', '낮은잔차가더정확한예측을보장', 'WSDM P0–P3와동일통제실험'은지지되지않는다.
''')
write('verification_summary_ko.md','제주 연안 PINN 최종 정합성 검증',f'''
**현재 Oracle120은 PASS WITH LIMITATIONS입니다. 확인된 재학습 필요 모델은0개입니다. 기존 WSDM P1–P3의 실행 시점 정합성은 별도로 UNVERIFIED입니다.**

이번 작업은 신규 학습/튜닝 없이 수행했습니다. CPU 수치·로그·예측 검산과 저장된 Auxiliary-only24개 모델의 Test 추론만 실행했습니다. GPU5/6/7에 각1개씩 사용했고 모든GPU PID의 process name `yeonwoo`를 확인했습니다. 원본 코드·데이터·통계·체크포인트·기존 결과를 덮어쓰지 않았으며 보호 대상654개 SHA가 유지됐습니다.

주요 검증 결과:

- **88/88 이류·해안 수치 테스트 PASS.** 실제격자는0.25°균일, 미터·cos(latitude)·K/day변환이일관됩니다. 육지값 변경 영향0, 과거보정stencil과동일. 위도속성의`decreasing`문구는실제증가좌표와불일치하지만 계산은실제좌표값을사용합니다.
- **120/120 학습 프로토콜 PASS.** 80actualepoch,동일subset/통계/초기Predictor/shuffle/LR,Validation-best확인. 실행소스해시53기록및bestcheckpoint120개해시일치.
- **24/24 RHS scale 재계산 PASS.** subset P0 RHS mean-square이며모든arm이같은sqrt(S)로정규화합니다. 'std'나'모델별고유RHSscale'로기술하면안됩니다.
- **120/120 Test SST·MHW 재검산 PASS.** 고정reference는1993–2013이며threshold/clim을원자료로재생성해비트단위일치. 모델/ratio/seed간동일truth.
- **Auxiliary Test24개 완료:** Baseline보다RMSE가낮은조건{ntest}/24;Validation은{nval}/24.25%·Lead14는평균약2.47%개선,3/3seed. 전체적으로큰효과가일관되게나타나지는않습니다.
- **NoAdv 대 Adv:** RMSE{noadvwins}/24,CSI{csiwins}/24에서NoAdv우위.8개ratio/lead평균RMSE모두NoAdv가낮지만90일시간CI에서0을제외하는것은{nexclude}/8입니다. Adv의공통이류포함잔차가더작다는결과와예측우위는별개입니다.

{table(matrix)}

**모델별 Test RMSE(K),3seed평균±SD**

{table(rmsetable)}

논문에 사용할 수 있는 해석: **동일한 SST label과학습프로토콜에서,이자료·해상도·대리수온·가중치조건의NoAdv는Adv보다대체로좋은예측성능을보였습니다. 이는이류물리법칙이틀렸다는검증이아닙니다.** PINN은Training-only oracleforcing을추가감독으로쓰므로SST-label efficiency실험이며total-information efficiency가아닙니다. Auxiliary의감소하는gradient와작은효과도ablation결과로그대로보고해야합니다.

필수 한계: NOAA SST≠GLORYSML평균,전체순단파단일지수근사와Kd=.06의미보정성,CPU투과on/off잔차부호반전에따른가열과보정가능성,원격자처리provenance부족,rawDOY MHW윤년차이,고정lead예측series의사후사건판정,고정3seed조건부CI와다중비교/Test반복열람. 정확한'제주남서부'ROI는확정하지못했으며기존SW-Korea box분석과구분합니다.

**산출물**

- 모델/물리: {artifact('model_definition_audit.md')}, {artifact('physics_equation_audit.md')}
- 이류/해안: {artifact('horizontal_advection_audit.md')}, {artifact('coastal_stencil_tests.md')}
- 프로토콜/평가: {artifact('experiment_protocol_audit.md')}, {artifact('mhw_evaluation_audit.md')}
- Auxiliary/통계: {artifact('auxiliary_ablation_audit.md')}, {artifact('statistical_comparison.md')}
- 최종조치: {artifact('retraining_decision.md')}
- 전체120수치: {artifact('test_results120.csv')}, {artifact('comparison_summary.csv')}
- 전체대응비교: {artifact('paired_seed_results.csv')}, {artifact('block_bootstrap.csv')}

과거모델별native residual을같은공통식의잔차로비교한주장은그대로사용할수없습니다. 이번보고서는모든120개예측에공통A/B식각각을적용한새진단을따로제공합니다. 특정WSDM원고표의감소율은원고/runmapping확보전확정하지않습니다.
''')
# Simple exportable plots, values also provided in CSV.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(left,right) in zip(axes,[('Baseline','Auxiliary'),('NoAdv','Adv')]):
 q=CI[(CI.reference==left)&(CI.candidate==right)&(CI.metric=='rmse')&(CI.block_days==90)].sort_values(['lead','ratio']);y=np.arange(len(q));ax.errorbar(q.estimate,y,xerr=np.array([q.estimate-q.ci_low,q.ci_high-q.estimate]),fmt='o',capsize=3);ax.axvline(0,color='gray',ls='--');ax.set_yticks(y,[f'{r.ratio}% / L{r.lead}' for r in q.itertuples()]);ax.set_title(right+' minus '+left);ax.set_xlabel('Test RMSE difference (K), paired 90-day block 95% CI');ax.invert_yaxis();ax.grid(alpha=.2)
fig.suptitle('Fixed 3 seeds; exploratory temporal uncertainty; 2,000 resamples');fig.tight_layout();fig.savefig(O/'paired_rmse_intervals.png',dpi=180);fig.savefig(O/'paired_rmse_intervals.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for name,g in DG.groupby('name'):
 g=g[g.epoch.isin([1,5,10,20,80])].sort_values('epoch');ax.plot(g.epoch,g.weighted_gradient_ratio*100,alpha=.25,color='steelblue')
ax.plot(med.epoch,med.weighted_gradient_percent,'o-',color='black',label='Across-run median');ax.set_yscale('log');ax.set_xlabel('Actual epoch (epoch 1 is after training)');ax.set_ylabel('Weighted auxiliary / data gradient (%)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(O/'auxiliary_gradient_decay.png',dpi=180);fig.savefig(O/'auxiliary_gradient_decay.pdf');plt.close(fig)
print('All ten requested reports and exported figures written')
