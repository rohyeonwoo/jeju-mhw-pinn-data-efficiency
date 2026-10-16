"""Finish narrative interpretation and explicit NoAdv-positive SST/CSI table from saved analysis CSVs."""
import os
from pathlib import Path
O=Path('/ceph_data/yeonwoo/MHW_subset/outputs/paper_analysis_20261016')
os.environ['TMPDIR']=str(O/'runtime');os.environ['CUDA_VISIBLE_DEVICES']=''
import pandas as pd,numpy as np,json,hashlib,datetime
from zoneinfo import ZoneInfo
p=pd.read_csv(O/'statistics/paired_seed_results.csv');b=pd.read_csv(O/'statistics/existing_bootstrap90.csv');g=p[(p.reference=='NoAdv')&(p.candidate=='Adv')];rows=[]
for (ratio,lead),x in g.groupby(['ratio','lead']):
 c=b[(b.ratio==ratio)&(b.lead==lead)&(b.reference=='NoAdv')&(b.candidate=='Adv')];r=c[c.metric=='rmse'].iloc[0];z=c[c.metric=='mhw_grid_csi'].iloc[0]
 assert abs(r.estimate-x.delta_rmse.mean())<1e-12
 assert abs(z.estimate+x.noadv_minus_adv_csi.mean())<1e-12
 rows.append(dict(ratio=ratio,lead=lead,delta_rmse_adv_minus_noadv_mean=x.delta_rmse.mean(),delta_rmse_sd=x.delta_rmse.std(ddof=1),rmse_ci_low=r.ci_low,rmse_ci_high=r.ci_high,noadv_rmse_wins=int((x.delta_rmse>0).sum()),delta_csi_noadv_minus_adv_mean=x.noadv_minus_adv_csi.mean(),delta_csi_sd=x.noadv_minus_adv_csi.std(ddof=1),csi_ci_low=-z.ci_high,csi_ci_high=-z.ci_low,noadv_csi_wins=int((x.noadv_minus_adv_csi>0).sum()),joint_noadv_rmse_csi_wins=int(((x.delta_rmse>0)&(x.noadv_minus_adv_csi>0)).sum()),n_seeds=3))
d=pd.DataFrame(rows);d.to_csv(O/'tables/tableS_noadv_adv.csv',index=False)
# Compact publication table; full-precision underlying CSV is retained.
compact=pd.DataFrame({'Ratio (%)':d.ratio,'Lead (d)':d.lead,'ΔRMSE (K)':d.apply(lambda r:f'{r.delta_rmse_adv_minus_noadv_mean:.4f} ± {r.delta_rmse_sd:.4f}',axis=1),'RMSE 95% CI':d.apply(lambda r:f'[{r.rmse_ci_low:.4f}, {r.rmse_ci_high:.4f}]',axis=1),'RMSE wins':d.noadv_rmse_wins.astype(str)+'/3','ΔCSI':d.apply(lambda r:f'{r.delta_csi_noadv_minus_adv_mean:.4f} ± {r.delta_csi_sd:.4f}',axis=1),'CSI 95% CI':d.apply(lambda r:f'[{r.csi_ci_low:.4f}, {r.csi_ci_high:.4f}]',axis=1),'CSI wins':d.noadv_csi_wins.astype(str)+'/3'})
caption='NoAdv–Adv paired comparison. Positive ΔRMSE=RMSE(Adv)−RMSE(NoAdv) and ΔCSI=CSI(NoAdv)−CSI(Adv) favor NoAdv. Mean ± three-seed sample SD; existing conditional 90-day bootstrap intervals.'
(O/'tables/tableS_noadv_adv.tex').write_text(compact.to_latex(index=False,escape=True,caption=caption,label='tab:noadv_adv'),encoding='utf-8')
summary=O/'reports/analysis_summary_ko.md';text=summary.read_text()
text=text.replace('## 2. Auxiliary-only의 독립 효과\n','## 2. Auxiliary-only의 독립 효과\n\n25%·Lead14에서 seed별 개선율 평균 2.466%, 3/3 seed 개선이 관찰됐다. 반면 10%·Lead7은 0/3 개선이며 평균 RMSE는 악화됐다. 그 외 다수 조건의 변화는 작다. 고정 진단 batch의 gradient 비 중앙값은 epoch1 1.7588% → epoch20 0.1985% → epoch80 0.04071%로 감소했고, Rres RMS는 0.17484 → 0.05187 → 0.01888 K/day다. 감소 자체와 원인 해석을 구분한다.\n')
text=text.replace('## 3. NoAdv와 Adv 차이가 반복되는 조건\n','## 3. NoAdv와 Adv 차이가 반복되는 조건\n\n8개 Ratio×Lead 조건 모두 평균 RMSE는 NoAdv가 낮다. 차이는 0.01252–0.06156 K다. 기존 90일 조건부 신뢰구간은 6/8 조건에서 0을 제외하지만, 50%·Lead14와 100%·Lead14는 0을 포함한다. 이는 모든 조건에서 확증된 효과라는 뜻이 아니다. CSI까지 NoAdv 우위 방향을 양수로 통일한 별도 표는 `tables/tableS_noadv_adv.csv`다.\n')
text=text.replace('## 4. 자료 감소가 제약의 유용성을 증가시키는가?\n','## 4. 자료 감소가 제약의 유용성을 증가시키는가?\n\n보편적·단조적 증가로 결론낼 수 없다. NoAdv는 Lead7의 10/25/50%에서 평균 seed 개선율이 양수지만 100%는 음수이며, Lead14는 50%에서 악화되는 등 비단조적이다. 작은 자료량에서 유리한 일부 조건을 물리 제약 전체의 자료 효율 향상으로 일반화하지 않는다.\n')
joint=int(((g.delta_rmse>0)&(g.noadv_minus_adv_csi>0)).sum())
text=text.replace('## 5. SST 차이와 MHW 차이의 연결\n',f'## 5. SST 차이와 MHW 차이의 연결\n\nNoAdv는 Adv보다 RMSE가 낮은 22/24 조건 중 {joint}개에서 CSI도 높다. 두 지표가 함께 좋아지는 경향은 있지만 항상 일치하지 않으며, POD/FAR trade-off와 함께 해석한다. 아래 일반 비교표의 candidate가 Adv일 때는 Adv 우위 횟수이므로 NoAdv 우위 횟수와 혼동하지 않는다.\n')
text=text.replace('## 6. 공간·계절 차이\n','## 6. 공간·계절 차이\n\nNoAdv−Adv 차이의 반대부호인 Adv−NoAdv RMSE를 보면, 겨울의 조건평균 차이는 Lead7 약 0.0779 K, Lead14 약 0.0735 K이며 여름은 각각 0.0173 K, 0.0018 K다. 일부 여름 조건은 Adv가 더 낮은 RMSE를 보인다. 이 요약은 ratio×seed의 기술적 평균이며 계절 효과의 독립 통계검정이 아니다. 새 SW-Jeju 영역은 40개 해양 격자, coast 141개, offshore 520개다.\n')
summary.write_text(text)
manifest=pd.read_csv(O/'figure_table_manifest.csv');row=dict(id='tableS_noadv_adv',kind='table',artifact='tables/tableS_noadv_adv.tex',plot_data='tables/tableS_noadv_adv.csv',caption=caption,models='NoAdv;Adv',sources='statistics/paired_seed_results.csv;statistics/existing_bootstrap90.csv',analysis_code='code/analyze.py;code/finalize_findings.py',analysis_conditions='4 ratios;2 leads;3 fixed seeds;exploratory Test',input_run_index='reports/input_completeness.csv',companion_data='statistics/paired_seed_results.csv;statistics/existing_bootstrap90.csv')
manifest=manifest[manifest.id!='tableS_noadv_adv'];manifest=pd.concat([manifest,pd.DataFrame([row])],ignore_index=True);manifest.to_csv(O/'figure_table_manifest.csv',index=False)
inv=json.loads((O/'reports/source_inventory.json').read_text());code=Path(__file__)
with code.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
inv[str(code)]={'size':code.stat().st_size,'mtime_ns':code.stat().st_mtime_ns,'sha256':h};(O/'reports/source_inventory.json').write_text(json.dumps(inv,indent=2))
check=json.loads((O/'reports/output_validation.json').read_text());check['tables']=int((manifest.kind=='table').sum());check['NoAdv_joint_rmse_csi_wins']=joint;check['NoAdv_CSI_interval_sign_conversion_checked']=True;(O/'reports/output_validation.json').write_text(json.dumps(check,indent=2))
readme=O/'README.md';rt=readme.read_text().replace('Table1–6 plus separate Validation table.','Table1–6 plus separate Validation and explicit NoAdv-positive comparison tables.');rt+='\nFinal numerical interpretation and explicit NoAdv-positive CSI confidence intervals: `code/finalize_findings.py` (run after the two scripts above).\n';readme.write_text(rt)
now=datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
with (O.parents[1]/'docs/research_log.md').open('a') as f:f.write(f'\n- {now} — 최종 해석 보강 및 tableS_noadv_adv 추가: SST/CSI 모두 NoAdv 우위를 양수로 통일, 기존 CSI CI 부호 반전/끝점 재정렬 검산. 7개 연구질문에 실제 수치와 비단조성/작은효과/계절 한계 명시. 기존 산출물 변경 없음; 새 분석 디렉터리 내 보고서·manifest 갱신.\n')
print('FINDINGS COMPLETE',check,flush=True)
