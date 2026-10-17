from eval_common import *
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=evaluation_protocol();verify_evaluation(p);rows=[]
 for j in p['jobs']:
  d=EVAL/'runs'/j['name'];f=d/'status.json'
  if not f.exists() or json.loads(f.read_text())['status']!='COMPLETED':continue
  m=json.loads((d/'metrics.json').read_text());assert m['checkpoint_sha256']==j['checkpoint_sha256'] and m['forcing_invariance']=='PASS';rows.append(dict(name=j['name'],**m))
 df=pd.DataFrame(rows);assert len(df);df.to_csv(EVAL/'results_by_seed.csv',index=False)
 keys=['rmse','mae','bias','acc','validation_rmse','generalization_gap_rmse','mhw_grid_precision','mhw_grid_pod','mhw_grid_csi','mhw_grid_f1','mhw_day_rmse','coast_rmse','offshore_rmse','DJF_rmse','MAM_rmse','JJA_rmse','SON_rmse']
 keys+= [x for x in df if x.startswith('mhw_event_') and pd.api.types.is_numeric_dtype(df[x])]
 summary=df.groupby(['ratio','lead','model_label'])[keys].agg(['mean','std','count']);summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(EVAL/'comparison_summary.csv',index=False)
 pairs=[]
 for (ratio,lead,seed),g in df.groupby(['ratio','lead','seed']):
  b=g[g.model_label=='Baseline']
  if len(b)!=1:continue
  b=b.iloc[0]
  for _,r in g[g.model_label!='Baseline'].iterrows():
   pairs.append(dict(ratio=ratio,lead=lead,seed=seed,model=r.model_label,test_rmse_improvement_pct=100*(b.rmse-r.rmse)/b.rmse,validation_rmse_improvement_pct=100*(b.validation_rmse-r.validation_rmse)/b.validation_rmse,csi_difference=r.mhw_grid_csi-b.mhw_grid_csi,f1_difference=r.mhw_grid_f1-b.mhw_grid_f1,precision_difference=r.mhw_grid_precision-b.mhw_grid_precision,pod_difference=r.mhw_grid_pod-b.mhw_grid_pod))
 paired=pd.DataFrame(pairs);paired.to_csv(EVAL/'paired_vs_baseline.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for col,lead in enumerate([7,14]):
  for row,key in enumerate(['rmse','mhw_grid_csi']):
   ax=axes[row,col]
   for label in ['Baseline','P0','NoAdv','Adv']:
    g=df[(df.lead==lead)&(df.model_label==label)].groupby('ratio')[key].agg(['mean','std']).sort_index()
    if len(g):ax.errorbar(g.index,g['mean'],yerr=g['std'].fillna(0),marker='o',capsize=3,label=label)
   ax.set(title=f'Lead {lead}: {key}',xlabel='Training subset (%)',ylabel=key);ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(EVAL/'test_overview.png',dpi=180);fig.savefig(EVAL/'test_overview.pdf');plt.close(fig)
 lines=[f'# Actual-epoch Oracle96 Test 평가 ({len(df)}/96 완료)',f'작성: {stamp()}','',
 '전체 Validation-best best.pt를 사전 SHA로 고정하여 평가했다. Test 기반 재선택·학습·통계 갱신은 없다. 미래 forcing을 제거/그대로 제공/NaN 교란한 전체 Test 예측은 bitwise 동일하며 모델 파라미터도 변하지 않았다.',
 'MHW는 기존 고정 DOY 90백분위 임계값과 5일 지속 조건을 유지한다. Precision/POD/CSI/F1은 해양 grid-day 기준이다. 별도 mhw_event_*는 해역 면적비 0.15 사건의 시작일·지속기간 지표이며 grid-day 지표와 혼동하지 않는다. 고정 lead의 연속 예측 날짜를 이용한 사후 지속기간 판정으로, 실시간 경보 검증과는 구별한다.',
 '과거 연구 과정에서 같은 Test 기간을 확인한 이력이 있으므로 탐색적 평가이다. Oracle PINN은 Train 미래 forcing이라는 추가 감독 정보를 사용했으므로 동일 총정보량 효율성을 증명하지 않는다. Seed 3개의 표본표준편차는 작은 표본의 기술통계이며 유의성을 자동으로 뜻하지 않는다. Validation–Test RMSE 차이는 기간 난이도/분포 차이도 포함한다.','',
 '|Ratio|Lead|Model|n|SST RMSE (K), mean±SD|MAE|Bias|MHW Precision|POD|CSI|F1|','|---|---|---|---|---|---|---|---|---|---|---|']
 for (ratio,lead,label),g in df.groupby(['ratio','lead','model_label']):
  sd=f'{g.rmse.std():.4f}' if len(g)>1 else 'NA'
  vals=[g[k].mean() for k in ['mae','bias','mhw_grid_precision','mhw_grid_pod','mhw_grid_csi','mhw_grid_f1']]
  lines.append(f'|{ratio}|{lead}|{label}|{len(g)}|{g.rmse.mean():.4f} ± {sd}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
 lines+=['','## Baseline 대비 seed 대응 비교','RMSE 개선율은 각 seed의 (Baseline−모델)/Baseline×100을 계산한 뒤 평균한다. 양수가 개선이며, CSI 차이도 양수가 개선이다.','|Ratio|Lead|Model|Test 개선율 평균 (%)|Validation 개선율 평균 (%)|Test 개선 seed|CSI 차이 평균|','|---|---|---|---|---|---|---|']
 if len(paired):
  for (ratio,lead,model),g in paired.groupby(['ratio','lead','model']):lines.append(f'|{ratio}|{lead}|{model}|{g.test_rmse_improvement_pct.mean():+.3f}|{g.validation_rmse_improvement_pct.mean():+.3f}|{(g.test_rmse_improvement_pct>0).sum()}/{len(g)}|{g.csi_difference.mean():+.4f}|')
 lines+=['','## 검증 근거','`evaluation_protocol.json`: 체크포인트/평가기준/코드 SHA 고정. `preflight.json`: 96 체크포인트와 CPU 지표 회귀 검사. `runs/*/forcing_invariance_test.json`: 전체 Test 예측 경로의 미래 forcing 비의존성. `runs/*/metrics.json`: 해안·외해·계절·사건 지표 및 GPU 사용량. `results_by_seed.csv`: seed별 전체 지표. `comparison_summary.csv`: 평균/표본표준편차. `paired_vs_baseline.csv`: seed 대응 개선율. `runs/*/fields.npz`: 예측/정답/날짜/마스크.','',f'학습 기간/Validation/Test 인덱스와 입력 통계는 원래 훈련 bundle을 그대로 사용했다. 평가 소스: scripts/oracle96_test/evaluate.py, metrics.py. 기존 src/mhw/mhw_skill.py의 사건 정의를 재사용했다.']
 (EVAL/'report_ko.md').write_text('\n'.join(lines)+'\n');print(f'Summarized {len(df)}/96',flush=True)
if __name__=='__main__':main()
