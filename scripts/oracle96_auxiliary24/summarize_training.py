"""Validation-only paired comparison; diagnostics describe regularization, not task success."""
from aux_common import *
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=plan();verify(p);done=[];epochs=[];diags=[];allmodels=[];audits=[]
 for j in protocol()['jobs']:
  s=json.loads((Path(j['run_dir'])/'status.json').read_text());assert s['status']=='COMPLETED'
  allmodels.append(dict(model=j['label'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],best_val_rmse=s['best_val_rmse'],best_epoch=s['best_epoch'],run_dir=j['run_dir']))
 for j in p['jobs']:
  d=Path(j['run_dir']);s=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {'status':'PENDING'}
  h=json.loads((d/'history.json').read_text()) if (d/'history.json').exists() else []
  ds=json.loads((d/'diagnostics.json').read_text()) if (d/'diagnostics.json').exists() else []
  for r in h:epochs.append(dict(name=j['name'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],**r))
  for r in ds:diags.append(dict(name=j['name'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],**r))
  if s['status']!='COMPLETED':continue
  assert len(h)==80 and s['steps']==j['total_updates'] and not s['early_stopping'] and s['test_evaluation']=='NOT_RUN'
  cfg=json.loads((d/'config.json').read_text());ref=json.loads((Path(j['baseline_run'])/'config.json').read_text());rh=json.loads((Path(j['baseline_run'])/'history.json').read_text())
  for k in ['shared_initial_sha256','stats','rhs_scale','split_sha256','tensor_fingerprints','parameters']:assert cfg[k]==ref[k]
  for x,y in zip(h,rh):assert x['sample_order_sha256']==y['sample_order_sha256'] and x['lr']==y['lr'] and x['physics_calls']==0 and x['rres_calls']==j['updates_per_epoch'] and x['sampling_exact_once']
  ca=json.loads((d/'checkpoint_audit.json').read_text());assert ca['status']=='PASS'
  assert all(x['model_state_unchanged'] and x['parameter_gradients_unchanged'] and x['RNG_unchanged'] for x in ds)
  audits.append(dict(name=j['name'],**ca))
  base=next(r for r in allmodels if r['model']=='Baseline' and all(r[k]==j[k] for k in ['ratio','lead','seed']))
  r=dict(name=j['name'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],baseline_rmse=base['best_val_rmse'],auxiliary_rmse=s['best_val_rmse'],improvement_pct=100*(base['best_val_rmse']-s['best_val_rmse'])/base['best_val_rmse'],best_epoch=s['best_epoch'],updates=s['steps'])
  for ep in [1,5,10,20,80]:
   x=next(x for x in ds if x['epoch']==ep)
   for k in ['rres_rms_kday','shared_aux_gradient_norm','shared_data_gradient_norm','weighted_gradient_ratio','weighted_aux_data_loss_ratio','rres_last_layer_parameter_norm']:r[f'{k}_epoch{ep}']=x[k]
  r['rres_rms_final_over_epoch1']=r['rres_rms_kday_epoch80']/r['rres_rms_kday_epoch1'] if r['rres_rms_kday_epoch1'] else None
  r['aux_grad_final_over_epoch1']=r['shared_aux_gradient_norm_epoch80']/r['shared_aux_gradient_norm_epoch1'] if r['shared_aux_gradient_norm_epoch1'] else None
  done.append(r);allmodels.append(dict(model='Auxiliary-only',ratio=j['ratio'],lead=j['lead'],seed=j['seed'],best_val_rmse=s['best_val_rmse'],best_epoch=s['best_epoch'],run_dir=j['run_dir']))
 pd.DataFrame(epochs).to_csv(AUX/'epoch_metrics.csv',index=False);pd.DataFrame(diags).to_csv(AUX/'diagnostics.csv',index=False)
 df=pd.DataFrame(done);df.to_csv(AUX/'seed_improvements.csv',index=False)
 all_df=pd.DataFrame(allmodels);all_df.to_csv(AUX/'all_models_validation_by_seed.csv',index=False)
 overview=all_df.groupby(['ratio','lead','model']).best_val_rmse.agg(['mean','std','count']).reset_index();overview.to_csv(AUX/'all_models_validation_summary.csv',index=False)
 summaries=[]
 if len(df):
  for (ratio,lead),g in df.groupby(['ratio','lead']):summaries.append(dict(ratio=ratio,lead=lead,n_seeds=len(g),baseline_rmse_mean=g.baseline_rmse.mean(),auxiliary_rmse_mean=g.auxiliary_rmse.mean(),auxiliary_rmse_sd=g.auxiliary_rmse.std() if len(g)>1 else None,mean_paired_improvement_pct=g.improvement_pct.mean(),sd_paired_improvement_pct=g.improvement_pct.std() if len(g)>1 else None,improved_seeds=int((g.improvement_pct>0).sum()),best_epoch_median=g.best_epoch.median()))
 pd.DataFrame(summaries).to_csv(AUX/'validation_summary.csv',index=False)
 (AUX/'checkpoint_audit.md').write_text('# Validation-best checkpoint audit\n\n감사용 초기 checkpoint 검증은 original_preflight/checkpoint_audit.md에 보존. 아래는 완료된 정식 학습 결과만 표시.\n\n|Run|Best epoch|기록 RMSE|재로드 RMSE|판정|\n|---|---|---|---|---|\n'+'\n'.join(f'|{r["name"]}|{r["epoch"]}|{r["expected_val_rmse"]:.9f}|{r["reloaded_val_rmse"]:.9f}|PASS|' for r in audits)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,lead in zip(axes,[7,14]):
  for model in ['Baseline','P0','NoAdv','Adv','Auxiliary-only']:
   g=overview[(overview.lead==lead)&(overview.model==model)].sort_values('ratio')
   if len(g):ax.errorbar(g.ratio,g['mean'],yerr=g['std'].fillna(0),marker='o',label=model,capsize=3)
  ax.set(title=f'Validation, lead {lead}',xlabel='Training subset (%)',ylabel='RMSE (K)');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(AUX/'figures/validation_comparison.png',dpi=180);plt.close(fig)
 if diags:
  dg=pd.DataFrame(diags);fig,axes=plt.subplots(1,2,figsize=(12,4))
  for lead in [7,14]:
   for ratio in [10,25,50,100]:
    g=dg[(dg.lead==lead)&(dg.ratio==ratio)&dg.epoch.isin([1,5,10,20,80])].groupby('epoch')[['rres_rms_kday','weighted_gradient_ratio']].mean()
    if len(g):
     axes[0].plot(g.index,g.rres_rms_kday,marker='o',label=f'{ratio}% L{lead}');axes[1].plot(g.index,g.weighted_gradient_ratio,marker='o',label=f'{ratio}% L{lead}')
  axes[0].set(title='Fixed Train batch: Rres RMS',ylabel='K/day',xlabel='Actual epoch');axes[1].set(title='Weighted auxiliary/data gradient norm ratio',xlabel='Actual epoch');axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(AUX/'figures/auxiliary_diagnostics.png',dpi=180);plt.close(fig)
 lines=[f'# Auxiliary24 Validation 보고 ({len(done)}/24 완료)',f'작성 {stamp()}','',
 'Zero-head 최소해는 의도된 출력 축소 정규화의 수학적 성질로 재분류했다. 이는 구현 오류나 실행 실패가 아니다. 기존 λaux=0.0001, Physics/tie OFF, v3 RHS scale, native architecture 유지. Test 실행 없음. 원 Oracle96 96개는 저장된 Validation 결과만 읽었다.',
 '학습 설정: 80 actual epochs, early stopping OFF, 매 epoch Validation-best 선택. 대응 Baseline initial SHA/실제 tensor, 입력/통계, 매 epoch sample order/LR를 검사한다. 진단은 고정 첫16 Train samples의 train-mode gradient (cuDNN GRU backward)로, 학습 batch 평균과 구분한다. RNG/model/.grad/mode 보존 및 smoke 진단ON/OFF 1step 동일성 검증.',
 '개선율은 각 seed (Baseline−Auxiliary)/Baseline×100 후 평균. 양수가 개선이다. 작은 차이를 통계적 유의성으로 해석하지 않으며 3seed의 방향 일관성을 함께 본다. 기존 Test 열람 이력이 있으나 이번 작업에서는 Test를 추가 사용하지 않는다.','',
 '|Ratio|Lead|완료 seed|Baseline RMSE 평균|Aux RMSE 평균±SD|대응 개선율 평균 (%)|개선 seed|Best epoch 중앙값|','|---|---|---|---|---|---|---|---|']
 for r in summaries:
  sd=f'{r["auxiliary_rmse_sd"]:.5f}' if r['auxiliary_rmse_sd'] is not None else 'NA'
  lines.append(f'|{r["ratio"]}|{r["lead"]}|{r["n_seeds"]}/3|{r["baseline_rmse_mean"]:.5f}|{r["auxiliary_rmse_mean"]:.5f} ± {sd}|{r["mean_paired_improvement_pct"]:+.3f}|{r["improved_seeds"]}/{r["n_seeds"]}|{r["best_epoch_median"]}|')
 if len(df):
  lines+=['',f'현재 완료 seed 조건 중 RMSE 개선 {int((df.improvement_pct>0).sum())}/{len(df)}. 개선율 범위 {df.improvement_pct.min():+.3f}% ~ {df.improvement_pct.max():+.3f}%.',
   f'고정 진단 batch의 Rres RMS가 epoch1보다 epoch80에 감소한 조건 {int((df.rres_rms_final_over_epoch1<1).sum())}/{len(df)}. 최종/초기 RMS 비 중앙값 {df.rres_rms_final_over_epoch1.median():.6g}.',
   f'Auxiliary gradient norm 최종/초기 비 중앙값 {df.aux_grad_final_over_epoch1.median():.6g}; 최종 gradient가 정확히0인 조건 {int((df.shared_aux_gradient_norm_epoch80==0).sum())}/{len(df)}. 작은 양수를 0이라고 표시하지 않았다.',
   f'가중 Aux/Data gradient 비 중앙값 epoch1={df.weighted_gradient_ratio_epoch1.median():.6g}, epoch80={df.weighted_gradient_ratio_epoch80.median():.6g}. 초기만 작용하는지는 감소율과 최종 크기를 함께 해석해야 한다. 정규화 출력 감소 자체를 SST 성능 향상 근거로 쓰지 않는다.']
 lines+=['','전체120개 비교는 all_models_validation_by_seed.csv/all_models_validation_summary.csv에 저장한다(미완료 신규 모델 제외). epoch_metrics.csv는 실제 학습 epoch별 기록, diagnostics.csv는 별도 고정 batch 진단, seed_improvements.csv는 seed 대응 개선율과 초기/최종 진단을 제공한다.','',f'기존 경로: {ORACLE}',f'신규 경로: {AUX}']
 (AUX/'report_ko.md').write_text('\n'.join(lines)+'\n');print('Validation-only summarized',len(done),'/24',flush=True)
if __name__=='__main__':main()
