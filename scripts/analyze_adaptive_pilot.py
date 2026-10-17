"""Final exploratory Validation report; no Test metrics consumed for selection."""
from adaptive_common import *
from summarize_adaptive_pilot import readcsv,merge
import statistics

def matched_gradient_summary(j, fixed_rows, adaptive_rows):
 ff=[r for r in fixed_rows if int(r['job'])==j['id'] and r['state']=='best']
 aa=[r for r in adaptive_rows if int(r['job'])==j['id']]
 assert len(ff)==len(aa)==32
 assert [int(r['batch']) for r in ff]==[int(r['batch']) for r in aa]
 nominal=[float(r['weighted_ratio']) for r in ff]
 effective=[float(r['weighted_ratio'])*float(r['checkpoint_warmup_factor']) for r in ff]
 ab=[float(r['lambda_base'])*float(r['G_phys'])/(float(r['G_data'])+1e-12) for r in aa]
 fc=[float(r['cosine']) for r in ff if r['cosine']!='']
 ac=[float(r['cosine']) for r in aa if r['cosine']!='']
 return dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],fixed_ratio_median=float(np.median(effective)),fixed_nominal_ratio_median=float(np.median(nominal)),adaptive_ratio_median=float(np.median([float(r['weighted_ratio']) for r in aa])),adaptive_base_ratio_median=float(np.median(ab)),fixed_effective_lambda=float(ff[0]['lambda_diagnostic'])*float(ff[0]['checkpoint_warmup_factor']),adaptive_effective_lambda=float(aa[0]['lambda_eff']),fixed_conflict_fraction=float(np.mean(np.array(fc)<0)),adaptive_conflict_fraction=float(np.mean(np.array(ac)<0)),fixed_device=ff[0]['device'],adaptive_device='cuda fp32',ratio_definition='Both primary ratios use actual checkpoint warmup-weighted coefficients; nominal/base ratios separately provided')

def main():
 pr=protocol();jobs=pr['jobs'];oldpr=json.loads((OLD/'protocol.json').read_text());validation=[];pairs=[];resources=[];trajectories=[];spatial=[];seasonal=[];extra=[]
 for j in jobs:
  d=Path(j['run_dir']);ev=OUT/'evaluations'/str(j['id']);res=json.loads((ev/'status.json').read_text());assert res['status']=='PASS';status=json.loads((d/'status.json').read_text());hist=json.loads((d/'history.json').read_text());assert status['status']=='trained'
  modelmetrics={}
  for arm,field in [('baseline','reference_baseline_run'),('fixed','reference_fixed_run'),('adaptive','run_dir')]:
   rd=Path(j[field]);s=json.loads((rd/'status.json').read_text());hh=json.loads((rd/'history.json').read_text())
   if arm=='adaptive':met=res['metrics'];source=ev
   else:
    oldj=next(x for x in oldpr['jobs'] if x['run_dir']==str(rd));oldmet=json.loads((Path(oldj['evaluation_dir'])/'metrics.json').read_text());met={k:oldmet['metrics'][k] for k in ['train','val']};source=Path(oldmet['fields']['val']['path']).parent
    path=Path(oldmet['fields']['val']['path']);assert sha(path)==oldmet['fields']['val']['sha256']
    with np.load(path) as z:
     err=z['pred_raw'].astype(float)-z['target_raw'].astype(float);mask=z['mask'].astype(bool);dates=z['target_dates'] if 'target_dates' in z else None;idx=z['target_index']
     if dates is None:
      import xarray as xr
      with xr.open_dataset(json.loads(Path(j['manifest']).read_text())['data_path']) as ds:dates=ds.time.values[idx]
     months=dates.astype('datetime64[M]').astype(int)%12+1;ss=np.array(['DJF','MAM','JJA','SON'])[(months%12)//3]
     for label in ['DJF','MAM','JJA','SON']:
      x=err[ss==label][mask[ss==label]];seasonal.append(dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=arm,season=label,rmse=float(np.sqrt((x*x).mean())),count=len(x)))
     for iy,ix in np.argwhere(mask[0,0]):spatial.append(dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=arm,iy=int(iy),ix=int(ix),rmse=float(np.sqrt((err[:,0,iy,ix]**2).mean()))))
   modelmetrics[arm]=met['val']['rmse'];row=dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=arm,validation_rmse=met['val']['rmse'],train_rmse=met['train']['rmse'],val_train_gap=met['val']['rmse']-met['train']['rmse'],best_update=s['best_epoch']*478,end_update=s['steps'],run_dir=str(rd),checkpoint_sha256=sha(rd/'best.pt'),peak_reserved_mib=s['peak_reserved_mib'],elapsed_logged_seconds=hh[-1]['elapsed_seconds'],clipped_fraction=sum(x['gradient_clipped_batches'] for x in hh)/(478*len(hh)))
   if arm=='adaptive':row.update(validation_physics_loss=met['val']['physics_loss'],validation_physics_residual_rmse=met['val']['raw_physics_residual_rmse'],lambda_at_best=res['lambda_base'])
   elif arm=='fixed':
    bestrow=next(x for x in hh if x['epoch']==s['best_epoch']);row.update(validation_physics_loss=bestrow['val_physics_loss'],validation_physics_residual_rmse=bestrow['val_common_res_rmse_kday'],lambda_at_best=.01,lambda_eff_at_best=bestrow['val_lambda_physics_effective'],physics_metric_source='saved Validation-best epoch history, native run_epoch')
   validation.append(row);resources.append(dict(**row,timing_caveat='Historical wall time affected by GPU concurrency/resume; controlled100-update benchmark is primary overhead comparison'))
  pairs.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],baseline_rmse=modelmetrics['baseline'],fixed_rmse=modelmetrics['fixed'],adaptive_rmse=modelmetrics['adaptive'],adaptive_vs_fixed_percent=100*(modelmetrics['fixed']-modelmetrics['adaptive'])/modelmetrics['fixed'],adaptive_vs_baseline_percent=100*(modelmetrics['baseline']-modelmetrics['adaptive'])/modelmetrics['baseline'],fixed_vs_baseline_percent=100*(modelmetrics['baseline']-modelmetrics['fixed'])/modelmetrics['baseline']))
  trajectory=readcsv(d/'lambda_trajectory.csv');trajectories += [dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],**r) for r in trajectory]
  values=[float(r['lambda_adaptive']) for r in trajectory];cos=[float(r['cosine']) for r in trajectory if r['cosine']];ratios=[float(r['effective_ratio']) for r in trajectory]
  extra.append(dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],lambda_mean=float(np.mean(values)),lambda_min=min(values),lambda_max=max(values),lambda_below_fixed_fraction=float(np.mean(np.array(values)<.01)),lower_clamp_fraction=sum(r['clamped_low']=='True' for r in trajectory)/len(trajectory),upper_clamp_fraction=sum(r['clamped_high']=='True' for r in trajectory)/len(trajectory),conflict_fraction=float(np.mean(np.array(cos)<0)),ratio_median=float(np.median(ratios)),measurement_count=len(trajectory),lambda_absolute_change_mean=float(np.mean(abs(np.diff(values)))),ema_lag_Gdata_mean=float(np.mean([abs(float(r['G_data'])-float(r['ema_G_data'])) for r in trajectory])),ema_lag_Gphys_mean=float(np.mean([abs(float(r['G_phys'])-float(r['ema_G_phys'])) for r in trajectory]))))
  seasonal+=readcsv(ev/'validation_seasonal.csv');spatial+=readcsv(ev/'validation_spatial.csv')
 summary=[]
 for ratio,lead in [(50,14),(10,7)]:
  pp=[r for r in pairs if r['ratio']==ratio and r['lead']==lead];c=dict(ratio=ratio,lead=lead)
  for model in ['baseline','fixed','adaptive']:
   x=[r[model+'_rmse'] for r in pp];c[model+'_mean']=statistics.mean(x);c[model+'_sd']=statistics.stdev(x)
  for key in ['adaptive_vs_fixed_percent','adaptive_vs_baseline_percent','fixed_vs_baseline_percent']:c[key]=statistics.mean(r[key] for r in pp)
  c['improved_vs_fixed_seeds']=sum(r['adaptive_rmse']<r['fixed_rmse'] for r in pp);c['improved_vs_baseline_seeds']=sum(r['adaptive_rmse']<r['baseline_rmse'] for r in pp);summary.append(c)
 for name,rows in [('validation_results',validation),('seed_improvements',pairs),('lambda_trajectory',trajectories),('training_resource_usage',resources),('adaptive_behavior',extra),('validation_seasonal',seasonal),('validation_spatial',spatial),('validation_summary',summary)]:csvsave(OUT/(name+'.csv'),rows)
 merge(['adaptive_best_gradients'],'evaluations')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,c in zip(axes,summary):ax.bar(['Baseline','Fixed','Adaptive'],[c[m+'_mean'] for m in ['baseline','fixed','adaptive']],yerr=[c[m+'_sd'] for m in ['baseline','fixed','adaptive']],capsize=4);ax.set(title=f'{c["ratio"]}% / Lead{c["lead"]}',ylabel='Validation RMSE (K)')
 fig.tight_layout();fig.savefig(OUT/'figures/validation_comparison.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for ax,j in zip(axes.flat,jobs):
  rr=[r for r in trajectories if r['job']==j['id']];ax.plot([int(r['update']) for r in rr],[float(r['lambda_adaptive']) for r in rr]);ax.axhline(.01,ls='--',color='grey');ax.set(title=f'{j["ratio"]}% L{j["lead"]} seed{j["seed"]}',xlabel='Training update',ylabel='Base lambda',ylim=(0,.031))
 fig.tight_layout();fig.savefig(OUT/'figures/lambda_trajectory.png',dpi=180);plt.close(fig)
 # Compare the SAME32 training batches at best checkpoints.
 fixed=readcsv(OUT/'gradient_diagnostics.csv');adaptive=readcsv(OUT/'adaptive_best_gradients.csv');alignment=[matched_gradient_summary(j,fixed,adaptive) for j in jobs]
 csvsave(OUT/'best_gradient_comparison.csv',alignment)
 def table(rows,keys):return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(f'{r[k]:.6g}' if isinstance(r.get(k),float) else str(r.get(k,'N/A')) for k in keys)+' |' for r in rows])
 lines=['# Adaptive Physics Weight pilot — Validation only','',stamp(),'','Exploratory pilot selected using previously inspected Test patterns. Six new Adaptive trainings, twelve strictly verified existing Baseline/Fixed models. No new Test evaluation and no automatic8-condition expansion.','', '## 1. Existing P0 equation audit','', (OUT/'physics_equation_audit.md').read_text(), '', '## 2. Physics residual diagnosis','', (OUT/'physics_residual_report.md').read_text(), '', '## 3. Gradient diagnosis','', (OUT/'gradient_report.md').read_text(), '', '## 4. Adaptive implementation and preflight','', (OUT/'adaptive_implementation.md').read_text(),(OUT/'preflight.md').read_text(), 'Controlled100-update timing/memory: resource_usage.csv; historical end-to-end timing: training_resource_usage.csv (concurrency/resume caveat).','', '## 5. Validation comparison','',table(summary,['ratio','lead','baseline_mean','baseline_sd','fixed_mean','fixed_sd','adaptive_mean','adaptive_sd','adaptive_vs_fixed_percent','adaptive_vs_baseline_percent','improved_vs_fixed_seeds','improved_vs_baseline_seeds']),'',table(pairs,['ratio','lead','seed','baseline_rmse','fixed_rmse','adaptive_rmse','adaptive_vs_fixed_percent','adaptive_vs_baseline_percent']), '', 'Adaptive behavior:',table(extra,['ratio','lead','seed','lambda_mean','lower_clamp_fraction','upper_clamp_fraction','ratio_median','conflict_fraction','lambda_below_fixed_fraction']),'','Matched best-checkpoint gradient batches: primary ratios use each checkpoint actual warmup-weighted coefficient. CSV also retains nominal/base ratios. Fixed diagnostics ran on CPU fp32 and Adaptive on GPU fp32, so small device arithmetic differences remain; these are not bitwise equivalence tests.',table(alignment,['ratio','lead','seed','fixed_ratio_median','adaptive_ratio_median','fixed_conflict_fraction','adaptive_conflict_fraction']),'','## 6. Interpretation and expansion decision','']
 lines += ['Native Validation physics metrics at each selected best checkpoint:',table([r for r in validation if r['model']!='baseline'],['ratio','lead','seed','model','validation_physics_loss','validation_physics_residual_rmse']),'']
 both=all(c['adaptive_mean']<c['fixed_mean'] for c in summary);casea=summary[0]['adaptive_mean']<summary[0]['fixed_mean'];consistent=all(c['improved_vs_fixed_seeds']>=2 for c in summary)
 decision='A: mean improvement in both cases; seed consistency must also be examined.' if both else 'B:50%L14 mean improvement only; existing favorable condition not preserved.' if casea else 'C: no improvement in both conditions, or inconsistent effect.'
 lines += [decision, f'At least2/3seeds improve versus Fixed in each case: {consistent}. This is a descriptive consistency check, not a preregistered significance test.','', 'Lower adaptive gradient ratio is magnitude control; cosine conflicts are not projected away. A lambda mean below .01 may indicate weaker physics rather than achieving the intended .25 ratio; clamp rates and matched gradient diagnostics above distinguish these observations. Full causal attribution is not possible. Seasonal and spatial Validation errors: validation_seasonal.csv/validation_spatial.csv; physics residual changes: validation_results.csv and original best-epoch histories. Gap decreases are decomposed using frozen Train/Validation errors. No causal contribution percentage is claimed.','', 'The observation-based native linear residual and true daily comparator quantify structural limitations: fixed1K contrast, horizon-constant positive we, reanalysis forcing and omitted processes. Poor closure alone is not evidence of a code bug.','', 'Recommendation: '+('Consider a separately authorized24-condition Adaptive extension after locking the current configuration; pilot6 reuse leaves18 additional trainings.' if both and consistent else 'Do not automatically expand. Retain the negative/unstable pilot findings and inspect clamp, EMA lag, conflicts and residual mismatch without further Test tuning.')]
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 benchmark=readcsv(OUT/'resource_usage.csv');avgseconds=float(np.mean([float(r['seconds'])/100 for r in benchmark if r['mode']=='adaptive']));steps=[r['end_update'] for r in validation if r['model']=='adaptive'];budget=18*38240*avgseconds/3600;observed=18*np.mean(steps)*avgseconds/3600
 (OUT/'followup_plan.md').write_text(f'# Follow-up plan only\n\nDecision: {decision}\n\nPossible extension:4ratios×2leads×3seeds=24Adaptive runs. If pilot6 remain compatible,18additional; Baseline/Fixed reuse48. Keep rho=.25,beta=.9,interval100,lambda[.0001,.03],warmup5 fixed before expansion. No automatic extension authorized/executed.\n\nTraining-step estimate from controlled pilot100-update benchmarks:18 extra runs ≈{observed:.2f} GPU-worker-hours at observed mean stopping updates, max-budget ≈{budget:.2f} GPU-worker-hours; excludes loading, Validation and I/O and is not wall-clock promise under concurrency. GPU1/2/3 only, max4processes each, memory admission required; measured planned2 each.\n\nFuture independent evaluation: after development freezes, obtain an untouched future period beyond existing2022-12-31 endpoint, predefine preprocessing/reference/metrics and do not tune on it. Existing1993–2022 source contains no unused future holdout beyond this endpoint. No new data acquisition or split modification in this pilot; if no independent holdout, all reuse of old Test remains exploratory.\n')
 for x in pr['reuse']:assert sha(Path(x['run_dir'])/'best.pt')==x['best_sha256'] and sha(Path(x['run_dir'])/'last.pt')==x['last_sha256']
 for n,hsh in pr['code_sha256'].items():assert sha(ROOT/n)==hsh
 assert len(validation)==18 and len(pairs)==6
 save(OUT/'final_integrity.json',dict(status='PASS',new_trainings=6,validation_comparisons=18,old_checkpoints_unchanged=True,no_test_evaluation=True,source_hashes_unchanged=True,timestamp=stamp(),summary=summary,decision=decision))
 log(f'Adaptive pilot complete: {decision}; results in {OUT}/report.md. Validation only; no Test/full extension.');print('ADAPTIVE ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
