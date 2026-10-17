"""All-condition v3 reporting; unchanged native MHW evaluator and paired circular bootstrap."""
import os,sys,json,csv,math,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import xarray as xr

def write_csv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def table(rows,keys):
 return ['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(f'{r[k]:.5f}' if isinstance(r.get(k),float) else str(r.get(k,'')) for k in keys)+' |' for r in rows]
def main(study):
 out=study/'analysis';out.mkdir(exist_ok=True)
 cache=study/'runtime/analysis';cache.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from mhw.mhw_skill import compute_climatology_threshold,day_of_year
 from summarize_jeju_hilam_mhw_skill import row_from_arrays
 protocol=json.loads((study/'protocol.json').read_text());jobs=protocol['jobs'];assert len(jobs)==48
 for j in jobs:assert json.loads((Path(j['run_dir'])/'evaluation_status.json').read_text())['status']=='completed'
 manifest=json.loads(Path(jobs[0]['manifest']).read_text());ds=xr.open_dataset(manifest['data_path'])
 sst=ds.sst_mean.values.astype(np.float32);times=ds.time.values;doy=day_of_year(times);mask=np.isfinite(ds.ocean_mask_static.values).astype(np.float32);assert int(mask.sum())==661
 ref=np.zeros(len(times),dtype=bool);ref[:int(round(len(times)*.70))]=True
 clim,threshold=compute_climatology_threshold(sst=sst,doy=doy,reference_time_mask=ref,window=5,quantile=.90)
 np.savez_compressed(out/'mhw_reference.npz',climatology=clim,threshold=threshold,mask=mask,reference_time_indices=np.flatnonzero(ref))
 mhwargs=SimpleNamespace(min_duration=5,area_frac=.15,onset_tolerance=3)
 rows=[];daily={};lookup={};by_lead={};stability=[]
 for j in jobs:
  d=Path(j['run_dir']);m=json.loads((d/'test_metrics.json').read_text());status=json.loads((d/'status.json').read_text());h=json.loads((d/'history.json').read_text())
  with np.load(d/'test_fields.npz') as f:
   pred=f['pred_raw'][:,0];obs=f['target_raw'][:,0];idx=f['target_index'];assert np.isfinite(pred).all()
   assert np.array_equal(f['mask'][:,0],np.broadcast_to(mask,obs.shape))
   np.testing.assert_array_equal(obs[:,mask>0],sst[idx][:,mask>0])
   if j['lead'] in by_lead:np.testing.assert_array_equal(idx,by_lead[j['lead']])
   else:by_lead[j['lead']]=idx.copy()
   assert np.all(np.diff(idx)==1)
   metrics=row_from_arrays(root_label='v3',run_id=j['name'],group=j['arm'],model=j['arm'],model_label=j['arm'],lead=j['lead'],history=30,seed=j['seed'],alpha=None,pred=pred,obs=obs,mask2d=mask,target_idx=idx,doy=doy,clim=clim,thresh=threshold,args=mhwargs)
   daily[j['ratio'],j['lead'],j['arm'],j['seed']]=f['daily_mse'].copy()
   assert np.isclose(np.sqrt(f['daily_mse'].mean()),m['test_rmse'],atol=2e-6,rtol=0)
  row=dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['arm'],rmse=m['test_rmse'],mae=m['test_mae'],bias=m['test_bias'],acc=metrics['test_acc'],csi=metrics['mhw_grid_csi'],pod=metrics['mhw_grid_pod'],far=metrics['mhw_grid_far'],best_epoch=m['best_epoch'],best_val_rmse=m['best_val_rmse'])
  rows.append(row);lookup[j['ratio'],j['lead'],j['seed'],j['arm']]=row
  stability.append(dict(ratio=j['ratio'],lead=j['lead'],model=j['arm'],seed=j['seed'],best_epoch=m['best_epoch'],best_val_rmse=m['best_val_rmse'],epochs=status['epochs'],steps=status['steps'],early_stopping=status['early_stopping'],max_gradient_norm=max(r['gradient_norm_max'] for r in h),clipped_batches=sum(r['gradient_clipped_batches'] for r in h),status='completed'))
  # Native MHW metrics may contain undefined event diagnostics; preserve as null.
  clean={k:None if isinstance(v,(float,np.floating)) and not math.isfinite(v) else v for k,v in metrics.items()}
  (d/'mhw_metrics.json').write_text(json.dumps(clean,indent=2,default=str,allow_nan=False)+'\n')
 seeds=[42,7,123];paired=[];summary=[];comparisons=[]
 for ratio in [10,25,50,100]:
  for lead in [7,14]:
   for seed in seeds:
    b=lookup[ratio,lead,seed,'baseline'];p=lookup[ratio,lead,seed,'p0']
    paired.append(dict(ratio=ratio,lead=lead,seed=seed,baseline_rmse=b['rmse'],p0_rmse=p['rmse'],delta_rmse_p0_minus_baseline=p['rmse']-b['rmse'],improvement_percent=(b['rmse']-p['rmse'])/b['rmse']*100))
   ps=[r for r in paired if r['ratio']==ratio and r['lead']==lead];imp=np.array([r['improvement_percent'] for r in ps]);delta=np.array([r['delta_rmse_p0_minus_baseline'] for r in ps])
   comparisons.append(dict(ratio=ratio,lead=lead,improvement_mean=float(imp.mean()),improvement_sd=float(imp.std(ddof=1)),delta_rmse_mean=float(delta.mean()),delta_rmse_sd=float(delta.std(ddof=1)),improved_seeds=int((imp>0).sum())))
   for arm in ['baseline','p0']:
    rs=[lookup[ratio,lead,s,arm] for s in seeds];r=dict(ratio=ratio,lead=lead,model=arm)
    for key in ['rmse','mae','bias','acc','csi','pod','far']:
     values=np.array([x[key] for x in rs]);r[key+'_mean']=float(values.mean());r[key+'_sd']=float(values.std(ddof=1))
    summary.append(r)
 # Same circular block index construction as archived summarize_physics_pilot.py:69–73.
 boots=[];replicates={}
 for lead in [7,14]:
  n=len(by_lead[lead])
  for block in [30,90,180]:
   generator=np.random.default_rng(20260925+lead+block)
   start=generator.integers(0,n,size=(4000,int(np.ceil(n/block))))
   idx=((start[:,:,None]+np.arange(block))%n).reshape(4000,-1)[:,:n]
   for ratio in [10,25,50,100]:
    b=np.array([np.sqrt(daily[ratio,lead,'baseline',s][idx].mean(axis=1)) for s in seeds]);p=np.array([np.sqrt(daily[ratio,lead,'p0',s][idx].mean(axis=1)) for s in seeds])
    delta=(p-b).mean(axis=0);imp=((b-p)/b*100).mean(axis=0)
    lo,hi=np.quantile(delta,[.025,.975]);ilo,ihi=np.quantile(imp,[.025,.975])
    boots.append(dict(ratio=ratio,lead=lead,block_days=block,replicates=4000,delta_rmse_ci_low=float(lo),delta_rmse_ci_high=float(hi),delta_ci_contains_zero=bool(lo<=0<=hi),improvement_ci_low=float(ilo),improvement_ci_high=float(ihi),scope='conditional on fitted models; paired time resampling, no seed resampling'))
    replicates[f'ratio{ratio}_lead{lead}_block{block}_delta']=delta;replicates[f'ratio{ratio}_lead{lead}_block{block}_improvement']=imp
 np.savez_compressed(out/'bootstrap_replicates.npz',**replicates)
 volume=[dict(ratio=r,lead7_improvement_mean=next(x['improvement_mean'] for x in comparisons if x['ratio']==r and x['lead']==7),lead14_improvement_mean=next(x['improvement_mean'] for x in comparisons if x['ratio']==r and x['lead']==14)) for r in [10,25,50,100]]
 for filename,values in [('runs.csv',rows),('paired_seed_results.csv',paired),('model_summary.csv',summary),('paired_summary.csv',comparisons),('volume_improvement.csv',volume),('mhw_summary.csv',summary),('stability.csv',stability),('bootstrap.csv',boots)]:write_csv(out/filename,values)
 plt.rcParams.update({'font.size':10,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 def export(fig,name):
  fig.tight_layout();fig.savefig(out/(name+'.png'));fig.savefig(out/(name+'.pdf'));plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.5,4))
 for lead,color in [(7,'#0072B2'),(14,'#D55E00')]:
  rs=[r for r in comparisons if r['lead']==lead];bs=[next(b for b in boots if b['ratio']==r['ratio'] and b['lead']==lead and b['block_days']==90) for r in rs]
  ax.plot([r['ratio'] for r in rs],[r['improvement_mean'] for r in rs],marker='o',color=color,label=f'Lead {lead} days')
  ax.fill_between([r['ratio'] for r in rs],[b['improvement_ci_low'] for b in bs],[b['improvement_ci_high'] for b in bs],color=color,alpha=.15)
 ax.axhline(0,color='black',lw=.8);ax.set(xticks=[10,25,50,100],xlabel='Training subset (%)',ylabel='Mean paired RMSE reduction (%)',title='Shading: conditional 95% CI, 90-day blocks');ax.legend();export(fig,'figure1_improvement')
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,lead in zip(axes,[7,14]):
  for arm,color in [('baseline','#0072B2'),('p0','#D55E00')]:
   rs=[r for r in summary if r['lead']==lead and r['model']==arm];ax.errorbar([r['ratio'] for r in rs],[r['rmse_mean'] for r in rs],yerr=[r['rmse_sd'] for r in rs],marker='o',capsize=3,color=color,label=arm)
  ax.set(xticks=[10,25,50,100],xlabel='Training subset (%)',title=f'Lead {lead} days; mean ± seed SD');ax.legend()
 axes[0].set_ylabel('SST RMSE (K)');export(fig,'figure2_rmse')
 dynamics=[]
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for row,lead in enumerate([7,14]):
  for arm,color in [('baseline','#0072B2'),('p0','#D55E00')]:
   job=next(j for j in jobs if j['ratio']==10 and j['lead']==lead and j['seed']==42 and j['arm']==arm);history=json.loads((Path(job['run_dir'])/'history.json').read_text());epochs=[r['epoch'] for r in history]
   dynamics.extend([dict(lead=lead,model=arm,**r) for r in history])
   axes[row,0].plot(epochs,[r['train_data_loss'] for r in history],color=color,label=arm+' train')
   axes[row,0].plot(epochs,[r['val_data_loss'] for r in history],color=color,ls='--',label=arm+' val')
   axes[row,2].plot(epochs,[r['gradient_norm_mean'] for r in history],color=color,label=arm+' norm')
   axes[row,2].plot(epochs,[r['gradient_clip_fraction'] for r in history],color=color,ls=':',label=arm+' clip fraction')
   if arm=='p0':axes[row,1].plot(epochs,[r['train_physics_loss'] for r in history],label='train');axes[row,1].plot(epochs,[r['val_physics_loss'] for r in history],label='val')
  for col,title in enumerate(['Data loss','P0 physics loss','Preclip gradient norm / clip fraction']):axes[row,col].set(xlabel='Validation interval (478 updates)',title=f'10%, seed42, lead{lead}: {title}');axes[row,col].legend(fontsize=7)
 export(fig,'figure3_dynamics');write_csv(out/'figure3_dynamics.csv',dynamics)
 queue=json.loads((study/'queue_state.json').read_text());training_attempts=[a for a in queue['attempts'] if a['phase']=='train'];retries=max(0,len(training_attempts)-48)
 preflight=json.loads((study/'preflight.json').read_text())
 prior=Path(preflight['previous_attempt']) if preflight.get('previous_attempt') else None
 prior_failed=len(list(prior.glob('runs/**/failure_train_*.json'))) if prior else 0
 execution=dict(planned=48,completed=48,unresolved_failures=0,prior_diagnostic_failures=prior_failed,prior_paired_interruptions=1 if prior_failed else 0,current_study_retries=retries,prior_fresh_restarts=2 if prior_failed else 0,historical_checkpoints_reused=0)
 (out/'execution_summary.json').write_text(json.dumps(execution,indent=2)+'\n')
 lines=['# v3 Baseline / P0 physics-only: 48-run final report','',f'Completed training: 48/48. Evaluated best-validation checkpoints: 48/48. Training retries: {retries}. Historical checkpoints reused: 0. λP0=0.1 fixed; Rres=0.','', '100% subset = 3,995 windows, not Full training. Error bars use sample SD across three seeds. Improvement is the mean of three paired relative improvements, not the ratio of mean RMSEs.','', 'Execution accounting: '+json.dumps(execution), '', '## A. Seed-paired SST results','']+table(paired,['ratio','lead','seed','baseline_rmse','p0_rmse','improvement_percent'])+['','## B. Data-volume comparison','']+table(volume,['ratio','lead7_improvement_mean','lead14_improvement_mean'])+['','## C. Model performance','']+table(summary,['ratio','lead','model','rmse_mean','rmse_sd','mae_mean'])+['','## D. MHW detection','']+table(summary,['ratio','lead','model','csi_mean','pod_mean','far_mean'])+['','## E. Stability','']+table(stability,['ratio','lead','model','seed','best_epoch','best_val_rmse','status'])+['','## Paired uncertainty','']+table(comparisons,['ratio','lead','improvement_mean','improvement_sd','delta_rmse_mean','delta_rmse_sd','improved_seeds'])+['','Primary bootstrap: paired circular 90-day blocks, 4,000 repetitions; 30/180-day sensitivity in bootstrap.csv. CI conditions on fitted models and resamples dates jointly across both models and all three seeds. It does not represent training-seed or model-family uncertainty.','']+table([r for r in boots if r['block_days']==90],['ratio','lead','delta_rmse_ci_low','delta_rmse_ci_high','delta_ci_contains_zero'])+['','## Research questions','']
 positive=[(r['ratio'],r['lead']) for r in comparisons if r['improvement_mean']>0];negative=[(r['ratio'],r['lead']) for r in comparisons if r['improvement_mean']<=0]
 lines.append(f'RQ1: Mean paired improvement >0 for (ratio, lead) {positive}; non-improvement for {negative}. Every seed is reported above; three seeds do not support strong significance claims.')
 for lead in [7,14]:
  values=[r['improvement_mean'] for r in comparisons if r['lead']==lead];lines.append(f'RQ2, lead {lead}: improvements by 10/25/50/100% = {values}; monotonically nonincreasing = {bool(np.all(np.diff(values)<=0))}.')
 signs=[(r,math.copysign(1,next(x['improvement_mean'] for x in comparisons if x['ratio']==r and x['lead']==7))==math.copysign(1,next(x['improvement_mean'] for x in comparisons if x['ratio']==r and x['lead']==14))) for r in [10,25,50,100]];lines.append(f'RQ3: Lead7/14 improvement signs agree by ratio: {signs}.')
 for ratio,lead in positive:
  b=next(r for r in summary if (r['ratio'],r['lead'],r['model'])==(ratio,lead,'baseline'));p=next(r for r in summary if (r['ratio'],r['lead'],r['model'])==(ratio,lead,'p0'));lines.append(f'RQ4: SST-improved ratio{ratio}/lead{lead}: ΔCSI={p["csi_mean"]-b["csi_mean"]:+.5f}, ΔPOD={p["pod_mean"]-b["pod_mean"]:+.5f}, ΔFAR={p["far_mean"]-b["far_mean"]:+.5f} (lower FAR is better).')
 lines+=['','## Historical context','']
 for lead,old in [(7,4.63),(14,2.85)]:
  value=next(r['improvement_mean'] for r in comparisons if r['ratio']==10 and r['lead']==lead);lines.append(f'10% lead{lead}: historical Full-statistics reference {old:+.2f}%, v3 {value:+.5f}%.')
 lines+=['Input/target statistics, anomaly climatology, coastal gradients and auxiliary-loss settings differ. Changes cannot be attributed to one isolated modification.','', 'MHW evaluation remains original raw-SST first70% reference, quantile0.9, DOY window5, min-duration5, area fraction0.15 and onset tolerance3; training anomaly climatology is separate.','', 'Figures and their CSV data are in analysis/. Generator: scripts/analyze_v3_study.py. All config, logs, checkpoints and date-level errors are preserved.']
 (study/'final_report.md').write_text('\n'.join(lines)+'\n');ds.close();print(study/'final_report.md',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);main(ap.parse_args().study)
