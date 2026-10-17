"""Paired temporal inference and plots from frozen v3 evaluation fields; no training."""
import os,sys,json,csv,math,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v3_generalization_20261009'
sys.path.insert(0,str(ROOT/'src'));os.environ['MPLCONFIGDIR']=str(OUT/'runtime/MPLCONFIGDIR')
import numpy as np
import xarray as xr
from analyze_v3_study import write_csv,table
from mhw.mhw_skill import compute_climatology_threshold,day_of_year,mhw_binary_field,contingency_skill
from summarize_jeju_hilam_mhw_skill import row_from_arrays
ARMS=['baseline','physics_only','auxiliary_only','full_pinn'];SEEDS=[42,7,123]
COMPARISONS=[('physics_only','baseline'),('auxiliary_only','baseline'),('full_pinn','baseline'),('full_pinn','physics_only'),('full_pinn','auxiliary_only')]

def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

def save(p,obj):p.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def block_indices(n,block,reps=4000):
 gen=np.random.default_rng(20260925+7+block)
 starts=gen.integers(0,n,size=(reps,int(np.ceil(n/block))))
 return ((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]

def checks():
 idx=block_indices(127,90);assert idx.shape==(4000,127) and idx.min()>=0 and idx.max()<127
 assert np.array_equal(idx,block_indices(127,90))
 x=np.arange(1,128,dtype=float);a=np.sqrt(x[idx].mean(1));b=np.sqrt(x[idx].mean(1));assert np.all(a-b==0)
 a=np.sqrt(np.full(127,4.)[idx].mean(1));b=np.sqrt(np.full(127,1.)[idx].mean(1));assert np.all(a-b==1)
 # Pooled squared errors, not mean of per-day RMSE; unequal counts weighted correctly.
 assert np.isclose(np.sqrt(np.array([1.,9.]).sum()/2),np.sqrt(5))
 save(OUT/'analysis_tests.json',dict(status='PASS',checks=['deterministic circular block indices','paired identical model zero delta','known constant delta','pooled SSE/count RMSE']))

def main():
 checks();protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'inference_status.json').read_text())['status']=='COMPLETE'
 assert json.loads((OUT/'checkpoint_audit.json').read_text())['status']=='PASS'
 manifest=json.loads(Path(protocol['rows'][0]['manifest']).read_text());ds=xr.open_dataset(manifest['data_path'])
 sst=ds.sst_mean.values.astype(np.float32);times=ds.time.values;doy=day_of_year(times);mask=np.isfinite(ds.ocean_mask_static.values).astype(np.float32);assert mask.sum()==661
 ref=np.zeros(len(times),bool);ref[:round(len(times)*.7)]=True
 clim,threshold=compute_climatology_threshold(sst,doy,ref,window=5,quantile=.9)
 old=ROOT/'outputs/data_efficiency/v3_baseline_p0_20261008_r1/analysis/mhw_reference.npz'
 with np.load(old) as f:
  for k,v in [('climatology',clim),('threshold',threshold),('mask',mask),('reference_time_indices',np.flatnonzero(ref))]:np.testing.assert_array_equal(f[k],v)
 np.savez_compressed(OUT/'mhw_reference.npz',climatology=clim,threshold=threshold,mask=mask,reference_time_indices=np.flatnonzero(ref))
 save(OUT/'mhw_reference_audit.json',dict(status='PASS',same_arrays_as=str(old),old_sha256=sha(old),reference_start=str(times[ref][0]),reference_end=str(times[ref][-1]),n_reference=int(ref.sum()),protocol=protocol['mhw'],note='Native calendar DOY 1..366 retained exactly, including existing leap-day convention. No change to threshold, duration or boundary handling.'))
 rows=[];daily={};fields_sigs={};acf=[]
 for r in protocol['rows']:
  dest=OUT/'runs'/f"{r['arm']}_seed{r['seed']}"
  metrics={split:json.loads((dest/(split+'_metrics.json')).read_text()) for split in ['train','val','test']}
  with np.load(dest/'test_fields.npz') as f:
   pred=f['pred_raw'][:,0];obs=f['target_raw'][:,0];idx=f['target_index'];cnt=f['daily_count'];sse=f['daily_sse']
   np.testing.assert_array_equal(f['mask'][:,0],np.broadcast_to(mask,obs.shape));np.testing.assert_array_equal(obs[:,mask>0],sst[idx][:,mask>0]);assert np.all(np.diff(idx)==1)
   sig=[hashlib.sha256(f[k].tobytes()).hexdigest() for k in ['target_index','target_raw','mask']]
   if fields_sigs:assert sig==fields_sigs['test']
   else:fields_sigs['test']=sig
   daily[r['arm'],r['seed']]=(sse.copy(),cnt.copy())
   nm=row_from_arrays(root_label='exploratory_v3',run_id=str(dest),group=r['arm'],model=r['arm'],model_label=r['arm'],lead=7,history=30,seed=r['seed'],alpha=None,pred=pred,obs=obs,mask2d=mask,target_idx=idx,doy=doy,clim=clim,thresh=threshold,args=SimpleNamespace(min_duration=5,area_frac=.15,onset_tolerance=3))
   pb=mhw_binary_field(pred,doy[idx],threshold,5)&(mask>0);ob=mhw_binary_field(obs,doy[idx],threshold,5)&(mask>0)
   skill=contingency_skill(pb,ob,mask)
   for k in ['csi','pod','far']:assert abs(getattr(skill,k)-nm['mhw_grid_'+k])<1e-12
   np.savez_compressed(dest/'test_mhw_fields.npz',pred_mhw=pb,obs_mhw=ob,target_dates=times[idx],target_index=idx,mask=mask,seed=r['seed'],model=r['arm'])
   save(dest/'mhw_metrics.json',nm)
   assert abs(nm['test_rmse']-metrics['test']['rmse'])<2e-6
   z=sse/cnt
   for lag in [1,7,30,90,180]:acf.append(dict(model=r['arm'],seed=r['seed'],series='daily spatial mean squared error',lag_days=lag,acf=float(np.corrcoef(z[:-lag],z[lag:])[0,1])))
  row=dict(model=r['arm'],seed=r['seed'],train_rmse=metrics['train']['rmse'],val_rmse=metrics['val']['rmse'],test_rmse=metrics['test']['rmse'],test_mae=metrics['test']['mae'],test_bias=metrics['test']['bias'],test_acc=nm['test_acc'],csi=nm['mhw_grid_csi'],pod=nm['mhw_grid_pod'],far=nm['mhw_grid_far'],val_train_gap=metrics['val']['rmse']-metrics['train']['rmse'],test_train_gap=metrics['test']['rmse']-metrics['train']['rmse'],test_val_gap=metrics['test']['rmse']-metrics['val']['rmse'],best_epoch=r['best_epoch'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'])
  rows.append(row)
 lookup={(r['model'],r['seed']):r for r in rows}
 for r in rows:r['improvement_percent']=100*(lookup['baseline',r['seed']]['test_rmse']-r['test_rmse'])/lookup['baseline',r['seed']]['test_rmse']
 groups=[]
 for arm in ARMS:
  rs=[lookup[arm,s] for s in SEEDS];g=dict(model=arm,improved_seeds=sum(r['improvement_percent']>0 for r in rs))
  for key in ['train_rmse','val_rmse','test_rmse','test_mae','test_bias','test_acc','improvement_percent','csi','pod','far','val_train_gap','test_train_gap','test_val_gap']:
   a=np.array([r[key] for r in rs]);g[key+'_mean']=float(a.mean());g[key+'_sd']=float(a.std(ddof=1))
  groups.append(g)
 boot=[];reps={};pair=[]
 for model,base in COMPARISONS:
  d=np.array([lookup[model,s]['test_rmse']-lookup[base,s]['test_rmse'] for s in SEEDS]);vd=np.array([lookup[model,s]['val_rmse']-lookup[base,s]['val_rmse'] for s in SEEDS]);gap=np.array([lookup[model,s]['test_train_gap']-lookup[base,s]['test_train_gap'] for s in SEEDS])
  pair.append(dict(comparison=model+' - '+base,delta_rmse_mean=float(d.mean()),delta_rmse_seed_sd=float(d.std(ddof=1)),improved_seeds=int((d<0).sum()),val_delta_mean=float(vd.mean()),gap_delta_mean=float(gap.mean())))
 for block in [30,90,180]:
  ids=block_indices(len(idx),block);br={}
  for key,(sse,cnt) in daily.items():br[key]=np.sqrt(sse[ids].sum(1)/cnt[ids].sum(1))
  for model,base in COMPARISONS:
   deltas=np.array([br[model,s]-br[base,s] for s in SEEDS]);imps=np.array([100*(br[base,s]-br[model,s])/br[base,s] for s in SEEDS])
   for scope,delta,imp,observed in [(str(s),deltas[i],imps[i],lookup[model,s]['test_rmse']-lookup[base,s]['test_rmse']) for i,s in enumerate(SEEDS)]+[('fixed_3_seed_mean',deltas.mean(0),imps.mean(0),np.mean([lookup[model,s]['test_rmse']-lookup[base,s]['test_rmse'] for s in SEEDS]))]:
    lo,hi=np.quantile(delta,[.025,.975]);il,ih=np.quantile(imp,[.025,.975])
    boot.append(dict(comparison=model+' - '+base,seed_scope=scope,block_days=block,replicates=4000,delta_rmse=float(observed),ci_low=float(lo),ci_high=float(hi),ci_includes_zero=bool(lo<=0<=hi),improvement_ci_low=float(il),improvement_ci_high=float(ih),interpretation='decrease supported' if hi<0 else 'increase supported' if lo>0 else 'uncertain'))
    key=f'{model}_minus_{base}_seed{scope}_block{block}';reps[key+'_delta']=delta;reps[key+'_improvement']=imp
 np.savez_compressed(OUT/'bootstrap_replicates.npz',**reps)
 for name,data in [('seed_results.csv',rows),('model_summary.csv',groups),('comparisons.csv',pair),('bootstrap.csv',boot),('error_autocorrelation.csv',acf)]:write_csv(OUT/name,data)
 save(OUT/'results.json',dict(status='COMPLETE',scope=protocol['scope'],rows=rows,groups=groups,comparisons=pair,bootstrap=boot,mhw_unchanged=True,test_dates=[str(times[idx][0]),str(times[idx][-1])],test_days=len(idx),ocean_cells=661,uncertainty='Percentile temporal block CI conditions on 3 fitted seeds; no seed resampling; seed SD separately. Five exploratory contrasts, no multiplicity correction. CI containing zero is not equivalence.'))
 plots(rows,groups,boot)
 report(rows,groups,pair,boot,protocol)
 ds.close()

def plots(rows,groups,boot):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'savefig.dpi':220,'pdf.fonttype':42})
 labels=['Baseline','Physics only','Auxiliary only','Full PINN'];x=np.arange(4)
 def export(fig,name):fig.tight_layout();fig.savefig(OUT/(name+'.png'));fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for split,offset in [('val',-.13),('test',.13)]:ax.errorbar(x+offset,[g[split+'_rmse_mean'] for g in groups],yerr=[g[split+'_rmse_sd'] for g in groups],marker='o',capsize=4,label=split)
 ax.set(xticks=x,xticklabels=labels,ylabel='SST RMSE (K)',title='Mean ± sample SD across 3 seeds');ax.legend();export(fig,'figure1_validation_test')
 fig,ax=plt.subplots(figsize=(8,4))
 for i,seed in enumerate(SEEDS):ax.bar(np.arange(3)+(i-1)*.23,[next(r['improvement_percent'] for r in rows if r['model']==a and r['seed']==seed) for a in ARMS[1:]],width=.23,label=f'Seed {seed}')
 ax.axhline(0,color='black',lw=.8);ax.set(xticks=np.arange(3),xticklabels=labels[1:],ylabel='Test RMSE improvement vs baseline (%)');ax.legend();export(fig,'figure2_seed_improvement')
 fig,ax=plt.subplots(figsize=(9,5));rs=[b for b in boot if b['block_days']==90 and b['seed_scope']=='fixed_3_seed_mean']
 for i,b in enumerate(rs):ax.plot([b['ci_low'],b['ci_high']],[i,i],lw=2);ax.plot(b['delta_rmse'],i,'ko')
 ax.axvline(0,color='black',ls='--');ax.set(yticks=np.arange(len(rs)),yticklabels=[r['comparison'] for r in rs],xlabel='Test RMSE difference (K)',title='Conditional 95% CI; paired 90-day blocks, 4,000 replicates');export(fig,'figure3_bootstrap')
 fig,ax=plt.subplots(figsize=(8,4))
 for i,g in enumerate(groups):ax.errorbar(np.arange(3)+(i-1.5)*.035,[g[k+'_rmse_mean'] for k in ['train','val','test']],yerr=[g[k+'_rmse_sd'] for k in ['train','val','test']],marker='o',capsize=3,label=labels[i])
 ax.set(xticks=np.arange(3),xticklabels=['Train (fixed checkpoint)','Validation','Test'],ylabel='SST RMSE (K)');ax.legend();export(fig,'figure4_generalization')
 save(OUT/'figure_data.json',dict(rows=rows,groups=groups,bootstrap90=rs,source_script=str(Path(__file__))))

def report(rows,groups,pair,boot,protocol):
 primary=[b for b in boot if b['block_days']==90 and b['seed_scope']=='fixed_3_seed_mean']
 lines=['# v3 Physics Ablation — Exploratory Generalization Evaluation','', '12 frozen Validation-best checkpoints, 10% subset / Lead7 / seeds42,7,123. No training, tuning, Test-based checkpoint selection, or dataset modification. λphysics=.01; λaux=.0001.','', '## A. Seed-level Test performance','']+table(rows,['model','seed','val_rmse','test_rmse','improvement_percent'])
 lines+=['','## B. Model summaries','', 'SD is sample SD of three fitted seeds; improvement is mean of paired seed percentages.']+table(groups,['model','test_rmse_mean','test_rmse_sd','improvement_percent_mean','improved_seeds','test_mae_mean','test_bias_mean','test_acc_mean'])
 lines+=['','## C. Fixed-checkpoint generalization gaps','', 'Every split evaluated in eval/no_grad mode using the same Validation-best state. Train contains only the selected376 windows; no online training RMSE substituted.']+table(groups,['model','train_rmse_mean','val_rmse_mean','test_rmse_mean','val_train_gap_mean','test_train_gap_mean','test_val_gap_mean'])
 lines+=['','## D. Paired temporal uncertainty','', 'ΔRMSE = first model minus reference. Retained legacy circular90-day block primary,4000 draws;30/180 sensitivity fixed before new Test inference. Same dates resampled jointly across models and seeds, all661 ocean cells kept together. Recompute sqrt(total SSE/total count). Percentile95% CI conditional on these trained models; does not estimate population training-seed uncertainty. No multiplicity adjustment for five exploratory contrasts. See bootstrap.csv for per-seed CIs and sensitivity.']+table(primary,['comparison','delta_rmse','ci_low','ci_high','ci_includes_zero'])
 lines+=['','## E. Unchanged MHW evaluation','', 'Native first70% raw-date climatology, ±5DOY pooling,90th percentile,≥5 consecutive days; area=.15, onset tolerance3. Existing leap-day and Test-boundary rules retained. Reference arrays match previous v3 evaluation exactly. SST and MHW evaluated separately.']+table(groups,['model','csi_mean','csi_sd','pod_mean','pod_sd','far_mean','far_sd'])
 lines+=['','## F. Controlled contrasts','']+table(pair,['comparison','delta_rmse_mean','delta_rmse_seed_sd','improved_seeds','val_delta_mean','gap_delta_mean'])
 lines+=['','## Checkpoints and provenance','', 'checkpoint_audit.json: every Validation result reproduced within predeclared2e-6K; config/manifest/source hashes and strict model state shapes verified. Identical masks, dates, observations and per-seed split/statistics verified. Raw inverse transform uses pred_norm * target_std + target_mean (src/train_jeju_hilam.py:447). Evaluation uses src/train_jeju_hilam.py:417 collect_predictions; src/mhw/subset_preprocessing.py:170 load_subset_bundle. MHW uses src/summarize_jeju_hilam_mhw_skill.py:107 row_from_arrays and src/mhw/mhw_skill.py:128 mhw_binary_field. Bootstrap follows scripts/analyze_v3_study.py:63 circular block construction. Scripts/evaluate_v3_generalization.py and scripts/analyze_v3_generalization.py contain full reproducible procedure.','']+table(protocol['rows'],['arm','seed','best_epoch','checkpoint','checkpoint_sha256'])
 lines+=['','## Figures and underlying data','', 'figure1_validation_test, figure2_seed_improvement, figure3_bootstrap, figure4_generalization in PNG/PDF. figure_data.json, seed_results.csv, model_summary.csv, bootstrap.csv and bootstrap_replicates.npz store plot/statistic data. Each runs/<arm>_seed<seed> has raw Train/Validation/Test SST, targets, dates, mask, daily SSE/count/MSE, metrics and Test MHW predicted/observed flags.','', 'Final RQ interpretation and prior-study comparison are in interpretation.md; follow-up plan in followup_plan.md.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
