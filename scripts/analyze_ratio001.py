"""All48 fixed-lambda evaluations: paired time bootstrap, scarcity contrasts, reports."""
import os,sys,json,csv,math,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v3_physics_ratio_lead_lambda001_20261009'
sys.path.insert(0,str(ROOT/'src'));os.environ['MPLCONFIGDIR']=str(OUT/'runtime')
import numpy as np
import xarray as xr
from analyze_v3_study import write_csv,table
from mhw.mhw_skill import day_of_year,mhw_binary_field,contingency_skill
from summarize_jeju_hilam_mhw_skill import row_from_arrays
RATIOS=[10,25,50,100];LEADS=[7,14];SEEDS=[42,7,123];ARMS=['baseline','physics_only']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def save(p,obj):p.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def indices(n,lead,block):
 g=np.random.default_rng(20260925+lead+block);starts=g.integers(0,n,(4000,int(np.ceil(n/block))))
 return ((starts[:,:,None]+np.arange(block))%n).reshape(4000,-1)[:,:n]
def checks():
 ids=indices(1639,7,90);assert ids.shape==(4000,1639) and np.array_equal(ids,indices(1639,7,90))
 a=np.ones(1639);b=np.full(1639,4.);assert np.all(np.sqrt(b[ids].mean(1))-np.sqrt(a[ids].mean(1))==1)
 assert np.all(np.sqrt(a[ids].mean(1))-np.sqrt(a[ids].mean(1))==0)
 save(OUT/'analysis_tests.json',dict(status='PASS',paired_identity=True,known_difference=True,deterministic=True))
def main():
 checks();prot=json.loads((OUT/'protocol.json').read_text());lock=json.loads((OUT/'checkpoint_lock.json').read_text());assert len(lock['rows'])==48
 m=json.loads(Path(prot['jobs'][0]['manifest']).read_text());ds=xr.open_dataset(m['data_path']);sst=ds.sst_mean.values.astype(np.float32);times=ds.time.values;doy=day_of_year(times)
 old=ROOT/'outputs/data_efficiency/v3_baseline_p0_20261008_r1';ref=old/'analysis/mhw_reference.npz'
 with np.load(ref) as z:clim=z['climatology'].copy();threshold=z['threshold'].copy();mask=z['mask'].copy();refdates=z['reference_time_indices'].copy()
 assert mask.sum()==661 and np.array_equal(refdates,np.arange(round(len(times)*.7)))
 save(OUT/'mhw_protocol.json',dict(reference=str(ref),sha256=sha(ref),dates=[str(times[refdates[0]]),str(times[refdates[-1]])],window=5,quantile=.9,min_duration=5,area_frac=.15,onset_tolerance=3,unchanged=True))
 rows=[];daily={};truth={};oldrows=[];gradrows=[]
 for j in prot['jobs']:
  ev=Path(j['evaluation_dir']);result=json.loads((ev/'metrics.json').read_text());assert result['status']=='completed'
  frozen=next(x for x in lock['rows'] if x['id']==j['id']);assert result['checkpoint_sha256']==frozen['checkpoint_sha256']==sha(Path(j['run_dir'])/'best.pt')
  resultfields=result['fields'];path=Path(resultfields['test']['path']);assert sha(path)==resultfields['test']['sha256']
  with np.load(path) as z:
   pred=z['pred_raw'][:,0];obs=z['target_raw'][:,0];idx=z['target_index'];counts=z['daily_count'];sse=z['daily_sse']
   assert np.all(counts==661) and np.all(np.diff(idx)==1)
   assert np.array_equal(z['mask'][:,0],np.broadcast_to(mask,obs.shape));np.testing.assert_array_equal(obs[:,mask>0],sst[idx][:,mask>0]);assert np.isfinite(pred).all()
   signatures=[hashlib.sha256(z[k].tobytes()).hexdigest() for k in ['target_index','target_raw','mask']]
   if j['lead'] in truth:assert truth[j['lead']]==signatures
   else:truth[j['lead']]=signatures
   key=(j['ratio'],j['lead'],j['seed'],j['arm']);daily[key]=(sse.copy(),counts.copy())
   assert abs(float(np.sqrt(sse.sum()/counts.sum()))-result['metrics']['test']['rmse'])<1e-12
   native=row_from_arrays(root_label='fixed001',run_id=j['name'],group=j['arm'],model=j['arm'],model_label=j['arm'],lead=j['lead'],history=30,seed=j['seed'],alpha=None,pred=pred,obs=obs,mask2d=mask,target_idx=idx,doy=doy,clim=clim,thresh=threshold,args=SimpleNamespace(min_duration=5,area_frac=.15,onset_tolerance=3))
   pb=mhw_binary_field(pred,doy[idx],threshold,5)&(mask>0);ob=mhw_binary_field(obs,doy[idx],threshold,5)&(mask>0);skill=contingency_skill(pb,ob,mask)
   for name in ['csi','pod','far']:assert abs(getattr(skill,name)-native['mhw_grid_'+name])<1e-12
   np.savez_compressed(ev/'mhw_fields.npz',pred_mhw=pb,obs_mhw=ob,target_index=idx,target_dates=times[idx],mask=mask)
   save(ev/'mhw_metrics.json',native)
   prior=old/f"runs/ratio_{j['ratio']}/lead{j['lead']}_seed{j['seed']}"/('baseline' if j['arm']=='baseline' else 'p0')
   oc=json.loads((prior/'config.json').read_text());nc=json.loads((Path(j['run_dir'])/'config.json').read_text());settings_diff={k:[v,nc['settings'].get(k)] for k,v in oc['settings'].items() if nc['settings'].get(k)!=v}
   evold=dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['arm'],same_parameters=oc['parameters']==nc['parameters'],same_shared_initial=oc['shared_initial_sha256']==nc['shared_initial_sha256'],same_statistics=oc['stats']==nc['stats'],same_split=oc['split_sha256']==nc['split_sha256'],setting_differences=json.dumps(settings_diff))
   if j['arm']=='baseline':
    with np.load(prior/'test_fields.npz') as zold:
     np.testing.assert_array_equal(zold['target_index'],idx);evold['prediction_max_abs_diff_K']=float(np.max(np.abs(zold['pred_raw'][:,0].astype(float)-pred)))
   else:assert evold['same_parameters'] and evold['same_shared_initial'] and evold['same_statistics'] and evold['same_split'] and set(settings_diff)<= {'lambda_physics','output_root'}
   oldrows.append(evold)
  mm=result['metrics'];status=json.loads((Path(j['run_dir'])/'status.json').read_text())
  row=dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['arm'],train_rmse=mm['train']['rmse'],validation_rmse=mm['val']['rmse'],rmse=mm['test']['rmse'],mae=mm['test']['mae'],bias=mm['test']['bias'],acc=native['test_acc'],csi=native['mhw_grid_csi'],pod=native['mhw_grid_pod'],far=native['mhw_grid_far'],test_train_gap=mm['test']['rmse']-mm['train']['rmse'],val_train_gap=mm['val']['rmse']-mm['train']['rmse'],test_val_gap=mm['test']['rmse']-mm['val']['rmse'],best_update=frozen['best_epoch']*478,end_update=status['steps'],reused_checkpoint=j['reuse'],reused_test_fields=resultfields['test']['reused'])
  rows.append(row)
  dg=OUT/'smoke'/j['name']/'initial_objective_gradients.json';diagnostic=json.loads(dg.read_text());manifest=json.loads(Path(j['manifest']).read_text())
  if j['arm']=='physics_only':
   phys=diagnostic['objectives']['physics'];gradrows.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],effective_epoch1_lambda=phys['weight'],weighted_gradient_ratio=phys['weighted_gradient_ratio'],cosine_to_data=phys['cosine_to_data'],data_gradient_norm=diagnostic['objectives']['data']['shared_gradient_norm'],physics_gradient_norm=phys['shared_gradient_norm'],target_std=manifest['statistics']['target'][1],rhs_scale=manifest['physics_rhs_scale'],scope='same first16 unshuffled train windows per pair; initial parameters, no update'))
 lookup={(r['ratio'],r['lead'],r['seed'],r['model']):r for r in rows};pairs=[];groups=[];conditions=[]
 for ratio in RATIOS:
  for lead in LEADS:
   for seed in SEEDS:
    b=lookup[ratio,lead,seed,'baseline'];p=lookup[ratio,lead,seed,'physics_only']
    pairs.append(dict(ratio=ratio,lead=lead,seed=seed,baseline_rmse=b['rmse'],physics_rmse=p['rmse'],improvement_percent=100*(b['rmse']-p['rmse'])/b['rmse'],delta_rmse=p['rmse']-b['rmse'],val_improvement_percent=100*(b['validation_rmse']-p['validation_rmse'])/b['validation_rmse'],gap_delta=p['test_train_gap']-b['test_train_gap'],train_error_delta=p['train_rmse']-b['train_rmse'],csi_delta=p['csi']-b['csi'],pod_delta=p['pod']-b['pod'],far_delta=p['far']-b['far']))
   rs=[r for r in pairs if r['ratio']==ratio and r['lead']==lead]
   cond=dict(ratio=ratio,lead=lead,improved_seeds=sum(r['delta_rmse']<0 for r in rs))
   for key in ['improvement_percent','delta_rmse','val_improvement_percent','gap_delta','train_error_delta','csi_delta','pod_delta','far_delta']:
    x=np.array([r[key] for r in rs]);cond[key+'_mean']=float(x.mean());cond[key+'_sd']=float(x.std(ddof=1))
   for arm in ARMS:
    rr=[lookup[ratio,lead,s,arm] for s in SEEDS];g=dict(ratio=ratio,lead=lead,model=arm)
    for key in ['rmse','mae','bias','acc','csi','pod','far','train_rmse','validation_rmse','test_train_gap','val_train_gap','test_val_gap']:
     x=np.array([r[key] for r in rr]);g[key+'_mean']=float(x.mean());g[key+'_sd']=float(x.std(ddof=1))
    groups.append(g);cond[arm+'_rmse_mean']=g['rmse_mean'];cond[arm+'_rmse_sd']=g['rmse_sd']
   conditions.append(cond)
 boots=[];contrasts=[];replicates={}
 for lead in LEADS:
  n=len(daily[10,lead,42,'baseline'][0])
  for block in [30,90,180]:
   ix=indices(n,lead,block);improvements={}
   for ratio in RATIOS:
    b=[];p=[]
    for seed in SEEDS:
     sse,count=daily[ratio,lead,seed,'baseline'];b.append(np.sqrt(sse[ix].sum(1)/count[ix].sum(1)))
     sse,count=daily[ratio,lead,seed,'physics_only'];p.append(np.sqrt(sse[ix].sum(1)/count[ix].sum(1)))
    b=np.array(b);p=np.array(p);delta=p-b;imp=100*(b-p)/b;improvements[ratio]=imp
    for label,d,im in [(str(s),delta[i],imp[i]) for i,s in enumerate(SEEDS)]+[('fixed3_mean',delta.mean(0),imp.mean(0))]:
     lo,hi=np.quantile(d,[.025,.975]);il,ih=np.quantile(im,[.025,.975]);boots.append(dict(ratio=ratio,lead=lead,seed_scope=label,block_days=block,replicates=4000,delta_ci_low=float(lo),delta_ci_high=float(hi),ci_includes_zero=bool(lo<=0<=hi),improvement_ci_low=float(il),improvement_ci_high=float(ih)))
     key=f'r{ratio}_l{lead}_seed{label}_b{block}';replicates[key+'_delta']=d;replicates[key+'_improvement']=im
   difference=improvements[10]-improvements[100]
   for label,x in [(str(s),difference[i]) for i,s in enumerate(SEEDS)]+[('fixed3_mean',difference.mean(0))]:
    lo,hi=np.quantile(x,[.025,.975]);contrasts.append(dict(lead=lead,seed_scope=label,block_days=block,definition='Improvement10 minus Improvement100 percentage points',ci_low=float(lo),ci_high=float(hi),ci_includes_zero=bool(lo<=0<=hi)))
    replicates[f'scarcity_l{lead}_seed{label}_b{block}']=x
 np.savez_compressed(OUT/'bootstrap_replicates.npz',**replicates)
 trends=[]
 for lead in LEADS:
  for seed in SEEDS+['mean']:
   values=[next(c['improvement_percent_mean'] for c in conditions if c['ratio']==r and c['lead']==lead) if seed=='mean' else next(p['improvement_percent'] for p in pairs if p['ratio']==r and p['lead']==lead and p['seed']==seed) for r in RATIOS]
   trends.append(dict(lead=lead,seed_scope=seed,improvement10=values[0],improvement25=values[1],improvement50=values[2],improvement100=values[3],monotonic_decrease_with_ratio=bool(np.all(np.diff(values)<=0)),improvement10_minus100=values[0]-values[3]))
 oldsummary=list(csv.DictReader((old/'analysis/paired_summary.csv').open()));oldcompare=[]
 for cond in conditions:
  z=next(r for r in oldsummary if int(r['ratio'])==cond['ratio'] and int(r['lead'])==cond['lead'])
  oldcompare.append(dict(ratio=cond['ratio'],lead=cond['lead'],lambda01_improvement=float(z['improvement_mean']),lambda001_improvement=cond['improvement_percent_mean'],difference_percentage_points=cond['improvement_percent_mean']-float(z['improvement_mean'])))
 for name,data in [('test_results.csv',rows),('seed_improvements.csv',pairs),('model_summary.csv',groups),('ratio_lead_summary.csv',conditions),('bootstrap_results.csv',boots),('scarcity_bootstrap.csv',contrasts),('scarcity_trends.csv',trends),('initial_gradient_diagnostics.csv',gradrows),('lambda_comparison.csv',oldcompare),('prior_configuration_audit.csv',oldrows)]:write_csv(OUT/name,data)
 result=dict(status='COMPLETE',conditions=conditions,groups=groups,seed_pairs=pairs,bootstrap=boots,scarcity_contrasts=contrasts,trends=trends,lambda_comparison=oldcompare,scope='exploratory replication; no Test retuning; fixed3-seed conditional temporal uncertainty',reused_checkpoints=prot['reuse_count'],new_training=prot['new_count'])
 save(OUT/'results.json',result);plots(result);reports(result,prot)
 ds.close()
 # Artifact gate: required48 fields, metrics,5 figures, unchanged core/manifests/checkpoints.
 assert len(rows)==48 and len(boots)==96
 for j in prot['jobs']:assert sha(j['manifest'])==j['manifest_sha256']
 for name,h in prot['code_sha256'].items():assert sha(ROOT/name)==h
 for name in ['figure1_rmse','figure2_improvement','figure3_bootstrap','figure4_gap','figure5_sst_mhw']:
  for ext in ['png','pdf']:assert (OUT/'figures'/f'{name}.{ext}').stat().st_size>1000
 save(OUT/'final_integrity.json',dict(status='PASS',checkpoints=48,evaluations=48,figures=5,bootstrap_rows=96,no_test_reselection=True,code_sha256=sha(Path(__file__))))
 from datetime import datetime
 from zoneinfo import ZoneInfo
 with (ROOT/'docs/research_log.md').open('a') as f:f.write(f"\n- {datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')} — Fixedλ=.01資料量/Lead48条件完了:新規{prot['new_count']},再利用{prot['reuse_count']}; Test平均改善{sum(c['improvement_percent_mean']>0 for c in conditions)}/8条件。90日4000回bootstrap/30・180日感度/気候値診断/MHW/5図/8RQ報告作成。既存Testの探索的再現評価。outputs/v3_physics_ratio_lead_lambda001_20261009/report.md。\n")
 print('ALL48 ANALYSIS COMPLETE',flush=True)

def plots(r):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'savefig.dpi':220,'pdf.fonttype':42,'font.size':10})
 groups=r['groups'];conditions=r['conditions'];pairs=r['seed_pairs'];figdir=OUT/'figures'
 def export(fig,name):fig.tight_layout();fig.savefig(figdir/(name+'.png'));fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,lead in zip(axes,LEADS):
  for arm in ARMS:
   rr=[g for g in groups if g['lead']==lead and g['model']==arm];ax.errorbar(RATIOS,[g['rmse_mean'] for g in rr],yerr=[g['rmse_sd'] for g in rr],marker='o',capsize=3,label=arm)
  ax.set(xticks=RATIOS,xlabel='Training subset (%)',title=f'Lead {lead}; mean ± seed SD');ax.legend()
 axes[0].set_ylabel('Test SST RMSE (K)');export(fig,'figure1_rmse')
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,lead in zip(axes,LEADS):
  for seed in SEEDS:
   rr=[p for p in pairs if p['lead']==lead and p['seed']==seed];ax.plot(RATIOS,[p['improvement_percent'] for p in rr],ls=':',alpha=.6,label=f'seed {seed}')
  rr=[c for c in conditions if c['lead']==lead];ax.plot(RATIOS,[c['improvement_percent_mean'] for c in rr],marker='o',lw=2,color='black',label='paired mean')
  ax.axhline(0,color='grey',lw=.8);ax.set(xticks=RATIOS,xlabel='Training subset (%)',title=f'Lead {lead}');ax.legend(fontsize=8)
 axes[0].set_ylabel('Test RMSE improvement (%)');export(fig,'figure2_improvement')
 fig,ax=plt.subplots(figsize=(8,5))
 for i,c in enumerate(conditions):
  b=next(x for x in r['bootstrap'] if x['ratio']==c['ratio'] and x['lead']==c['lead'] and x['block_days']==90 and x['seed_scope']=='fixed3_mean')
  ax.plot([b['delta_ci_low'],b['delta_ci_high']],[i,i],lw=2);ax.plot(c['delta_rmse_mean'],i,'ko')
 ax.axvline(0,color='black',ls='--');ax.set(yticks=np.arange(8),yticklabels=[f"{c['ratio']}% / Lead {c['lead']}" for c in conditions],xlabel='Physics − Baseline RMSE (K)',title='Conditional95% CI; paired90-day blocks,4000 draws');export(fig,'figure3_bootstrap')
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,lead in zip(axes,LEADS):
  for arm in ARMS:
   rr=[g for g in groups if g['lead']==lead and g['model']==arm];ax.plot(RATIOS,[g['test_train_gap_mean'] for g in rr],marker='o',label=arm)
  ax.set(xticks=RATIOS,xlabel='Training subset (%)',title=f'Lead {lead}');ax.legend()
 axes[0].set_ylabel('Fixed-checkpoint Test − Train RMSE (K)');export(fig,'figure4_gap')
 fig,ax=plt.subplots(figsize=(9,5));v=np.array([[c['improvement_percent_mean'],100*c['csi_delta_mean'],100*c['pod_delta_mean'],-100*c['far_delta_mean']] for c in conditions]);im=ax.imshow(v,cmap='RdBu',vmin=-max(abs(v.min()),abs(v.max()),.01),vmax=max(abs(v.min()),abs(v.max()),.01),aspect='auto')
 ax.set(xticks=np.arange(4),xticklabels=['RMSE reduction (%)','ΔCSI (pp)','ΔPOD (pp)','−ΔFAR (pp)'],yticks=np.arange(8),yticklabels=[f"{c['ratio']}% / Lead {c['lead']}" for c in conditions],title='Positive favors Physics; SST and detection assessed separately')
 for i in range(8):
  for j in range(4):ax.text(j,i,f'{v[i,j]:+.2f}',ha='center',va='center',fontsize=9)
 fig.colorbar(im,ax=ax);export(fig,'figure5_sst_mhw');save(figdir/'source_data.json',r)

def reports(r,prot):
 primary=[b for b in r['bootstrap'] if b['block_days']==90 and b['seed_scope']=='fixed3_mean']
 lines=['# Fixed λphys=.01 Physics-only — ratio/lead exploratory replication','', '48conditions; ratios10/25/50/100%, leads7/14, seeds42/7/123, identical native all-head Baseline/Physics-only, auxiliary0. 100% is3995windows, not Full training. Selected only by Validation RMSE,478-update interval,80-block max,patience12; actual end updates may differ.','', '## Ratio/Lead SST performance','', 'RMSE SD is sample SD across3 seeds; improvement is mean of paired seed percentages.']+table(r['conditions'],['ratio','lead','baseline_rmse_mean','baseline_rmse_sd','physics_only_rmse_mean','physics_only_rmse_sd','improvement_percent_mean','improved_seeds'])
 lines+=['','## Temporal bootstrap','', '90-day paired circular primary;30/180 sensitivity,4000replicates. Common date indices resampled across both models, all ratios and fixed seeds within each lead; all ocean cells retained jointly. Δ=Physics−Baseline. CI conditional on fitted seeds, not training-seed population. No multiplicity correction; CI including0 is not equivalence.']+table(primary,['ratio','lead','delta_ci_low','delta_ci_high','ci_includes_zero'])
 lines+=['','## Fixed-checkpoint generalization and SST/MHW metrics','']+table(r['groups'],['ratio','lead','model','train_rmse_mean','validation_rmse_mean','rmse_mean','test_train_gap_mean','mae_mean','bias_mean','acc_mean','csi_mean','pod_mean','far_mean'])
 lines+=['','## Scarcity hypothesis','', 'A decrease of improvement as ratio increases supports the directional H1 within this setting. Lower absolute RMSE with more data alone is not evidence of a larger Physics benefit.']+table(r['trends'],['lead','seed_scope','improvement10','improvement25','improvement50','improvement100','monotonic_decrease_with_ratio','improvement10_minus100'])
 lines+=['','## Paired10%−100% improvement contrast (percentage points)','']+table([x for x in r['scarcity_contrasts'] if x['block_days']==90],['lead','seed_scope','ci_low','ci_high','ci_includes_zero'])
 lines+=['','## Previous λ=.1 comparison','']+table(r['lambda_comparison'],['ratio','lead','lambda01_improvement','lambda001_improvement','difference_percentage_points'])
 lines+=['','## Data and gradient confounders','', 'climatology_diagnostics.csv records unique input dates, empty/singleton366-calendar bins, each anomaly feature Train/Validation/Test std and |z|>5/10/20 rates. Distribution scan reused from verified identical v2 anomaly/statistics artifacts; raw anomaly std derived from z std × feature normalization std, counting actual history occurrences. v3 changes only coastal RHS vs v2. Empty bins use periodic subset-only interpolation; no Full fallback. Feb29 follows leap-aligned2000 month/day bins. Finite heavy tails retained. initial_gradient_diagnostics.csv reports first16 selected windows, epoch1 effectiveλ=.002, separate data/physics norms and cosine; all production training uses nominalλ=.01 with unchanged5-block warmup. A ratio effect may reflect climatology estimation, normalization and gradient balance as well as sample count.','', '## Methods and traceability','', 'run_manifest.csv lists all48 run paths/status/best/end updates; reuse_audit.json inventories all candidate checkpoints. protocol.json fixes config/manifests/source; checkpoint_lock.json fixes48 Validation-best hashes before Test. train_v3_ablation.py:77 unchanged worker; train_v3_study.py:104 native setup; src/mhw/subset_preprocessing.py:170 manifest loader; src/mhw/subset_climatology.py:117 subset anomaly loader. src/train_jeju_hilam.py:417 collect_predictions and447 target inverse transform. src/summarize_jeju_hilam_mhw_skill.py:107 MHW metrics; src/mhw/mhw_skill.py:128 binary events. MHW uses unchanged first70% raw dates, q.9,±5DOY,≥5consecutive days and native leap convention. New evaluation outputs hold fields or verified links to identical prior predictions. Train RMSE reevaluated from frozen checkpoint, not online logs. All source scripts: prepare_ratio001.py, preflight_ratio001.py, ratio001_worker.py, manage_ratio001.py, evaluate_ratio001.py, analyze_ratio001.py.','', '## Figures','']
 for name in ['figure1_rmse','figure2_improvement','figure3_bootstrap','figure4_gap','figure5_sst_mhw']:lines.append(f'- [{name}](figures/{name}.png), [PDF](figures/{name}.pdf)')
 lines+=['','CSV originals: test_results.csv, model_summary.csv, seed_improvements.csv, bootstrap_results.csv, scarcity_bootstrap.csv, scarcity_trends.csv, initial_gradient_diagnostics.csv, climatology_diagnostics.csv, lambda_comparison.csv. Figure data also in figures/source_data.json. No separate unused future period exists inside the fixed1993–2022 source after the2022-12-31 Test endpoint; no split changes made.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 cond=r['conditions'];n=sum(x['improvement_percent_mean']>0 for x in cond);support=sum(x['delta_ci_high']<0 for x in primary);uniform=sum(x['improved_seeds']==3 for x in cond)
 text=['# Interpretation — eight research questions','',f'RQ1: Test mean paired Physics improvement occurs in {n}/8conditions. Values and negative results are retained in ratio_lead_summary.csv.', '', 'RQ2: Scarcity trend must be assessed separately by lead and seed; monotonic flags and matched10−100 CI in scarcity_trends.csv/scarcity_bootstrap.csv. Nonmonotonic patterns do not establish a general data-scarcity advantage.']
 for lead in LEADS:
  tr=next(x for x in r['trends'] if x['lead']==lead and x['seed_scope']=='mean');text.append(f"Lead{lead}: improvements10/25/50/100={tr['improvement10']:+.3f}/{tr['improvement25']:+.3f}/{tr['improvement50']:+.3f}/{tr['improvement100']:+.3f}%; monotonic decreasing={tr['monotonic_decrease_with_ratio']};10−100={tr['improvement10_minus100']:+.3f}pp.")
 text += ['', 'RQ3: Compare the lead-specific patterns above; no unsupported population-level lead interaction claim.', '',f'RQ4: All3seeds improve in {uniform}/8conditions; seed_improvements.csv preserves each direction.', '', f'RQ5: Primary conditional90-day CI entirely below0 in {support}/8conditions; per-seed and30/180-day sensitivity are in bootstrap_results.csv. Seed SD and temporal uncertainty are distinct.','', 'RQ6: Ratio changes input-date coverage, supported DOY density, anomaly variance and normalization/RHS scales. See climatology_diagnostics.csv and initial_gradient_diagnostics.csv; do not attribute all ratio differences solely to sample count.','', 'RQ7: SST improvement and MHW detection are separate. The following table preserves detection tradeoffs; positiveCSI/POD and negativeFAR are favorable.']+table(cond,['ratio','lead','improvement_percent_mean','csi_delta_mean','pod_delta_mean','far_delta_mean'])
 text += ['', 'RQ8: lambda_comparison.csv contrasts.1/.01; prior_configuration_audit.csv verifies P0 settings differ only by nominallambda/output path, but prior Baseline lacked inactive heads (numerical prediction differences separately reported). Both use unchanged warmup. Prior Test inspection motivated.01; results are exploratory, not independent confirmation. No Test-based tuning or checkpoint reselection.','', 'Generalization gap: Test−Train decreases can combine increased Train error and reduced Test error; seed_improvements.csv explicitly separates both. Date/distribution differences also affect gaps, so gap reduction is not an isolated causal measure of overfitting.']
 (OUT/'interpretation.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
