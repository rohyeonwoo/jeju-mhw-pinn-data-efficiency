"""Analysis of frozen Oracle120 outputs only. No torch, training or checkpoint loading."""
import os,sys,json,hashlib,datetime
from pathlib import Path
R=Path('/ceph_data/yeonwoo/MHW_subset'); O=R/'outputs/paper_analysis_20261016'; V=R/'outputs/paper_final_verification_20261015'; T=R/'outputs/v8_actual_epoch_oracle96_20261011'; A=R/'outputs/v8_actual_epoch_auxiliary24_20261015'
for k in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR']: os.environ[k]=str(O/'runtime')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xarray as xr
from zoneinfo import ZoneInfo
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':350,'axes.spines.top':False,'axes.spines.right':False})
MODELS=['Baseline','Auxiliary','P0','NoAdv','Adv']; LABEL={'Baseline':'Baseline','Auxiliary':'Auxiliary-only','P0':'P0','NoAdv':'NoAdv','Adv':'Adv'}
COL=dict(zip(MODELS,['#555555','#CC79A7','#E69F00','#0072B2','#009E73'])); MARK=dict(zip(MODELS,['o','s','^','D','v'])); RAT=[10,25,50,100]; SEEDS=[42,7,123]
manifest=[];sources={};J=lambda p:json.loads(Path(p).read_text())
def hashfile(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source(p):
 p=Path(p);key=str(p)
 if key not in sources:sources[key]={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':hashfile(p)}
 return key
def readcsv(p):source(p);return pd.read_csv(p)
def csv(d,p):d.to_csv(O/p,index=False)
def write(p,s):(O/p).write_text(s,encoding='utf-8')
def table(d,name,caption):
 csv(d,'tables/'+name+'.csv')
 write('tables/'+name+'.tex',d.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.5f}',caption=caption,label='tab:'+name,longtable=len(d)>45))
 manifest.append(dict(id=name,kind='table',artifact='tables/'+name+'.tex',plot_data='tables/'+name+'.csv',caption=caption,models=';'.join(MODELS),sources=';'.join(sources),analysis_code='code/analyze.py'))
def fig(f,name,data,caption,supp=False,source_paths=None):
 base='figures/'+('supplementary/' if supp else 'main/')+name
 for ext in ['png','pdf','svg']: f.savefig(O/(base+'.'+ext),bbox_inches='tight',dpi=350)
 plt.close(f); csv(data,'plot_data/'+name+'.csv'); write('captions/'+name+'.md',caption+'\n')
 manifest.append(dict(id=name,kind='figure',artifact=base+'.png;'+base+'.pdf;'+base+'.svg',plot_data='plot_data/'+name+'.csv',caption=caption,models=';'.join(MODELS),sources=';'.join(map(str,source_paths or sources)),analysis_code='code/analyze.py'))
def mean_sd(x):return f'{x.mean():.4f} ± {x.std(ddof=1):.4f}'
def markdown_frame(frame,index=True,**kwargs):
    data=frame.reset_index() if index else frame
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}'
        return str(v).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(str,data.columns))+' |','| '+' | '.join(['---']*len(data.columns))+' |']
    rows.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in data.itertuples(index=False,name=None))
    return '\n'.join(rows)
# Local Markdown serializer avoids modifying the Python environment for tabulate.
pd.DataFrame.to_markdown=markdown_frame

def savejson(p,x):write(p,json.dumps(x,indent=2,ensure_ascii=False,default=str))
def log(s):print(datetime.datetime.now().isoformat(timespec='seconds'),s,flush=True)
log('Read existing summaries; do not rerun final verification')
d=readcsv(V/'test_results120.csv');protocol=readcsv(V/'protocol120.csv');bs=readcsv(V/'block_bootstrap.csv');val=readcsv(A/'all_models_validation_by_seed.csv');diag=readcsv(V/'aux_diagnostics_rechecked.csv');ep=readcsv(A/'epoch_metrics.csv')
val['model']=val['model'].replace({'Auxiliary-only':'Auxiliary'})
for p in ['analysis_status.json','mhw_reference_check.json','verification_matrix.csv','dataset_metadata.json','model_definition_audit.md','experiment_protocol_audit.md','horizontal_advection_audit.md','auxiliary_ablation_audit.md','mhw_evaluation_audit.md','physics_equation_audit.md','stencil_tests.csv','physics_numerical_checks.json','manifest_checks.json']:source(V/p)
expected={(r,l,m,s) for r in RAT for l in [7,14] for m in MODELS for s in SEEDS}
assert set(map(tuple,d[['ratio','lead','model','seed']].values))==expected and len(d)==120
assert len(protocol)==120 and (protocol.epochs==80).all()
assert set(map(tuple,val[['ratio','lead','model','seed']].values))==expected and len(val)==120
assert np.allclose(val.merge(protocol,on=['ratio','lead','model','seed']).best_val_rmse,val.merge(protocol,on=['ratio','lead','model','seed']).val_rmse)
val_mae=[]
for row in val.itertuples():
 history=readcsv(Path(row.run_dir)/'history.csv'); best=history[history.epoch==row.best_epoch];assert len(best)==1
 assert abs(best.iloc[0].val_rmse-row.best_val_rmse)<1e-10
 val_mae.append(best.iloc[0].val_mae)
val['best_val_mae']=val_mae

metrics=['rmse','mae','mhw_grid_csi','mhw_grid_pod','mhw_grid_far','mhw_grid_precision','mhw_grid_f1']
summary=d.groupby(['ratio','lead','model'])[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();csv(summary,'statistics/test_summary.csv')
base=d[d.model=='Baseline'][['ratio','lead','seed']+metrics].rename(columns={k:'baseline_'+k for k in metrics})
seeds=d.merge(base,on=['ratio','lead','seed'],validate='many_to_one');seeds['paired_rmse_improvement_pct']=100*(seeds.baseline_rmse-seeds.rmse)/seeds.baseline_rmse
for m in metrics[2:]:seeds['delta_'+m+'_vs_baseline']=seeds[m]-seeds['baseline_'+m]
csv(seeds,'tables/test_seed_metrics.csv');csv(val,'tables/validation_seed_metrics.csv')
# Six prespecified pairs: candidate minus reference except the explicitly named CSI convention.
pairs=[('Baseline','Auxiliary'),('Baseline','P0'),('Baseline','NoAdv'),('Baseline','Adv'),('Auxiliary','NoAdv'),('NoAdv','Adv')];paired=[];ps=[]
for ref,cand in pairs:
 aa=d[d.model==ref].set_index(['ratio','lead','seed']);bb=d[d.model==cand].set_index(['ratio','lead','seed']);
 for key in aa.index:
  a,b=aa.loc[key],bb.loc[key];r,l,s=key
  row=dict(ratio=r,lead=l,seed=s,reference=ref,candidate=cand,reference_rmse=a.rmse,candidate_rmse=b.rmse,delta_rmse=b.rmse-a.rmse,improvement_pct=100*(a.rmse-b.rmse)/a.rmse,delta_csi=b.mhw_grid_csi-a.mhw_grid_csi,delta_pod=b.mhw_grid_pod-a.mhw_grid_pod,delta_far=b.mhw_grid_far-a.mhw_grid_far)
  row['noadv_minus_adv_csi']=a.mhw_grid_csi-b.mhw_grid_csi if ref=='NoAdv' else np.nan;paired.append(row)
pdseeds=pd.DataFrame(paired);csv(pdseeds,'statistics/paired_seed_results.csv')
for key,g in pdseeds.groupby(['ratio','lead','reference','candidate']):
 r,l,ref,cand=key; ci=bs[(bs.ratio==r)&(bs.lead==l)&(bs.reference==ref)&(bs.candidate==cand)&(bs.metric=='rmse')&(bs.block_days==90)];assert len(ci)==1
 assert abs(ci.iloc[0].estimate-g.delta_rmse.mean())<1e-12
 ps.append(dict(ratio=r,lead=l,reference=ref,candidate=cand,delta_rmse_mean=g.delta_rmse.mean(),delta_rmse_sd=g.delta_rmse.std(ddof=1),paired_improvement_pct_mean=g.improvement_pct.mean(),paired_improvement_pct_sd=g.improvement_pct.std(ddof=1),candidate_rmse_wins=int((g.delta_rmse<0).sum()),delta_csi_mean=g.delta_csi.mean(),candidate_csi_wins=int((g.delta_csi>0).sum()),ci95_low=ci.iloc[0].ci_low,ci95_high=ci.iloc[0].ci_high,block_days=90,n_seeds=3))
ps=pd.DataFrame(ps);csv(ps,'statistics/paired_summary.csv');csv(bs[bs.block_days==90],'statistics/existing_bootstrap90.csv')
# Separate validation comparisons; do not invent missing validation MHW metrics.
vseeds=val.merge(val[val.model=='Baseline'][['ratio','lead','seed','best_val_rmse']].rename(columns={'best_val_rmse':'baseline_val_rmse'}),on=['ratio','lead','seed']);vseeds['improvement_pct']=100*(vseeds.baseline_val_rmse-vseeds.best_val_rmse)/vseeds.baseline_val_rmse
vsummary=[]
for key,g in vseeds.groupby(['ratio','lead','model']):
 r,l,m=key;vsummary.append(dict(ratio=r,lead=l,model=LABEL[m],validation_rmse=mean_sd(g.best_val_rmse),validation_mae=mean_sd(g.best_val_mae),mean_seed_improvement_pct=g.improvement_pct.mean(),improved_seeds=int((g.improvement_pct>0).sum())))
table(pd.DataFrame(vsummary),'table_validation','Validation-best SST RMSE (K); three-seed sample SD. Separate from Test.')
setup=pd.DataFrame([
('Study domain','30–37°N, 123–131°E; 29 × 33; 0.25°'),('Ocean/support cells','661 / 660'),('Data period','1993-01-01 to 2022-12-31'),('Sources','NOAA OISST; ERA5 wind/flux; GLORYS mixed-layer variables'),('History / leads','30 days / 7 and 14 days'),('Nested training windows','10%: 376; 25%: 987; 50%: 1974; 100%: 3995'),('Training blocks','8, 21, 42, 85 nonoverlapping 90-day blocks'),('100% definition','3995 retained windows; not all 7639 common training candidates'),('Seeds / conditions','42, 7, 123 / 4 ratios × 2 leads × 5 models = 120'),('Training / selection','80 actual epochs; validation-minimum checkpoint; final-epoch LR=0'),('Model','SeaCast-inspired JejuHiLAM; 15,019,779 parameters'),('Training forcing','Oracle future physical forcing in constraint loss only'),('Test targets, lead 7','2018-07-07 to 2022-12-31 (1639 days)'),('Test targets, lead 14','2018-07-08 to 2022-12-31 (1638 days)'),('MHW reference','1993-01-01 to 2013-12-31; shared across all ratios/models'),('MHW definition','DOY ±5 days; 90th percentile; ≥5 consecutive days; no gap merging'),('Uncertainty','3-seed sample SD; existing paired circular 90-day blocks, 2000 draws'),('Interpretation','Exploratory Test; fixed period and seeds; no independent confirmation')],columns=['Item','Configuration']);table(setup,'table1_setup','Dataset, study domain and Oracle120 experimental configuration.')
models=pd.DataFrame([
('Baseline','D','None','OFF','OFF'),('Auxiliary-only','D + 0.0001 U','None','OFF','ON'),('P0','D + λ(e) P0','Surface flux + advection + learned entrainment','ON','OFF'),('NoAdv','D + λ(e) PA','Flux − shortwave penetration + MLD entrainment proxy','OFF','OFF'),('Adv','D + λ(e) PB','NoAdv RHS + horizontal advection','OFF','OFF')],columns=['Model','Active loss','Physics RHS','we head active','Rres head active']);table(models,'table2_models','Five models share architecture. λ(e)=0.1 min(e/5,1); P0 and Adv differ in more than advection. U penalizes Rres toward zero.')
sst=[];mhw=[]
for key,g in seeds.groupby(['ratio','lead','model']):
 r,l,m=key;sst.append(dict(ratio=r,lead=l,model=LABEL[m],RMSE_K=mean_sd(g.rmse),MAE_K=mean_sd(g.mae),seed_mean_improvement_pct=g.paired_rmse_improvement_pct.mean(),improved_seeds=int((g.paired_rmse_improvement_pct>0).sum())))
 mhw.append(dict(ratio=r,lead=l,model=LABEL[m],**{k.replace('mhw_grid_','').upper():mean_sd(g[k]) for k in metrics[2:]}))
table(pd.DataFrame(sst),'table3_sst','Test SST metrics: mean ± sample SD of three seeds; improvement averages seedwise percentages. 100%=3995 windows; exploratory Test.')
table(pd.DataFrame(mhw),'table4_mhw','Pooled grid-day Test MHW metrics, summarized across three seeds. Shared fixed threshold and observed labels across ratios; FAR=FP/(TP+FP).')
table(ps,'table5_paired','Six paired comparisons at identical ratio/lead/seed. ΔRMSE=candidate−reference (K). CI: existing 90-day blocks conditional on fixed seeds; no multiplicity correction.')
vm=readcsv(V/'verification_matrix.csv');vm['evidence_file']=str(V/'verification_summary_ko.md');csv(vm,'tables/verification_evidence_original.csv')
verification_rows=[('Model and protocol','protocol120.csv','120 runs, 80 epochs; prior audit'),('MHW threshold','mhw_reference_check.json','Prior bitwise reference check; common evaluation threshold'),('Test metrics and alignment','analysis_status.json','Prior 120-run metric recomputation and ground-truth check'),('Stencil','stencil_tests.csv','Prior 88 tests; regular-grid scope'),('Coordinates / support','dataset_metadata.json','Actual ascending coordinates; 661 ocean / 660 support'),('Auxiliary diagnostics','aux_diagnostics_rechecked.csv','Saved diagnostic gradients; no new optimizer updates'),('Physics units','horizontal_advection_audit.md','K/day conversion and coordinate conventions; upstream limitations')]
table(pd.DataFrame([dict(item=a,evidence=str(V/b),scope=c,status='Existing evidence reused; audit not rerun') for a,b,c in verification_rows]),'table6_evidence','Existing numerical and reproducibility evidence. Status is attributed to the prior report, not a fresh full audit.')
log('Tables and paired comparisons complete; inspect frozen prediction alignment')
# Coordinates read only from data; never fit statistics or modify source data.
meta=J(V/'dataset_metadata.json'); dataset=Path(meta['path']);ds=xr.open_dataset(dataset); lat=ds.latitude.values;lon=ds.longitude.values;times=ds.time.values
YY,XX=np.meshgrid(lat,lon,indexing='ij');sea0=None;truth={};checks=[];regional=[];spatial=[];example={};case_dates={};run_sources=[]
# Descriptive ROI declared before computing this new box’s error summaries. It is a new geographic box, not a prior domain definition.
region_protocol='New exploratory SW-Jeju box: 32.0–33.25°N, 125.0–126.5°E, inclusive, intersected with the saved ocean mask. Chosen geographically south/west of Jeju (approx.33.4°N,126.5°E), before computing error summaries for this new box; not independently validated as a coastal dynamical region. Historical SW-Korea box:33.5–35°N,125.5–127.5°E. Coast: ocean cells with a cardinal land neighbor; domain exterior is not land. Case rule: for each lead choose the earliest June–August day in the final complete Test year (2022); fixed ratio100/seed42 for all models. No skill-based selection.'
write('reports/spatial_case_protocol.md',region_protocol+'\n')
for row in d.itertuples():
 folder=(V/'auxiliary_test/runs' if row.model=='Auxiliary' else T/'test_evaluation_20261012/runs')/row.name
 p=folder/'fields.npz';q=folder/'daily_metrics.npz';run_sources.extend([p,q]);assert p.is_file() and q.is_file()
 with np.load(p) as z:
  pred=np.squeeze(z['pred_raw']);obs=np.squeeze(z['target_raw']);mask=np.squeeze(z['mask']);idx=z['target_index'];dates=z['target_dates'];sea=mask[0]>0 if mask.ndim==3 else mask>0
  assert pred.shape==obs.shape==(len(idx),len(lat),len(lon));assert np.array_equal(dates,times[idx]);assert np.all(np.diff(idx)==1);assert np.all(mask==sea);assert np.isfinite(pred[:,sea]).all() and np.isfinite(obs[:,sea]).all()
  if sea0 is None:
   sea0=sea.copy();land=~sea;coast=np.zeros_like(sea);coast[1:]|=land[:-1];coast[:-1]|=land[1:];coast[:,1:]|=land[:,:-1];coast[:,:-1]|=land[:,1:];coast&=sea
   regions={'All':sea,'SW_Jeju_exploratory':sea&(YY>=32)&(YY<=33.25)&(XX>=125)&(XX<=126.5),'Coast':coast,'Offshore':sea&~coast,'Historical_SW_Korea':sea&(YY>=33.5)&(YY<=35)&(XX>=125.5)&(XX<=127.5)}
  assert np.array_equal(sea,sea0)
  with np.load(q) as zz:pm=zz['pred_mhw'].astype(bool);om=zz['obs_mhw'].astype(bool);assert np.array_equal(zz['target_index'],idx)
  if row.lead not in truth:truth[row.lead]={'obs':obs.copy(),'dates':dates.copy(),'idx':idx.copy(),'om':om.copy()}
  tr=truth[row.lead];assert np.array_equal(obs[:,sea],tr['obs'][:,sea]);assert np.array_equal(idx,tr['idx']);assert np.array_equal(om,tr['om']);assert np.array_equal(dates,tr['dates'])
  err=pred.astype('float64')-obs.astype('float64');months=pd.DatetimeIndex(dates).month.to_numpy();year=pd.DatetimeIndex(dates).year.to_numpy()
  for reg,rm in regions.items():
   for season,mos in [('ALL',range(1,13)),('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:
    sel=np.isin(months,list(mos));e=err[sel][:,rm];pp=pm[sel][:,rm];oo=om[sel][:,rm];h=int((pp&oo).sum());f=int((pp&~oo).sum());n=int((~pp&oo).sum());den=h+f+n
    regional.append(dict(ratio=row.ratio,lead=row.lead,seed=row.seed,model=row.model,region=reg,season=season,n_cells=int(rm.sum()),n_dates=int(sel.sum()),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),csi=h/den if den else 0,pod=h/(h+n) if h+n else 0,far=f/(h+f) if h+f else 0,tp=h,fp=f,fn=n))
  gridrmse=np.sqrt(np.mean(err**2,axis=0))
  for iy,ix in zip(*np.where(sea)):spatial.append(dict(ratio=row.ratio,lead=row.lead,seed=row.seed,model=row.model,latitude=lat[iy],longitude=lon[ix],rmse=gridrmse[iy,ix]))
  if row.ratio==100 and row.seed==42:
   kk=np.flatnonzero((year==2022)&np.isin(months,[6,7,8]))[0];case_dates[row.lead]=str(dates[kk])[:10]
   example[(row.lead,row.model)]={'pred':pred[kk].copy(),'obs':obs[kk].copy(),'pm':pm[kk].copy(),'om':om[kk].copy(),'date':str(dates[kk])[:10]}
  checks.append(dict(name=row.name,ratio=row.ratio,lead=row.lead,seed=row.seed,model=row.model,field_path=str(p),daily_path=str(q),field_bytes=p.stat().st_size,field_mtime_ns=p.stat().st_mtime_ns,source_fields_sha256_prior=row.source_fields_sha256,n_targets=len(idx),n_ocean=int(sea.sum()),date_alignment='checked',target_mask_label_pairing='checked',metric_audit='reused prior audit; not rerun'))
 if len(checks)%20==0:log(f'Input alignment and regional summaries: {len(checks)}/120')
reg=pd.DataFrame(regional);sp=pd.DataFrame(spatial);csv(reg,'plot_data/regional_seasonal_seed_metrics.csv');csv(sp,'plot_data/grid_rmse_seed.csv');csv(pd.DataFrame(checks),'reports/input_completeness.csv')
region_counts=pd.DataFrame([dict(region=k,n_cells=int(v.sum()),definition=region_protocol if k=='SW_Jeju_exploratory' else k) for k,v in regions.items()]);csv(region_counts,'tables/region_definitions.csv')
# Pointwise and seasonal paired differences always defined positive in favor of NoAdv.
keys=['ratio','lead','seed','latitude','longitude'];sd=sp[sp.model=='Adv'].merge(sp[sp.model=='NoAdv'],on=keys,suffixes=('_adv','_noadv'));sd['delta_rmse_adv_minus_noadv']=sd.rmse_adv-sd.rmse_noadv;csv(sd,'plot_data/grid_delta_adv_minus_noadv.csv')
rkeys=['ratio','lead','seed','region','season'];rd=reg[reg.model=='Adv'].merge(reg[reg.model=='NoAdv'],on=rkeys,suffixes=('_adv','_noadv'));rd['delta_rmse_adv_minus_noadv']=rd.rmse_adv-rd.rmse_noadv;rd['delta_csi_noadv_minus_adv']=rd.csi_noadv-rd.csi_adv
for k in ['tp','fp','fn']:rd['delta_'+k+'_noadv_minus_adv']=rd[k+'_noadv']-rd[k+'_adv']
csv(rd,'statistics/region_season_paired.csv');rs=rd.groupby(['ratio','lead','region','season'])[['delta_rmse_adv_minus_noadv','delta_csi_noadv_minus_adv','delta_tp_noadv_minus_adv','delta_fp_noadv_minus_adv','delta_fn_noadv_minus_adv']].agg(['mean','std']);rs.columns=['_'.join(c) for c in rs.columns];csv(rs.reset_index(),'tables/region_season_paired_summary.csv')
log('Analyze observed-field advection diagnostic on saved Test dates (not interpolated training-path terms)')
# NumPy equivalent of the documented ocean-neighbor finite difference on this regular grid.
def neighbor(a):
 left=np.zeros_like(a);right=np.zeros_like(a);up=np.zeros_like(a);down=np.zeros_like(a)
 left[...,1:]=a[...,:-1];right[...,:-1]=a[...,1:];up[...,1:,:]=a[...,:-1,:];down[...,:-1,:]=a[...,1:,:]
 return left,right,up,down
ml,mr,mu,md=neighbor(sea0);support=sea0&(ml|mr)&(mu|md);dx=6371000*np.cos(np.deg2rad(YY))*np.gradient(np.deg2rad(XX),axis=1);dy=6371000*np.gradient(np.deg2rad(YY),axis=0)
fdir=R/'outputs/v7_gpu_mixed_budget_comparison_20261011/features';source(fdir/'manifest.json')
u=np.load(fdir/'u.npy',mmap_mode='r');v=np.load(fdir/'v.npy',mmap_mode='r');valid=np.load(fdir/'valid.npy',mmap_mode='r');advs={};phrows=[];histrows=[];advmap=[];maprows=[]
for lead,tr in truth.items():
 field=np.where(sea0,tr['obs'],0).astype(float);left,right,up,down=neighbor(field)
 gx=np.where(ml&mr,(right-left)*.5,np.where(mr,right-field,np.where(ml,field-left,0)))/dx
 gy=np.where(mu&md,(down-up)*.5,np.where(md,down-field,np.where(mu,field-up,0)))/dy
 idx=tr['idx'];good=support&(valid[idx]>0);adv=np.where(good,-86400*(u[idx]*gx+v[idx]*gy),np.nan);advs[lead]=adv
 months=pd.DatetimeIndex(tr['dates']).month.to_numpy()
 for region,mask in regions.items():
  for season,mos in [('ALL',range(1,13)),('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:
   vals=adv[np.isin(months,list(mos))][:,mask];vals=vals[np.isfinite(vals)]
   phrows.append(dict(lead=lead,region=region,season=season,n_cell_days=len(vals),mean_kday=float(vals.mean()),absmean_kday=float(abs(vals).mean()),rms_kday=float(np.sqrt(np.mean(vals**2))),p01_kday=np.quantile(vals,.01),p50_kday=np.quantile(vals,.5),p99_kday=np.quantile(vals,.99),diagnostic='observed target-day SST; observed mixed-layer velocity; not training interpolation'))
  vals=adv[:,mask];vals=vals[np.isfinite(vals)];counts,bins=np.histogram(vals,bins=np.linspace(-2,2,101));under=int((vals< -2).sum());over=int((vals>2).sum())
  for lo,hi,n in zip(bins[:-1],bins[1:],counts):histrows.append(dict(lead=lead,region=region,bin_left=lo,bin_right=hi,count=n,n_total=len(vals),below_minus2=under,above_plus2=over))
 cnt=np.isfinite(adv).sum(axis=0);mag=np.sqrt(np.divide(np.nansum(adv**2,axis=0),cnt,out=np.full(sea0.shape,np.nan),where=cnt>0));
 for iy,ix in zip(*np.where(support)):advmap.append(dict(lead=lead,latitude=lat[iy],longitude=lon[ix],advection_rms_kday=mag[iy,ix]))
ph=pd.DataFrame(phrows);ah=pd.DataFrame(histrows);am=pd.DataFrame(advmap);csv(ph,'statistics/observed_advection_season_region.csv');csv(ah,'plot_data/advection_histogram.csv');csv(am,'plot_data/observed_advection_rms_map.csv')
assoc=sd.groupby(['ratio','lead','latitude','longitude']).delta_rmse_adv_minus_noadv.mean().reset_index().merge(am,on=['lead','latitude','longitude']);csv(assoc,'plot_data/advection_spatial_association.csv')
for key,g in assoc.groupby(['ratio','lead']):maprows.append(dict(ratio=key[0],lead=key[1],descriptive_spatial_pearson=g.advection_rms_kday.corr(g.delta_rmse_adv_minus_noadv),n_cells=len(g),inference='No p-value; spatially dependent cells; association not causality'))
csv(pd.DataFrame(maprows),'statistics/advection_spatial_correlations.csv')
# Previously computed physical-residual/SST daily correlations, matched definitions A and B.
resid=d[['ratio','lead','seed','model','rmse','A_residual_rms_kday','B_residual_rms_kday','A_daily_rmse_correlation','B_daily_rmse_correlation']];csv(resid,'statistics/residual_error_existing.csv')
log('Render main and supplementary figures')
# Figure 1: code-native experimental design, no generated raster assets.
f=plt.figure(figsize=(11,5));gs=f.add_gridspec(1,2,width_ratios=[1,1.65]);ax=f.add_subplot(gs[0]);ax.pcolormesh(lon,lat,np.where(sea0,1,np.nan),cmap='Blues',vmin=0,vmax=2,shading='nearest');ax.contour(lon,lat,sea0,levels=[.5],colors='.4',linewidths=.5);ax.plot(126.5,33.4,'k*',ms=8);ax.annotate('Jeju',(126.5,33.4),(127,34));ax.add_patch(Rectangle((125,32),1.5,1.25,fill=False,color='#CC79A7',ls='--'));ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)',title='(a) Saved study grid: 661 ocean cells');ax.set_aspect(1/np.cos(np.deg2rad(33.5)))
ax=f.add_subplot(gs[1]);ax.axis('off');texts=[('Data and split','1993–2022 daily SST + physical fields\n30-day history; fixed Validation and Test periods'),('Nested training subsets','10%: 376  ⊂  25%: 987  ⊂  50%: 1,974  ⊂  100%: 3,995 windows\n90-day blocks; subset-specific training statistics'),('Five matched models','Baseline   |   Auxiliary-only   |   P0   |   NoAdv   |   Adv\nSame architecture; identical ratio / lead / seed pairing'),('Training and evaluation','2 leads (7, 14 days) × 3 seeds × 4 ratios × 5 models = 120 runs\n80 actual epochs; Validation-best; exploratory Test\nOracle physical forcing used in training constraint only')]
for i,(title,txt) in enumerate(texts):
 y=.95-i*.245;ax.text(.02,y,title,weight='bold',va='top',transform=ax.transAxes);ax.text(.02,y-.055,txt,va='top',transform=ax.transAxes,fontsize=8.5)
f.tight_layout();fig(f,'figure1_experimental_design',setup,'Figure 1. Study grid and matched Oracle120 design. The dashed box is the new exploratory SW-Jeju region, not a prior dynamical definition. The 100% subset contains 3,995 retained windows and is smaller than the original full training set. All five models use oracle forcing only where specified in their training losses; Test is exploratory.')
# Figure2: five models, seed SD.
f,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,lead in zip(axs,[7,14]):
 for j,m in enumerate(MODELS):
  g=summary[(summary.lead==lead)&(summary.model==m)].set_index('ratio').loc[RAT];ax.errorbar(np.arange(4)+(j-2)*.035,g.rmse_mean,yerr=g.rmse_std,color=COL[m],marker=MARK[m],lw=1,capsize=2,label=LABEL[m])
 ax.set(xticks=range(4),xticklabels=RAT,xlabel='Training subset (%)',title=f'Lead {lead} days');ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Test SST RMSE (K)');axs[1].legend(frameon=False);f.tight_layout();fig(f,'figure2_sst_performance',summary,'Figure 2. Test SST RMSE across training subsets and forecast leads. Points show means of three matched seeds; bars show sample SD (not confidence intervals). The 100% subset comprises 3,995 windows. All values are from fixed Validation-best checkpoints on the previously inspected Test period.')
# Figure3 distinguish time CI and seed scatter.
npair=pdseeds[(pdseeds.reference=='NoAdv')&(pdseeds.candidate=='Adv')];npmean=ps[(ps.reference=='NoAdv')&(ps.candidate=='Adv')]
f,axs=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
for ax,lead in zip(axs,[7,14]):
 for i,r in enumerate(RAT):
  g=npair[(npair.lead==lead)&(npair.ratio==r)];c=npmean[(npmean.lead==lead)&(npmean.ratio==r)].iloc[0];ax.vlines(i,c.ci95_low,c.ci95_high,color=COL['NoAdv'],lw=3);ax.scatter(i,c.delta_rmse_mean,c='k',marker='D',s=32,zorder=4)
  for j,s in enumerate(SEEDS):ax.scatter(i+(j-1)*.12,g.set_index('seed').loc[s,'delta_rmse'],facecolors='none',edgecolors=COL['Adv'],marker=['o','s','^'][j],s=35,label=f'Seed {s}' if i==0 else None)
 ax.axhline(0,color='.4',lw=.8,ls='--');ax.set(xticks=range(4),xticklabels=RAT,xlabel='Training subset (%)',title=f'Lead {lead} days');ax.text(.02,.98,'Positive: NoAdv better',va='top',transform=ax.transAxes,fontsize=8)
axs[0].set_ylabel('ΔRMSE = Adv − NoAdv (K)');axs[1].legend(frameon=False);f.tight_layout();fig(f,'figure3_advection_effect',npair.merge(npmean[['ratio','lead','ci95_low','ci95_high']],on=['ratio','lead']),'Figure 3. Paired horizontal-advection constraint comparison. Open markers identify seeds; black diamonds are means. Blue intervals reuse the existing 2,000-replicate circular 90-day temporal block bootstrap, conditional on the same three seeds. Positive Adv−NoAdv RMSE favors NoAdv. Temporal intervals and between-seed dispersion quantify different uncertainties; no multiplicity adjustment or population-of-seeds inference is implied.')
# Figure4 MHW three complementary metrics.
f,axs=plt.subplots(2,3,figsize=(12,6),sharex=True)
for i,lead in enumerate([7,14]):
 for j,metric in enumerate(['mhw_grid_csi','mhw_grid_pod','mhw_grid_far']):
  ax=axs[i,j]
  for k,m in enumerate(MODELS):
   g=summary[(summary.lead==lead)&(summary.model==m)].set_index('ratio').loc[RAT];ax.errorbar(np.arange(4)+(k-2)*.035,g[metric+'_mean'],yerr=g[metric+'_std'],color=COL[m],marker=MARK[m],capsize=2,lw=1,label=LABEL[m])
  ax.set(xticks=range(4),xticklabels=RAT,title=f'Lead {lead}: '+{'mhw_grid_csi':'CSI ↑','mhw_grid_pod':'POD ↑','mhw_grid_far':'FAR ↓'}[metric],ylabel='Score');ax.grid(axis='y',alpha=.2)
  if i==1:ax.set_xlabel('Training subset (%)')
axs[0,2].legend(frameon=False,fontsize=7);f.tight_layout();fig(f,'figure4_mhw_detection',summary,'Figure 4. Pooled grid-day MHW CSI, POD, and FAR across subsets and leads, with three-seed sample SD. FAR is the false-alarm ratio FP/(TP+FP), not the false-positive rate. A single fixed threshold and common observed labels are used across all ratios and models. Five-day events join predictions issued at different times at a fixed lead, rather than a single-issuance five-day alarm.')
# Figure5 maps all ratios + seasonal panels.
f,axs=plt.subplots(3,4,figsize=(12,9));gridmean=sd.groupby(['ratio','lead','latitude','longitude']).delta_rmse_adv_minus_noadv.mean().reset_index();lim=max(.05,float(np.quantile(abs(gridmean.delta_rmse_adv_minus_noadv),.98)))
for i,lead in enumerate([7,14]):
 for j,r in enumerate(RAT):
  ax=axs[i,j];g=gridmean[(gridmean.ratio==r)&(gridmean.lead==lead)];z=g.pivot(index='latitude',columns='longitude',values='delta_rmse_adv_minus_noadv').reindex(index=lat,columns=lon);im=ax.pcolormesh(lon,lat,z,cmap='RdBu',vmin=-lim,vmax=lim,shading='nearest');ax.contour(lon,lat,sea0,[.5],colors='.4',linewidths=.4);ax.set(title=f'{r}%, lead {lead}',xlabel='Longitude (°E)',ylabel='Latitude (°N)')
for j,r in enumerate(RAT):
 ax=axs[2,j]
 for lead,col in [(7,'#0072B2'),(14,'#E69F00')]:
  g=rd[(rd.ratio==r)&(rd.lead==lead)&(rd.region=='All')&rd.season.isin(['DJF','MAM','JJA','SON'])].groupby('season').delta_rmse_adv_minus_noadv.agg(['mean','std']).loc[['DJF','MAM','JJA','SON']];ax.errorbar(range(4),g['mean'],yerr=g['std'],marker='o',color=col,capsize=2,label=f'Lead {lead}')
 ax.axhline(0,color='.5',lw=.7);ax.set(xticks=range(4),xticklabels=['DJF','MAM','JJA','SON'],title=f'{r}%: seasonal',ylabel='ΔRMSE (K)')
axs[2,0].legend(frameon=False,fontsize=7);f.subplots_adjust(right=.90,hspace=.5,wspace=.5);cb=f.colorbar(im,ax=axs[:2,:],fraction=.025,pad=.02);cb.set_label('Adv − NoAdv RMSE (K); positive favors NoAdv')
fig(f,'figure5_spatial_seasonal',gridmean,'Figure 5. Spatial and seasonal SST differences at all ratios. Maps average seedwise pointwise Adv−NoAdv RMSE (K); the shared color scale is clipped at the 98th percentile of absolute mapped values to reveal structure. Seasonal points average seedwise regional RMSE differences, with sample SD. Positive values favor NoAdv. Maps are descriptive and contain spatially dependent cells. Seasonal plot values are in statistics/region_season_paired.csv (region=All) and regional_seasonal_seed_metrics.csv.')
# S1 fixed diagnostic medians and batch-aggregated epoch losses.
diag=diag.drop_duplicates(['name','epoch'],keep='last');diag['weighted_gradient_percent']=100*diag.weighted_gradient_ratio
agg=diag.groupby('epoch')[['rres_rms_kday','auxiliary_loss','weighted_gradient_percent']].agg(['median','min','max','count']);agg.columns=['_'.join(c) for c in agg.columns];agg=agg.reset_index();csv(agg,'statistics/auxiliary_diagnostic_by_epoch.csv');csv(agg[agg.epoch.isin([1,20,80])],'tables/auxiliary_epoch1_20_80.csv')
f,axs=plt.subplots(1,3,figsize=(11,3.5))
for ax,k,ylabel in zip(axs,['rres_rms_kday','auxiliary_loss','weighted_gradient_percent'],['Rres output RMS (K day⁻¹)','Unweighted auxiliary loss','Weighted auxiliary / data gradient (%)']):
 gg=diag[diag.epoch.isin([1,5,10,20,80])].groupby('epoch')[k];med=gg.median();q1=gg.quantile(.25);q3=gg.quantile(.75);ax.plot(med.index,med,color=COL['Auxiliary']);ax.fill_between(med.index,q1,q3,color=COL['Auxiliary'],alpha=.2);ax.set(xlabel='Actual epoch (diagnostic batch)',ylabel=ylabel,yscale='log')
f.tight_layout();fig(f,'S1_auxiliary_decay',diag,'Figure S1. Saved auxiliary diagnostic batches: median and interquartile range at common scheduled epochs 1, 5, 10, 20 and 80, each with all 24 runs. Irregular best-candidate diagnostics are retained in CSV but excluded from the plotted aggregate trend; epoch-specific counts are provided. Gradient ratio is 100×0.0001||∇shared U||/||∇shared D||. Epoch 1 is after one training epoch. Declining magnitudes do not establish a causal mechanism or exact zero in the Rres head.',supp=True)
# S2 all six prescribed comparisons seedwise percentages.
f,axs=plt.subplots(2,3,figsize=(12,6),sharex=True)
for ax,(ref,cand) in zip(axs.ravel(),pairs):
 g=pdseeds[(pdseeds.reference==ref)&(pdseeds.candidate==cand)]
 for lead,c in [(7,'#0072B2'),(14,'#E69F00')]:
  for j,r in enumerate(RAT):
   v=g[(g.lead==lead)&(g.ratio==r)].improvement_pct;ax.scatter(np.full(3,j)+(np.arange(3)-1)*.07+(-.16 if lead==7 else .16),v,c=c,s=16,label=f'Lead {lead}' if j==0 else None)
 ax.axhline(0,color='.5',lw=.8);ax.set(title=f'{LABEL[ref]} → {LABEL[cand]}',xticks=range(4),xticklabels=RAT,ylabel='Seedwise RMSE improvement (%)',xlabel='Training subset (%)')
axs[0,0].legend(frameon=False);f.tight_layout();fig(f,'S2_seed_improvement',pdseeds,'Figure S2. Seedwise RMSE improvement percentages for six prespecified paired comparisons: 100(reference−candidate)/reference. Each dot represents an identical ratio/lead/seed pair; positive favors the candidate. Percentage averages in tables are averages of these seedwise values, not percentages computed from group means.',supp=True)
# S3 observed advection distribution, seasonal strength, spatial descriptive association.
f,axs=plt.subplots(1,3,figsize=(12,3.7))
for regname,c in [('All','#0072B2'),('SW_Jeju_exploratory','#CC79A7')]:
 g=ah[(ah.lead==7)&(ah.region==regname)];axs[0].plot((g.bin_left+g.bin_right)/2,g['count']/g.n_total/.04,color=c,label=regname.replace('_',' '))
 gg=ph[(ph.lead==7)&(ph.region==regname)&ph.season.isin(['DJF','MAM','JJA','SON'])].set_index('season').loc[['DJF','MAM','JJA','SON']];axs[1].plot(range(4),gg.rms_kday,'o-',c=c,label=regname.replace('_',' '));axs[1].plot(range(4),gg.absmean_kday,'s--',c=c,alpha=.6)
axs[0].set(xlabel='Observed advection (K day⁻¹)',ylabel='Density within displayed range',title='(a) Lead-7 Test target dates');axs[0].legend(frameon=False,fontsize=7)
axs[1].set(xticks=range(4),xticklabels=['DJF','MAM','JJA','SON'],ylabel='K day⁻¹',title='(b) Solid: RMS; dashed: mean |term|')
g=assoc[(assoc.ratio==100)&(assoc.lead==14)];axs[2].scatter(g.advection_rms_kday,g.delta_rmse_adv_minus_noadv,s=5,alpha=.4,c=COL['NoAdv']);axs[2].axhline(0,color='.5',lw=.7);axs[2].set(xlabel='Observed advection RMS (K day⁻¹)',ylabel='Adv − NoAdv RMSE (K)',title='(c) 100%, lead 14; descriptive cells')
f.tight_layout();fig(f,'S3_physical_diagnostics',ph,'Figure S3. Physical diagnostic from observed Test-day SST and saved mixed-layer velocities using the documented ocean-neighbor stencil, signed metre spacing and −86400(uTx+vTy). These observed-field tendencies are not the model-interpolated training-path advection terms. Panel (a) displays −2 to 2 K/day without renormalizing the omitted tails; tail counts are saved separately. Panel (b) compares seasonal RMS and mean absolute magnitude. Panel (c) is a descriptive spatial association, not independent samples or a causal test. Companion CSVs: advection_histogram.csv, observed_advection_rms_map.csv, advection_spatial_association.csv.',supp=True)
# S4 region-season all-ratio contrasts.
f,axs=plt.subplots(2,2,figsize=(11,7))
regionnames=['All','SW_Jeju_exploratory','Coast','Offshore'];colors=['#555555','#CC79A7','#0072B2','#E69F00']
for i,lead in enumerate([7,14]):
 for j,(metric,ylabel) in enumerate([('delta_rmse_adv_minus_noadv','Adv − NoAdv RMSE (K)'),('delta_csi_noadv_minus_adv','NoAdv − Adv CSI')]):
  ax=axs[i,j]
  for k,(rn,c) in enumerate(zip(regionnames,colors)):
   g=rd[(rd.lead==lead)&(rd.region==rn)&(rd.season=='ALL')].groupby('ratio')[metric].agg(['mean','std']).loc[RAT];ax.errorbar(np.arange(4)+(k-1.5)*.04,g['mean'],yerr=g['std'],color=c,marker='o',capsize=2,label=rn.replace('_',' '))
  ax.axhline(0,color='.5',lw=.7);ax.set(xticks=range(4),xticklabels=RAT,xlabel='Training subset (%)',ylabel=ylabel,title=f'Lead {lead}; positive favors NoAdv')
axs[0,0].legend(frameon=False,fontsize=7);f.tight_layout();fig(f,'S4_regional_differences',rd,'Figure S4. Paired regional SST and MHW differences at all ratios and leads, with three-seed sample SD. Positive signs favor NoAdv for both displayed definitions. The SW-Jeju box is newly specified and exploratory (see spatial_case_protocol.md); coast/offshore follow the saved scoring definition. Regional and seasonal TP/FP/FN values are provided in regional_seasonal_seed_metrics.csv and region_season_paired.csv.',supp=True)
# S5 neutral fixed-calendar case, can include non-event cells/day.
f,axs=plt.subplots(2,4,figsize=(12,6));caserows=[]
for i,lead in enumerate([7,14]):
 ex=example[(lead,'Baseline')];date=ex['date'];obs=ex['obs'];obsmin=float(obs[sea0].min());obsmax=float(obs[sea0].max());im=axs[i,0].pcolormesh(lon,lat,np.where(sea0,obs,np.nan),cmap='viridis',vmin=obsmin,vmax=obsmax);axs[i,0].set_title(f'Observed SST, {date}');f.colorbar(im,ax=axs[i,0],label='SST (°C)',fraction=.05)
 for j,m in enumerate(['Baseline','NoAdv','Adv'],1):
  e=example[(lead,m)];code=np.full(sea0.shape,np.nan);code[sea0]=0;code[sea0&e['pm']&e['om']]=1;code[sea0&e['pm']&~e['om']]=2;code[sea0&~e['pm']&e['om']]=3
  from matplotlib.colors import ListedColormap,BoundaryNorm
  cmap=ListedColormap(['#eeeeee','#009E73','#E69F00','#CC79A7']);im=axs[i,j].pcolormesh(lon,lat,code,cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),shading='nearest');axs[i,j].set_title(f'{m}, lead {lead}')
  for iy,ix in zip(*np.where(sea0)):caserows.append(dict(lead=lead,date=date,ratio=100,seed=42,model=m,latitude=lat[iy],longitude=lon[ix],observed_sst=obs[iy,ix],predicted_sst=e['pred'][iy,ix],observed_mhw=int(e['om'][iy,ix]),predicted_mhw=int(e['pm'][iy,ix]),class_code=int(code[iy,ix])))
 for j,ax in enumerate(axs[i]):ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)' if j==0 else '')
f.subplots_adjust(wspace=.5,hspace=.45,bottom=.16);cax=f.add_axes([.3,.04,.5,.025]);cb=f.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,1,2,3]);cb.ax.set_xticklabels(['TN','TP','FP','FN']);fig(f,'S5_calendar_case',pd.DataFrame(caserows),'Figure S5. Reproducible calendar-selected case: earliest June–August day of the final complete Test year, fixed 100% subset and seed 42. Selection does not depend on skill or event intensity, so this is an illustrative day rather than a claim of a representative extreme event. MHW labels are taken from the already evaluated complete fixed-lead series. All five-model predictions for this date are retained in the companion case_all_models.csv.',supp=True)
allcases=[]
for (lead,m),e in example.items():
 for iy,ix in zip(*np.where(sea0)):allcases.append(dict(lead=lead,model=m,date=e['date'],latitude=lat[iy],longitude=lon[ix],obs_sst=e['obs'][iy,ix],pred_sst=e['pred'][iy,ix],obs_mhw=int(e['om'][iy,ix]),pred_mhw=int(e['pm'][iy,ix])))
csv(pd.DataFrame(allcases),'plot_data/case_all_models.csv')
# Supplementary physical residual correlation, full existing seed data.
f,axs=plt.subplots(1,2,figsize=(10,3.5))
for ax,k in zip(axs,['A','B']):
 for m in MODELS:
  g=d[d.model==m];ax.scatter(g.rmse,g[k+'_residual_rms_kday'],color=COL[m],marker=MARK[m],s=25,label=LABEL[m])
 ax.set(xlabel='Test SST RMSE (K)',ylabel='Posthoc residual RMS (K day⁻¹)',title=f'Common {k} budget diagnostic')
axs[1].legend(frameon=False,fontsize=7);f.tight_layout();fig(f,'S6_residual_error',resid,'Figure S6. Existing common-equation posthoc residual RMS and SST RMSE for all 120 frozen predictions. A excludes horizontal advection and B includes it. Daily within-run residual/error correlations are also provided. Descriptive associations do not show independent physical closure or causal improvement.',supp=True)
# Reports answer the seven research questions using actual computed values.
phys=seeds[seeds.model.isin(['P0','NoAdv','Adv'])];aux=seeds[seeds.model=='Auxiliary'];nwin=int((npair.delta_rmse>0).sum());cwin=int((npair.noadv_minus_adv_csi>0).sum());means=npmean.sort_values(['lead','ratio'])
limitations='''# Limitations\n\n- Exploratory Test: the fixed Test period has been inspected previously; these are not independent confirmatory results. No new training, tuning, inference, checkpoint selection or full verification audit was run.\n- Three fixed seeds and a single fixed period do not support seed-population or geographic generalization. The reused 90-day bootstrap intervals are conditional on these seeds; temporal autocorrelation is handled by paired circular blocks, with all grid cells kept together. No multiplicity adjustment was applied. Grid cells are spatially dependent, not independent sample replicates.\n- 100% means 3,995 windows inside retained 90-day blocks, not all original training windows. Actual epochs imply different optimizer-update counts by ratio; the last epoch has zero learning rate.\n- Oracle future physical forcing is used by specified training constraints, not the predictor inputs at Test; this is not an operational proof for forecast forcing.\n- One common fixed MHW reference is used across ratios/models, so changing thresholds do not confound ratio comparisons here. Skill still depends on the fixed period, subset statistics, and model calibration. MHW events use raw day-of-year, five-day runs, no smoothing/gap merging, zero scores at zero denominators, and edge-censored series. Fixed-lead series join different issuance dates.\n- SST metrics use equal ocean-grid weights, not area weighting. The historical SW-Korea box is not SW Jeju. The new SW-Jeju box is exploratory and geographically motivated; no optimized region selection was performed.\n- Observed-field advection diagnostics use target-day SST and velocity to describe physical magnitude; they are distinct from model-interpolated training-path terms. Spatial associations and physical residual correlations reuse related variables and cannot establish an independent oceanographic cause.\n- A large advective term does not guarantee a beneficial fixed discrete constraint. NoAdv outperforming Adv does not imply ocean advection is unimportant. P0 and Adv differ in flux, entrainment and velocity definitions; only NoAdv–Adv isolates this constraint toggle.\n- Auxiliary diagnostic batches are fixed training subsets and sampled at irregular epochs. Declining output/gradient is observed; complete zeroing or a unique causal explanation is not established.\n- Existing WSDM results remain separate and were neither revalidated nor mixed with Oracle120. Prior-report audit caveats are attributed rather than converted into new PASS claims.\n'''
write('reports/limitations.md',limitations)
claims=[]
def claim(kind,text,evidence,limit):claims.append(dict(category=kind,claim=text,evidence=evidence,limitation=limit))
claim('Directly supported',f'NoAdv has lower Test RMSE than Adv in {nwin}/24 paired seed conditions and higher CSI in {cwin}/24.','statistics/paired_seed_results.csv; statistics/paired_summary.csv','Exploratory Test, fixed period and three seeds.')
for m in ['P0','NoAdv','Adv']:
 g=seeds[seeds.model==m];claim('Directly supported',f'{m} improves RMSE over Baseline in {int((g.paired_rmse_improvement_pct>0).sum())}/24 seeds; mean seed improvement {g.paired_rmse_improvement_pct.mean():.3f}%.','tables/test_seed_metrics.csv','Cross-condition average is descriptive, not an equally weighted new scientific endpoint.')
claim('Directly supported',f'Auxiliary-only improves Test RMSE in {int((aux.paired_rmse_improvement_pct>0).sum())}/24 seed conditions.','tables/test_seed_metrics.csv; tables/validation_seed_metrics.csv','Small and heterogeneous changes; validation and Test separated.')
claim('Limited interpretation','The utility of the fixed constraints depends on term selection, training ratio and lead.','tables/table5_paired.csv; figures/main/figure2_sst_performance.png','No universal monotonic scarcity benefit or operational generalization.')
claim('Not supported','Advection is physically unimportant, or a physical causal mechanism has been identified.','statistics/observed_advection_season_region.csv; statistics/advection_spatial_correlations.csv','Predictive constraint utility does not establish physical-process importance; descriptive dependence only.')
claim('Not supported','Physics constraints universally improve forecasts or Test is independent confirmation.','tables/table3_sst.csv; reports/limitations.md','Failures and small effects must be retained.')
csv(pd.DataFrame(claims),'reports/claim_evidence_matrix.csv');write('reports/claim_evidence_matrix.md','# Claim–evidence matrix\n\n'+pd.DataFrame(claims).to_markdown(index=False)+'\n')
answers=['# 최종 분석 요약\n',f'Oracle120 기존 결과만 사용했다. 120개 조건의 입력 파일 존재 및 날짜·관측 SST·Mask·MHW 라벨 대응을 확인했다. 모델 학습/추론/감사 전체 재실행 없이, 기존 지표와 90일 bootstrap을 재사용하고 저장 예측으로 공간·계절 요약을 생성했다.\n', '## 1. 고정 물리 제약이 Baseline보다 일관되게 우수한가?\n']
for m in ['P0','NoAdv','Adv']:
 g=seeds[seeds.model==m];gg=ps[(ps.reference=='Baseline')&(ps.candidate==m)];answers.append(f'- {m}: seed별 RMSE 개선 {int((g.paired_rmse_improvement_pct>0).sum())}/24, 조건별 평균 RMSE 개선 {int((gg.delta_rmse_mean<0).sum())}/8. seed 개선율 범위 {g.paired_rmse_improvement_pct.min():.3f}~{g.paired_rmse_improvement_pct.max():.3f}%.')
answers+=['\n보편적 우위로 기술하지 않는다. 모델별·조건별 효과와 실패를 함께 제시한다.\n','## 2. Auxiliary-only의 독립 효과\n',f'Test RMSE 개선 {int((aux.paired_rmse_improvement_pct>0).sum())}/24. Validation 개선 {int((vseeds[vseeds.model=="Auxiliary"].improvement_pct>0).sum())}/24.\n']
answers.append(ps[(ps.reference=='Baseline')&(ps.candidate=='Auxiliary')][['ratio','lead','delta_rmse_mean','paired_improvement_pct_mean','candidate_rmse_wins','ci95_low','ci95_high']].to_markdown(index=False))
answers+=['\nEpoch 1/20/80의 원본 진단 batch 중앙값은 tables/auxiliary_epoch1_20_80.csv에 수록했다. 감소 관찰과 원인 해석을 구분하며, Rres가 완전히 0이라고 하지 않는다.\n','## 3. NoAdv와 Adv 차이가 반복되는 조건\n',f'NoAdv 우위: RMSE {nwin}/24, CSI {cwin}/24 seed 조건. 아래 ΔRMSE는 Adv−NoAdv로 양수가 NoAdv 우위다.\n']
answers.append(means[['ratio','lead','delta_rmse_mean','delta_rmse_sd','candidate_rmse_wins','ci95_low','ci95_high']].rename(columns={'candidate_rmse_wins':'Adv_RMSE_wins_out_of_3'}).to_markdown(index=False))
answers+=['\n## 4. 자료 감소가 제약의 유용성을 증가시키는가?\n','비율별 대응 개선율은 아래와 같다. 값은 동일 조건 내 seed 개선율의 평균이며 자료 부족에 따른 단조 개선을 가정하지 않는다. 100%는 3,995개 window다.\n',ps[ps.reference=='Baseline'].pivot_table(index=['candidate','lead'],columns='ratio',values='paired_improvement_pct_mean').to_markdown(),'\n## 5. SST 차이와 MHW 차이의 연결\n']
quad=pdseeds.copy();quad['sst_candidate_better']=quad.delta_rmse<0;quad['csi_candidate_better']=quad.delta_csi>0
# pandas-version-neutral explicit computation.
q=pd.DataFrame([dict(reference=k[0],candidate=k[1],n=len(g),sst_and_csi_better=int(((g.delta_rmse<0)&(g.delta_csi>0)).sum()),sst_better_csi_not_better=int(((g.delta_rmse<0)&(g.delta_csi<=0)).sum()),descriptive_delta_correlation=g.delta_rmse.corr(g.delta_csi)) for k,g in quad.groupby(['reference','candidate'])]);csv(q,'statistics/sst_mhw_concordance.csv');answers.append(q.to_markdown(index=False));answers.append('\nRMSE 개선이 CSI 개선을 보장하지 않는다. POD/FAR와 TP/FP/FN을 함께 판단한다.\n')
answers+=['## 6. 공간·계절 차이\n',region_protocol+'\n',rd[(rd.region=='All')&(rd.season!='ALL')].groupby(['lead','season']).delta_rmse_adv_minus_noadv.agg(['mean','min','max']).to_markdown(),'\n위 mean/min/max는 ratio×seed 조건별 대응차의 기술적 요약이다. Figure5와 지역·계절 원본 CSV에서 조건별 차이를 확인한다. 관측장 기반 이류항 크기와 오차차의 공간 상관은 인과 근거가 아니다.\n','## 7. 논문에서 허용되는 결론\n','고정 열수지 제약의 예측 효용은 항 구성, 자료량 및 lead에 따라 달라진다. 현재 실험에서 수평 이류를 제외한 제약의 우위가 반복되지만, 이는 이산화된 학습 제약의 효용에 관한 결과이며 실제 해양 이류의 중요성을 부정하지 않는다. Auxiliary-only 효과는 조건부이며 큰 보편적 개선으로 서술하지 않는다. 단일 Test 기간과 3개 seed에 조건부인 탐색 결과라는 한계를 명시한다.\n','## 산출물\n','Main Figure 1–5, Supplementary S1–S6; Table1–6 및 별도 Validation/seed 표; 모두 영어 Figure·캡션, 350dpi PNG·PDF·SVG 및 plot CSV. 수치 검증 보고서는 기존 증거를 인용하며 새 PASS로 포장하지 않는다. WSDM 확정 기존 표는 [기존 별도 표 디렉터리]('+str(R/'paper_analysis_20260929/results/paper_tables')+')에 그대로 두었다.\n']
write('reports/analysis_summary_ko.md','\n'.join(answers))
missing='''# Missing assets and scope decisions\n\n- No requested main Figure was omitted; S1–S5 are present, plus S6 residual/error diagnostics.\n- Validation MHW metrics are not present in the consolidated validation inputs. Validation RMSE and MAE come from the saved selected epoch in each run history; no Validation MHW values were reconstructed or inferred.\n- No prior verified SW-Jeju mask was found in the final evidence. Historical SW-Korea is kept separate. A new exploratory geographical box is explicitly documented before skill summaries.\n- Independent oceanographic cause, operational forecast-forcing utility, and seed-population uncertainty cannot be inferred from these assets. No new experiments were added.\n- Existing WSDM quantitative tables are linked as a separate historical artifact; not copied into or revalidated as part of Oracle120.\n- Observed target-day advection diagnostic is available; a full per-grid archive of each interpolated training-path advection term is not supplied here. Do not label Figure S3 as that training-path distribution. Existing posthoc residuals are reused for model-dependent diagnostics.\n- Calendar-selected Figure S5 is an illustrative fixed-date case, not a severity-selected representative extreme.\n''';write('reports/missing_assets.md',missing)
repro=f'''# Reproducibility summary\n\n- Analysis: `{O/'code/analyze.py'}`; CPU only, CUDA hidden; no model/checkpoint loading.\n- Run: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {Path(sys.executable)} -B {O/'code/analyze.py'}`\n- All new outputs and runtime caches are under `{O}`. Existing inputs are opened read-only.\n- Tables are recalculated from test_results120.csv and fixed validation metrics, with sample SD (ddof=1). Seedwise percentages precede averaging.\n- All six pairs join identical ratio, lead, seed. NoAdv ΔRMSE=Adv−NoAdv and ΔCSI=NoAdv−Adv.\n- Existing 90-day bootstrap: 2,000 paired circular temporal draws, common times across models and fixed three seeds; all grid cells together. No bootstrap rerun was needed.\n- Input presence/alignment for all120: reports/input_completeness.csv. New checks do not rerun the full prior audit.\n- Input SHA-256 for small analysis files and documented original field hashes: reports/source_inventory.json and input_completeness.csv. Current large inputs use stat metadata during this analysis; prior hashes are attributed, not claimed freshly recalculated.\n- Figures:350dpi PNG,vector PDF/SVG; numeric plot data and English captions. Companion plot-data files contain maps, regional results, cases and histograms.\n- Final epoch gradients and RMS must be distinguished between epoch training aggregates and the saved fixed diagnostic batch.\n- Supplemental observed-advection equations duplicate the documented stencil on the actual regular grid; not a re-audit of the physics implementation.\n''';write('reports/reproducibility_summary.md',repro)
# Each artifact source is supplemented by a precise run index and source hashes.
for x in manifest:
 x['analysis_conditions']='Oracle120; ratios10/25/50/100; leads7/14; seeds42/7/123; Validation-best; exploratory Test'
 x['input_run_index']='reports/input_completeness.csv'
 x['companion_data']='plot_data/; statistics/; reports/source_inventory.json'
csv(pd.DataFrame(manifest),'figure_table_manifest.csv')
source(O/'code/analyze.py');savejson('reports/source_inventory.json',sources)
# Verify generated results and formats with meaningful numerical checks.
assert len(pdseeds)==144 and len(ps)==48 and len(summary)==40 and len(checks)==120
assert (pd.read_csv(O/'tables/region_definitions.csv').n_cells>0).all()
assert np.allclose(summary.mhw_grid_precision_mean+summary.mhw_grid_far_mean,1)
assert len(agg[agg.epoch.isin([1,20,80])])==3
for p,info in sources.items():assert Path(p).stat().st_size==info['size'] and Path(p).stat().st_mtime_ns==info['mtime_ns'],p
pngs=list((O/'figures').rglob('*.png'));assert len(pngs)==11
from PIL import Image
for p in pngs:
 with Image.open(p) as im:assert min(im.info['dpi'])>=300 and min(im.size)>600
for x in manifest:
 for p in x['artifact'].split(';'):assert (O/p).is_file() and (O/p).stat().st_size>100
validation={'input_runs':120,'test_groups':40,'paired_seeds':144,'paired_groups':48,'figures':len(pngs),'tables':sum(x['kind']=='table' for x in manifest),'raw_dates_targets_masks_labels':'120 aligned; prior full metrics audit reused','source_stats_unchanged':True,'png_dpi_min':350,'bootstrap_recomputed':False,'training_or_inference_run':False,'NoAdv_rmse_wins':nwin,'NoAdv_csi_wins':cwin,'roi_counts':region_counts[['region','n_cells']].to_dict('records')}
savejson('reports/output_validation.json',validation)
now=datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
with (R/'docs/research_log.md').open('a') as f:f.write(f'\n\n### {now} — Oracle120 논문 최종 분석·Figure/Table 생성\n- 대상: {O}\n- 변경: 기존120개 예측·평가/검증자료를 이용한 Table1–6, 별도Validation/seed표, Main1–5 및 S1–S6 (PNG350dpi/PDF/SVG), plot CSV, 대응통계, 주장근거/한계/재현성 보고서 생성.\n- 이유: 사용자 첨부 최종분석 프롬프트 이행. 기존 데이터/결과/체크포인트 수정·삭제·이동 없음. 모델 학습·튜닝·추론·WSDM/Oracle120 감사 전체 재실행 없음.\n- 검증: 120개 파일 완전성 및 dates/targets/mask/MHWlabel 정렬, 144개 seed대응/48개 조건대응, 기존90일 CI재사용 확인. NoAdv RMSE {nwin}/24, CSI {cwin}/24. 새 SW-Jeju 좌표상자와 observed-field 이류진단 범위 명시. output_validation.json 참조.\n')
log('COMPLETE '+json.dumps(validation,ensure_ascii=False))
