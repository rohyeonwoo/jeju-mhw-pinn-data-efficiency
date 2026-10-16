"""CPU-only independent checks on frozen prediction files and paired time blocks."""
import setproctitle
setproctitle.setproctitle("yeonwoo")
import os,sys,json,time,hashlib,importlib.util,math
from pathlib import Path
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';T=R/'outputs/v8_actual_epoch_oracle96_20261011';A=R/'outputs/v8_actual_epoch_auxiliary24_20261015'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'scripts/oracle96_test'));sys.path.insert(0,str(R/'scripts/oracle96_auxiliary24'))
import aux_worker as W
import metrics as M
import mixed_budget_physics as mb
import numpy as np,pandas as pd,torch,xarray as xr
from mhw.datasets import compute_grid_spacing_m
from mhw.mhw_skill import compute_climatology_threshold,day_of_year,mhw_binary_field,enforce_min_duration
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 p=O/'runtime'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
torch.set_num_threads(1);J=lambda p:json.loads(Path(p).read_text());save=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,default=str));sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
DS=xr.open_dataset(W.BUDGET);sst=DS.sst_mean.values.astype('float32');la,lo=np.meshgrid(DS.latitude.values,DS.longitude.values,indexing='ij');dx,dy=compute_grid_spacing_m(la,lo);ar=mb.arrays()
with np.load(T/'evaluation_reference.npz') as z:clim=z['clim'];th=z['threshold'];refidx=z['reference_indices']
ref=np.isin(np.arange(len(sst)),refidx);cc,tt=compute_climatology_threshold(sst,day_of_year(DS.time.values),ref,5,.9)
assert np.array_equal(cc,clim,equal_nan=True) and np.array_equal(tt,th,equal_nan=True)
toy=np.array([0,1,1,1,1,0,1,1,1,1,1,0],bool)[:,None,None];assert np.array_equal(enforce_min_duration(toy,5).ravel(),np.array([0,0,0,0,0,0,1,1,1,1,1,0],bool))
save(O/'mhw_reference_check.json',dict(status='PASS',threshold_bitwise_equal=True,climatology_bitwise_equal=True,reference_first=str(DS.time.values[refidx[0]]),reference_last=str(DS.time.values[refidx[-1]]),n_reference=len(refidx),reference_sha256=sha(T/'evaluation_reference.npz'),leap_example={str(d):int(v) for d,v in zip(['2019-03-01','2020-02-29','2020-03-01'],day_of_year(np.array(['2019-03-01','2020-02-29','2020-03-01'],dtype='datetime64[D]')))},duration_test='PASS'))
# Same posthoc A/B equations for ALL 120; no learned heads needed.
def physics(f,lead):
 idx=f['target_index']-lead;pred=np.squeeze(f['pred_raw']);mask=np.squeeze(f['mask']);daily={k:[] for k in ['A_sse','B_sse','A_signed','B_signed','count','A_cum_sse','B_cum_sse']}
 for start in range(0,len(idx),32):
  ii=idx[start:start+32];days=ii[:,None]+np.arange(lead);bs={'mixed_'+k:torch.tensor(np.array(ar[k][days][:,:,None])) for k in ['h','qnet','ssr','u','v','tb','we','valid']};bs.update(dx=torch.tensor(dx)[None,None],dy=torch.tensor(dy)[None,None])
  with torch.no_grad():
   rr,rhs,s,ts=mb.path(torch.tensor(pred[start:start+32,None]),torch.tensor(sst[ii,None]),torch.tensor(mask[start:start+32,None]),bs,lead,False)
  for tag,r in [('A',rr),('B',rr-ts['advection']*s)]:
   r=torch.where(s>0,r,torch.zeros_like(r)).reshape(lead,len(ii),-1).double();sup=s.reshape(lead,len(ii),-1)
   daily[tag+'_sse'].extend(r.square().sum((0,2)).numpy());daily[tag+'_signed'].extend(r.sum((0,2)).numpy());daily[tag+'_cum_sse'].extend(r.sum(0).square().sum(1).numpy())
  daily['count'].extend(sup.sum((0,2)).numpy())
 daily={k:np.asarray(v) for k,v in daily.items()};out={}
 err=(pred.astype(float)-np.squeeze(f['target_raw']).astype(float));sea=np.squeeze(f['mask'][0])>0;er=np.sqrt(np.mean(err[:,sea]**2,axis=1))
 for tag in ['A','B']:
  dr=np.sqrt(daily[tag+'_sse']/daily['count']);out[tag+'_residual_rms_kday']=float(np.sqrt(daily[tag+'_sse'].sum()/daily['count'].sum()));out[tag+'_residual_mean_kday']=float(daily[tag+'_signed'].sum()/daily['count'].sum());out[tag+'_cumulative_rms_k']=float(np.sqrt(daily[tag+'_cum_sse'].sum()/(daily['count'].sum()/lead)));out[tag+'_cumulative_mean_k']=lead*out[tag+'_residual_mean_kday'];out[tag+'_daily_rmse_correlation']=float(np.corrcoef(er,dr)[0,1])
 return out,daily
jobs=[]
for j in W.protocol()['jobs']:jobs.append(dict(j,label=j['label'],folder=str(T/'test_evaluation_20261012/runs'/j['name'])))
for j in W.plan()['jobs']:jobs.append(dict(j,label='Auxiliary',folder=str(O/'auxiliary_test/runs'/j['name'])))
(O/'recomputed').mkdir(exist_ok=True);truth={};allrows=[]
for j in jobs:
 dest=O/'recomputed'/j['name'];dest.mkdir(exist_ok=True);folder=Path(j['folder'])
 while not (folder/'status.json').exists() or J(folder/'status.json')['status']!='COMPLETED':
  if (folder/'status.json').exists() and J(folder/'status.json')['status']=='FAILED':raise RuntimeError(str(folder))
  print('Waiting inference',j['name'],flush=True);time.sleep(15)
 if (dest/'metrics.json').exists():allrows.append(J(dest/'metrics.json'));continue
 with np.load(folder/'fields.npz') as z:f={k:z[k] for k in z.files}
 dates=f['target_dates'];idx=f['target_index'];assert np.array_equal(dates,DS.time.values[idx]);assert np.array_equal(np.squeeze(f['target_raw'])[:,np.isfinite(DS.ocean_mask_static.values)],sst[idx][:,np.isfinite(DS.ocean_mask_static.values)])
 measured,daily=M.score(f,dates,clim,th);old=J(folder/'metrics.json');keys=['rmse','mae','bias','mhw_grid_csi','mhw_grid_pod','mhw_grid_far','mhw_grid_f1','mhw_grid_precision','mhw_grid_hits','mhw_grid_misses','mhw_grid_false_alarms']
 assert all(abs(measured[k]-old[k])<=1e-12 for k in keys)
 if j['lead'] not in truth:truth[j['lead']]={k:daily[k] for k in ['target_index','target_dates','obs_mhw']}
 for k,v in truth[j['lead']].items():assert np.array_equal(daily[k],v)
 with np.load(folder/'daily_metrics.npz') as z:
  for k in ['sse','count','hits','misses','false_alarms','pred_mhw','obs_mhw']:assert np.array_equal(daily[k],z[k]),(j['name'],k)
 measured.update(name=j['name'],model=j['label'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],source_fields_sha256=sha(folder/'fields.npz'),metric_recompute='PASS',ground_truth_recheck='PASS')
 # Historical ROI, explicitly SW-Korea, NOT asserted geographically SW-Jeju.
 sea=np.squeeze(f['mask'][0])>0;roi=sea&(la>=33.5)&(la<=35)&(lo>=125.5)&(lo<=127.5);err=np.squeeze(f['pred_raw']).astype(float)-np.squeeze(f['target_raw']).astype(float);summer=np.isin(pd.DatetimeIndex(dates).month,[6,7,8]);measured['historical_SW_Korea_box_cells']=int(roi.sum());measured['historical_SW_Korea_box_rmse']=float(np.sqrt(np.mean(err[:,roi]**2)));measured['historical_SW_Korea_JJA_rmse']=float(np.sqrt(np.mean(err[summer][:,roi]**2)))
 phy,pdaily=physics(f,j['lead']);measured.update(phy);save(dest/'metrics.json',measured)
 np.savez_compressed(dest/'daily.npz',**{k:daily[k] for k in ['target_index','sse','count','hits','misses','false_alarms']},**{'physics_'+k:v for k,v in pdaily.items()});allrows.append(measured);print('Rechecked',j['name'],measured['rmse'],flush=True)
d=pd.DataFrame(allrows);d.to_csv(O/'test_results120.csv',index=False)
metrics=['rmse','mae','bias','mhw_grid_csi','mhw_grid_pod','mhw_grid_far','mhw_grid_precision','mhw_grid_f1','coast_rmse','offshore_rmse','DJF_rmse','MAM_rmse','JJA_rmse','SON_rmse','historical_SW_Korea_box_rmse','historical_SW_Korea_JJA_rmse','A_residual_rms_kday','B_residual_rms_kday','A_cumulative_mean_k','B_cumulative_mean_k','A_daily_rmse_correlation','B_daily_rmse_correlation']
summary=d.groupby(['ratio','lead','model'])[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(O/'comparison_summary.csv',index=False)
pairs=[('Baseline','Auxiliary'),('Baseline','P0'),('Baseline','NoAdv'),('Baseline','Adv'),('NoAdv','Adv'),('Auxiliary','NoAdv')];seedrows=[];cirows=[];pairsummary=[]
# Pre-generate shared circular time-block resamples for each lead and block size.
# Keep all cells together, identical dates across models AND fixed seeds; mean of seed metrics.
for lead in [7,14]:
 n=len(truth[lead]['target_index']);weights={}
 for block in [30,90,180]:
  rng=np.random.default_rng(20261015+lead+block);w=np.zeros((2000,n),dtype=np.float64)
  for rep in range(2000):
   starts=rng.integers(n,size=math.ceil(n/block));ix=((starts[:,None]+np.arange(block))%n).ravel()[:n];w[rep]=np.bincount(ix,minlength=n)
  weights[block]=w
 for ratio in [10,25,50,100]:
  subset=d[(d.lead==lead)&(d.ratio==ratio)];modeldata={}
  for label in ['Baseline','P0','NoAdv','Adv','Auxiliary']:
   modeldata[label]=[]
   for seed in [42,7,123]:
    row=subset[(subset.model==label)&(subset.seed==seed)].iloc[0];z=np.load(O/'recomputed'/row['name']/'daily.npz');modeldata[label].append({k:z[k] for k in ['sse','count','hits','misses','false_alarms']});z.close()
  for left,right in pairs:
   a=subset[subset.model==left].set_index('seed');b=subset[subset.model==right].set_index('seed');delta=b.rmse-a.rmse;improvement=100*(a.rmse-b.rmse)/a.rmse
   for seed in [42,7,123]:seedrows.append(dict(ratio=ratio,lead=lead,seed=seed,reference=left,candidate=right,reference_rmse=a.loc[seed,'rmse'],candidate_rmse=b.loc[seed,'rmse'],delta_rmse=delta[seed],improvement_pct=improvement[seed],**{'delta_'+k:b.loc[seed,k]-a.loc[seed,k] for k in ['mhw_grid_csi','mhw_grid_pod','mhw_grid_far','mhw_grid_hits','mhw_grid_misses','mhw_grid_false_alarms']}))
   pairsummary.append(dict(ratio=ratio,lead=lead,reference=left,candidate=right,delta_rmse_mean=delta.mean(),delta_rmse_sd=delta.std(ddof=1),mean_paired_improvement_pct=improvement.mean(),candidate_wins=int((delta<0).sum())))
   for block,w in weights.items():
    samples={}
    for label in [left,right]:
     vals=[]
     for v in modeldata[label]:
      arr=np.stack([v[k] for k in ['sse','count','hits','misses','false_alarms']],axis=1);res=w@arr;sse,cnt,h,m,f=res.T
      vals.append(np.stack([np.sqrt(sse/cnt),np.divide(h,h+m+f,out=np.zeros_like(h),where=h+m+f>0),np.divide(h,h+m,out=np.zeros_like(h),where=h+m>0),np.divide(f,h+f,out=np.zeros_like(f),where=h+f>0)],axis=1))
     samples[label]=np.mean(vals,axis=0)
    diff=samples[right]-samples[left]
    for col,name in enumerate(['rmse','mhw_grid_csi','mhw_grid_pod','mhw_grid_far']):
     low,high=np.quantile(diff[:,col],[.025,.975]);cirows.append(dict(ratio=ratio,lead=lead,reference=left,candidate=right,metric=name,block_days=block,replicates=2000,estimate=float(b[name].mean()-a[name].mean()),ci_low=float(low),ci_high=float(high),conditional_fixed_seeds=True))
   print('Bootstrap',ratio,lead,left,right,flush=True)
pd.DataFrame(seedrows).to_csv(O/'paired_seed_results.csv',index=False);pd.DataFrame(pairsummary).to_csv(O/'paired_summary.csv',index=False);pd.DataFrame(cirows).to_csv(O/'block_bootstrap.csv',index=False)
save(O/'analysis_status.json',dict(status='COMPLETE',models=120,recomputed_metrics_match=120,ground_truth_same=True,block_days=[30,90,180],replicates=2000,resampling='paired circular time blocks; all grid cells jointly; same times across fixed three seeds; mean per-seed metrics; MHW labels fixed before resampling',bootstrap_seed='20261015+lead+block',NoAdv_vs_Adv_seed_wins=int(sum(x['reference']=='NoAdv' and x['candidate']=='Adv' and x['delta_rmse']>0 for x in seedrows))))
