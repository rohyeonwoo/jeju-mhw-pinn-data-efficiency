"""Frozen scientific scoring definitions; full consecutive fixed-lead Test series."""
from eval_common import *
import math
import numpy as np
import pandas as pd
from mhw.mhw_skill import compute_mhw_skill_metrics,mhw_binary_field,day_of_year
from summarize_jeju_hilam_mhw_skill import anomaly_metrics

def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 return None if isinstance(x,float) and not math.isfinite(x) else x

def score(fields,dates,clim,threshold):
 idx=fields['target_index'];assert np.all(np.diff(idx)==1)
 assert np.all(np.diff(dates).astype('timedelta64[D]').astype(int)==1)
 pred=np.squeeze(fields['pred_raw']);obs=np.squeeze(fields['target_raw']);sea=np.squeeze(fields['mask'][0])>0
 assert pred.shape==obs.shape and pred.ndim==3 and len(pred)==len(dates)
 assert np.all(np.squeeze(fields['mask'])==sea)
 assert np.isfinite(pred[:,sea]).all() and np.isfinite(obs[:,sea]).all()
 doy=day_of_year(dates);err=pred.astype(np.float64)-obs.astype(np.float64)
 def region(e):return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(e.mean()))
 result=region(err[:,sea]);am=anomaly_metrics(pred,obs,clim[doy],sea);result.update(am);result['acc']=result.pop('test_acc')
 land=~sea;coast=np.zeros_like(sea);coast[1:]|=land[:-1];coast[:-1]|=land[1:];coast[:,1:]|=land[:,:-1];coast[:,:-1]|=land[:,1:];coast&=sea
 for name,mask in [('coast',coast),('offshore',sea&~coast)]:
  result.update({name+'_'+k:v for k,v in region(err[:,mask]).items()})
 months=pd.DatetimeIndex(dates).month.to_numpy()
 for name,mos in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:result.update({name+'_'+k:v for k,v in region(err[np.isin(months,mos)][:,sea]).items()})
 result.update(compute_mhw_skill_metrics(pred,obs,doy,threshold,sea,min_duration=5,area_frac=.15,onset_tolerance_days=3))
 pm=mhw_binary_field(pred,doy,threshold,5)&sea;om=mhw_binary_field(obs,doy,threshold,5)&sea
 hits=(pm&om).reshape(len(idx),-1).sum(1);miss=(~pm&om).reshape(len(idx),-1).sum(1);fa=(pm&~om).reshape(len(idx),-1).sum(1)
 h,m,f=map(int,[hits.sum(),miss.sum(),fa.sum()]);result['mhw_grid_precision']=h/(h+f) if h+f else 0.
 assert h==result['mhw_grid_hits'] and m==result['mhw_grid_misses'] and f==result['mhw_grid_false_alarms']
 assert abs(result['mhw_grid_csi']-(h/(h+m+f) if h+m+f else 0))<1e-12
 result.update(n_targets=len(idx),ocean_cells=int(sea.sum()),coast_cells=int(coast.sum()),test_first_date=str(dates[0])[:10],test_last_date=str(dates[-1])[:10],mhw_day_rmse=float(np.sqrt(np.mean(err[om]**2))) if om.any() else None,mhw_day_bias=float(err[om].mean()) if om.any() else None)
 daily=dict(target_index=idx,target_dates=dates,sse=np.sum(err[:,sea]**2,axis=1),count=np.full(len(idx),sea.sum()),signed=err[:,sea].sum(1),abs_error=np.abs(err[:,sea]).sum(1),hits=hits,misses=miss,false_alarms=fa,pred_mhw=pm,obs_mhw=om)
 return safe(result),daily
