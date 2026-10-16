from cpu_diagnostics_common import *
import torch,xarray as xr
from torch.utils.data import DataLoader
from mhw import subset_preprocessing as p
from mhw.ocean_stencil import ocean_neighbor_support,shifted_neighbors
import train_hybrid_jeju_forecast as h
from types import SimpleNamespace
torch.set_num_threads(2)

def summary(obs,phys,daily_res,lead):
 error=phys.astype(float)-obs;zero=np.sqrt(np.mean(obs.astype(float)**2));rmse=np.sqrt(np.mean(error**2));daily=np.sqrt(np.mean(daily_res))
 return dict(n=obs.size,rmse_K=rmse,bias_K=error.mean(),pearson=corr(obs,phys),zero_rhs_rmse_K=zero,rmse_skill_vs_zero=1-rmse/zero if zero else None,rmse_per_day=rmse/lead,observed_delta_std_K=obs.std(),physics_delta_std_K=phys.std(),daily_residual_rmse_Kday=daily,cancellation_ratio=(rmse/lead)/daily if daily else None)

def main():
 manifest=json.loads(Path(jobs()[0]['manifest']).read_text());ds=xr.open_dataset(manifest['data_path']);base=p.make_raw_base(ds,30,7);times=ds.time.values
 sea=base.ocean_mask>0;support=ocean_neighbor_support(torch.tensor(sea.astype('f4'))[None,None])[0,0].numpy()>0
 # Coast = valid cell touching an in-domain land cell; domain edges reported separately.
 land=torch.tensor((~sea).astype('f4'))[None,None];near=sum(shifted_neighbors(land))[0,0].numpy()>0
 coast=near[support];edge=np.zeros_like(sea);edge[[0,-1],:]=True;edge[:,[0,-1]]=True;regions={'all':np.ones(int(support.sum()),bool),'coast_adjacent_land':coast,'offshore':~coast,'domain_edge':edge[support]}
 assert sea.sum()==661 and support.sum()==660 and np.all(np.diff(times).astype('timedelta64[D]').astype(int)==1)
 agg=[];spatial=[];relationships=[];strata=[];regression=[]
 for jb,j in pair_jobs():
  m=json.loads(Path(j['manifest']).read_text());cfg=json.loads((Path(j['run_dir'])/'config.json').read_text());args=SimpleNamespace(**cfg['settings']);pa=p.p0_scale_args();lead=j['lead']
  assert args.entrainment_dt==1 and args.physics_mode=='p0_original'
  for split in ['train','val','test']:
   f=fields(j,split,['pred_raw','target_raw','we_raw','target_index']);b=fields(jb,split,['pred_raw','target_raw','target_index']);idx=f['target_index'];assert np.array_equal(idx,b['target_index']) and np.array_equal(f['target_raw'],b['target_raw']);starts=np.array(m[split+'_start_indices']);assert np.array_equal(idx,starts+29+lead)
   current=idx-lead;obs_delta=(base.target_array[idx]-base.target_array[current])[:,support].astype(float);assert np.isfinite(obs_delta).all()
   # Daily reanalysis forcing varies throughout the forecast horizon.
   for name in ['qnet','mlotst','u_ml','v_ml']:
    a=base.physics_arrays[name][current[:,None]+np.arange(lead)][...,sea];assert np.isfinite(a).all(),name
   modes=['linear_observed_zero_we','linear_observed_learned_we','daily_observed_learned_we','linear_predicted_learned_we']
   accumulated={k:[] for k in modes};squared={k:[] for k in modes};cursor=0
   with torch.no_grad():
    for batch in DataLoader(p.PhysicsScaleDataset(base,starts,lead),batch_size=32,shuffle=False):
     n=len(batch['Tm']);sl=slice(cursor,cursor+n);we=torch.tensor(f['we_raw'][sl]);target=batch['y_raw'];tm=batch['Tm'];mask=batch['mask'];pred=torch.tensor(f['pred_raw'][sl]);observed=torch.tensor(base.target_array[current[sl,None]+np.arange(lead+1)])
     assert we.ndim==4 and torch.isfinite(we[:,:,sea]).all()
     for mode in modes:
      if mode=='daily_observed_learned_we':
       arr=[]
       for step in range(lead):
        bb={**batch,**{k:batch[k][:,step:step+1] for k in ['qnet_seq','hm_seq','u_seq','v_seq']}}
        rhs,_=h._linear_rhs_per_day(observed[:,step+1:step+2],observed[:,step:step+1],mask,bb,pa,1,we);arr.append(rhs[:,0,support].numpy())
       rhs=np.stack(arr);dt=(observed[:,1:]-observed[:,:-1]).numpy()[:,:,support].transpose(1,0,2)
      else:
       end=pred if mode=='linear_predicted_learned_we' else target;wv=torch.zeros_like(we) if mode=='linear_observed_zero_we' else we
       r,rm=h._linear_rhs_per_day(end,tm,mask,batch,pa,lead,wv);rhs=r.reshape(lead,n,1,*sea.shape)[:,:,0,support].numpy();dt=((end-tm)/lead)[:,0,support].numpy()[None]
      residual=dt.astype(float)-rhs;cum=rhs.astype(float).sum(0);sq=(residual**2).mean(0)
      assert np.isfinite(cum).all() and np.isfinite(sq).all()
      # Exact telescoping identity and Jensen inequality: no clipping before integration.
      actual_delta=(pred-tm)[:,0,support].numpy() if mode=='linear_predicted_learned_we' else obs_delta[sl]
      np.testing.assert_allclose(residual.sum(0) if dt.shape[0]==lead else (np.broadcast_to(dt,rhs.shape).astype(float)-rhs).sum(0),actual_delta-cum,atol=5e-6,rtol=2e-5)
      assert np.all(((actual_delta-cum)/lead)**2<=sq+1e-6)
      accumulated[mode].append(cum);squared[mode].append(sq)
     cursor+=n
   assert cursor==len(idx);sr=seasons(times[idx]);data={};ep=(f['pred_raw'][:,0]-f['target_raw'][:,0])[:,support].astype(float);eb=(b['pred_raw'][:,0]-b['target_raw'][:,0])[:,support].astype(float);delta_se=ep**2-eb**2
   for mode in modes:
    integral=np.concatenate(accumulated[mode]);sq=np.concatenate(squared[mode]);delta=(f['pred_raw'][:,0]-base.target_array[current])[:,support] if mode=='linear_predicted_learned_we' else obs_delta
    residual=delta-integral;data[mode+'_cumulative_residual_K']=residual.astype('f4');data[mode+'_daily_residual_mse']=sq.astype('f4')
    for region,grid in regions.items():
     for season in ['all','DJF','MAM','JJA','SON']:
      keep=np.ones(len(idx),bool) if season=='all' else sr==season
      if keep.any() and grid.any():agg.append(dict(**meta(j),split=split,mode=mode,region=region,season=season,**summary(delta[keep][:,grid],integral[keep][:,grid],sq[keep][:,grid],lead)))
    if mode=='linear_observed_learned_we':
     for ci,(iy,ix) in enumerate(np.argwhere(support)):spatial.append(dict(**meta(j),split=split,iy=int(iy),ix=int(ix),latitude=float(base.lat2d[iy,ix]),longitude=float(base.lon2d[iy,ix]),coast=bool(coast[ci]),**summary(delta[:,ci],integral[:,ci],sq[:,ci],lead)))
    score=abs(residual);day_score=np.sqrt((residual**2).mean(1));cell_score=np.sqrt((residual**2).mean(0))
    relationships.append(dict(**meta(j),split=split,mode=mode,day_corr_residual_vs_pinn_rmse=corr(day_score,np.sqrt((ep**2).mean(1))),day_corr_residual_vs_delta_mse=corr(day_score,delta_se.mean(1)),cell_corr_residual_vs_delta_mse=corr(cell_score,delta_se.mean(0)),pooled_corr_abs_residual_vs_delta_se=corr(score,delta_se),coast_cells=int(coast.sum()),scope='association only; shared targets and learned we induce dependence'))
    for region,grid in regions.items():
     for season in ['all','DJF','MAM','JJA','SON']:
      keep=np.ones(len(idx),bool) if season=='all' else sr==season
      select=np.broadcast_to(keep[:,None]&grid[None,:],score.shape)
      if not select.any():continue
      for top in [1.,.5,.25,.1]:
       threshold=np.quantile(score[select],1-top);sel=select&(score>=threshold);rmseb=np.sqrt((eb[sel]**2).mean());rmsep=np.sqrt((ep[sel]**2).mean())
       strata.append(dict(**meta(j),split=split,mode=mode,region=region,season=season,top_fraction=top,threshold_abs_cum_K=threshold,n=int(sel.sum()),baseline_rmse=rmseb,physics_rmse=rmsep,improvement_percent=100*(rmseb-rmsep)/rmseb,delta_mse=delta_se[sel].mean()))
   np.savez_compressed(OUT/'fields'/f"physics_r{j['ratio']}_l{lead}_s{j['seed']}_{split}.npz",target_index=idx,support=support,observed_delta_K=obs_delta.astype('f4'),**data)
   regression.append(dict(**meta(j),split=split,windows=len(idx),finite=True,telescoping=True,jensen=True,model_inference=False));print('PHYSICS',meta(j),split,flush=True)
  # Durable partial results per condition; only new directory.
  for name,rows in [('cumulative_physics.csv',agg),('physics_spatial.csv',spatial),('residual_error_relationship.csv',relationships),('residual_error_strata.csv',strata)]:csvsave(name,rows)
 ds.close();save(OUT/'physics_regression.json',regression);finish('physics',conditions=24,splits=72,physics_cells=660,coast_cells=int(coast.sum()))
if __name__=='__main__':main()
