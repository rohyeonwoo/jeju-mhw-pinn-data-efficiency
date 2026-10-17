from cpu_diagnostics_common import *
import xarray as xr
from mhw import subset_climatology as c

def describe(a,weights=None):
 a=np.asarray(a,dtype=float)
 if weights is None:return dict(mean=float(a.mean()),std=float(a.std()),variance=float(a.var()),count=int(a.size))
 w=np.asarray(weights,dtype=float);w=w/w.sum();mu=float(np.dot(w,a.mean(1)));var=float(np.dot(w,(a*a).mean(1))-mu*mu)
 return dict(mean=mu,std=math.sqrt(max(var,0)),variance=max(var,0),count=int(np.count_nonzero(weights)*a.shape[1]))

def main():
 # Reuse exhaustive tensor occurrence scan only after exact v2/v3 dependency checks.
 source=ROOT/'outputs/v2_final_preflight_20261008/input_feature_split_details.csv';old=readcsv(source);lookup={(int(r['ratio']),int(r['seed']),int(r['lead']),r['split'],r['variable']):r for r in old}
 j0=jobs()[0];m0=json.loads(Path(j0['manifest']).read_text());ds=xr.open_dataset(m0['data_path']);times=ds.time.values;bins=c.calendar_bins(times);sea=np.isfinite(ds.ocean_mask_static.values);raw=c.load_raw_anomaly_fields(ds,m0['raw_thetao_path']);coverage=[];doyrows=[];rough=[];dist=[];features=[];reg=[];seen=set()
 for jb,j in pair_jobs():
  m=json.loads(Path(j['manifest']).read_text());parent=json.loads(Path(m['parent_manifest']).read_text());assert sha(m['parent_manifest'])==m['parent_manifest_sha256']
  for key in ['statistics','input_time_indices','train_start_indices','val_start_indices','test_start_indices','climatology_sha256','climatology_rule','data_sha256','raw_thetao_sha256']:assert m[key]==parent[key],key
  cp=Path(j['manifest']).parent/m['climatology_file'];assert sha(cp)==m['climatology_sha256']
  with np.load(cp) as z:clims={k:z[k].copy() for k in c.ANOMALIES};counts=z['calendar_counts'];fitted=z['input_time_indices'];assert np.array_equal(fitted,m['input_time_indices']);assert np.array_equal(counts,np.bincount(bins[fitted],minlength=366))
  weights={}
  for split in ['train','val','test']:
   starts=np.array(m[split+'_start_indices']);w=np.bincount((starts[:,None]+np.arange(30)).ravel(),minlength=len(times));weights[split]=w
   for name in m['input_feature_names']:
    rr=lookup[j['ratio'],j['seed'],j['lead'],'validation' if split=='val' else split,name];assert np.allclose([float(rr['normalization_mean']),float(rr['normalization_std'])],m['statistics'][name],rtol=1e-12,atol=1e-12)
    features.append(dict(**meta(j),split=split,variable=name,z_mean=float(rr['mean']),z_std=float(rr['std']),max_abs_z=max(abs(float(rr['min'])),abs(float(rr['max']))),fraction_abs_z_gt5=float(rr['fractions_gt5']),fraction_abs_z_gt10=float(rr['fractions_gt10']),fraction_abs_z_gt20=float(rr['fractions_gt20']),source=str(source),basis='actual30-day input-history occurrences; v2/v3 dependencies identical'))
  # Static geography inputs are fixed and normalized by the same manifest values.
  lat=np.broadcast_to(ds.latitude.values[:,None],sea.shape);lon=np.broadcast_to(ds.longitude.values[None,:],sea.shape)
  for name,v in [('ocean_mask_static',sea.astype(float)),('lat2d',lat),('lon2d',lon)]:
   mu,sd=m['statistics'][name];z=(v[sea]-mu)/sd
   for split in ['train','val','test']:features.append(dict(**meta(j),split=split,variable=name,z_mean=float(z.mean()),z_std=float(z.std()),max_abs_z=float(abs(z).max()),fraction_abs_z_gt5=float(np.mean(abs(z)>5)),fraction_abs_z_gt10=float(np.mean(abs(z)>10)),fraction_abs_z_gt20=float(np.mean(abs(z)>20)),source='manifest and immutable grid',basis='ocean nodes, static repeated each history day'))
  for name in c.ANOMALIES:
   values=raw[name][:,sea];cl=clims[name][:,sea];anom=values-cl[bins];assert np.isfinite(anom).all();mu,sd=m['statistics'][name]
   for split,w in weights.items():
    idx=np.flatnonzero(w);a=anom[idx];v=values[idx];stat=describe(a);tensor=describe(a,w[idx]);rs=describe(v)
    dist.append(dict(**meta(j),split=split,variable=name,unique_dates=len(idx),raw_mean=rs['mean'],raw_std=rs['std'],anomaly_mean=stat['mean'],anomaly_std=stat['std'],anomaly_variance_to_raw=stat['variance']/rs['variance'],history_weighted_anomaly_mean=tensor['mean'],history_weighted_anomaly_std=tensor['std'],unique_z_std=stat['std']/sd,history_z_std=tensor['std']/sd,normalization_std=sd))
    row=lookup[j['ratio'],j['seed'],j['lead'],'validation' if split=='val' else split,name]
    assert abs(tensor['std']/sd-float(row['std']))<2e-5,(name,split)
   # Singleton fitted DOYs force the corresponding training anomaly to zero.
   single=counts[bins[fitted]]==1;assert np.all(abs(anom[fitted[single]])<2e-5)
   if (j['ratio'],j['seed']) not in seen:
    step=np.roll(cl,-1,axis=0)-cl
    rough.append(dict(ratio=j['ratio'],seed=j['seed'],variable=name,climatology_daystep_rms=float(np.sqrt((step.astype(float)**2).mean())),climatology_daystep_abs_p95=float(np.quantile(abs(step),.95)),climatology_daystep_abs_max=float(abs(step).max()),singleton_training_date_fraction=float(single.mean()),singleton_anomaly_absmax=float(abs(anom[fitted[single]]).max()) if single.any() else 0.))
  if (j['ratio'],j['seed']) not in seen:
   coverage.append(dict(ratio=j['ratio'],seed=j['seed'],unique_input_days=len(fitted),empty_doy=int((counts==0).sum()),empty_fraction=float(np.mean(counts==0)),singleton_doy=int((counts==1).sum()),singleton_fraction=float(np.mean(counts==1)),count_min=int(counts.min()),count_median=float(np.median(counts)),count_max=int(counts.max()),feb29_count=int(counts[59]),land_climatology_zero=all(np.all(v[:,~sea]==0) for v in clims.values()),finite=all(np.isfinite(v).all() for v in clims.values())))
   for i,n in enumerate(counts):doyrows.append(dict(ratio=j['ratio'],seed=j['seed'],leap_aligned_bin=i,month_day=str(np.datetime64('2000-01-01')+np.timedelta64(i,'D'))[5:],count=int(n)))
   seen.add((j['ratio'],j['seed']))
  reg.append(dict(**meta(j),v2_v3_statistics_and_climatology_equal=True,history_distribution_recomputed=True,singleton_zero_verified=True));print('CLIMATOLOGY',meta(j),flush=True)
 for name,rows in [('climatology_coverage.csv',coverage),('doy_counts.csv',doyrows),('climatology_roughness.csv',rough),('anomaly_distributions.csv',dist),('all_input_zscores.csv',features)]:csvsave(name,rows)
 save(OUT/'climatology_regression.json',reg);ds.close();finish('climatology',unique_climatologies=12,manifests=24,features=22,anomaly_channels=7,limitation='Quality indicators without independent climatology truth. Train-vs-heldout differences include time trends and training reuse; no causal attribution to climatology alone.')
if __name__=='__main__':main()
