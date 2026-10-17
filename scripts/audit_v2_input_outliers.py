"""Read-only v2 input analysis; CPU only, no model training or data changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json, csv, hashlib
import numpy as np
import xarray as xr
from mhw import subset_climatology as c, subset_preprocessing as p
OUT=ROOT/'outputs/v2_final_preflight_20261008'
V2=ROOT/'data/subsets/preprocessing_subset_v2'

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def weights(starts,nt,history=30):
    return np.bincount((np.asarray(starts)[:,None]+np.arange(history)).ravel(),minlength=nt)

def describe(z,w,empty):
    use=w>0;v=z[use].astype(np.float64);ww=w[use].astype(np.float64)
    n=int(ww.sum()*z.shape[1]);mean=float(np.dot(v.sum(axis=1),ww)/n)
    second=float(np.dot((v*v).sum(axis=1),ww)/n)
    counts={str(t):int(np.dot((np.abs(v)>t).sum(axis=1),ww)) for t in (5,10,20)}
    missing=empty[use];ne=int(ww[missing].sum()*z.shape[1]);no=n-ne
    ec={str(t):int(np.dot((np.abs(v[missing])>t).sum(axis=1),ww[missing])) for t in (5,10,20)}
    return dict(n=n,min=float(v.min()),max=float(v.max()),mean=mean,std=float(np.sqrt(max(0,second-mean*mean))),
        counts=counts,fractions={t:a/n for t,a in counts.items()},empty_doy_n=ne,empty_doy_counts=ec,
        empty_doy_fractions={t:a/ne if ne else None for t,a in ec.items()},
        observed_doy_fractions={t:(counts[t]-ec[t])/no if no else None for t in counts})

def main():
    # Independent small reference verifies exact tensor-occurrence weighting.
    z=np.array([[1.,6.],[21.,-11.],[2.,3.]])
    w=np.array([2,1,0]);d=describe(z,w,np.array([False,True,False]))
    literal=np.repeat(z,w,axis=0).ravel()
    assert d['n']==len(literal) and d['counts']=={'5':4,'10':2,'20':1}
    np.testing.assert_allclose([d['mean'],d['std']],[literal.mean(),literal.std()])
    old={str(f):p.sha256_file(f) for directory in [V2,V2.parent/'preprocessing_subset_v1'] for f in directory.rglob('*') if f.is_file()}
    first=json.loads((V2/'ratio_10_lead7_seed42.json').read_text())
    assert p.sha256_file(first['data_path'])==first['data_sha256']
    assert p.sha256_file(first['raw_thetao_path'])==first['raw_thetao_sha256']
    ds=xr.open_dataset(first['data_path']);base=p.make_raw_base(ds)
    raw=c.load_raw_anomaly_fields(ds,first['raw_thetao_path']);bins=c.calendar_bins(ds.time.values)
    ocean=base.ocean_mask.astype(bool);ij=np.argwhere(ocean);dates=ds.time.values.astype('datetime64[D]').astype(str)
    lats=ds.latitude.values;lons=ds.longitude.values;nt=len(dates)
    details=[];summaries=[];locations=[];scales=[];anomaly_groups=[]
    for ratio in (10,25,50,100):
      for seed in (42,7,123):
        paths={lead:V2/f'ratio_{ratio}_lead{lead}_seed{seed}.json' for lead in (7,14)}
        ms={lead:json.loads(path.read_text()) for lead,path in paths.items()}
        m=ms[7];clims=c.read_climatology(m,paths[7]);rebased=c.apply_to_base(base,raw,bins,clims)
        with np.load(V2/m['climatology_file']) as ar:empty=ar['calendar_counts'][bins]==0
        ws={lead:{split:weights(mm[key],nt) for split,key in [('train','train_start_indices'),('validation','val_start_indices'),('test','test_start_indices')]} for lead,mm in ms.items()}
        for lead,mm in ms.items():
            assert mm['statistics']==ms[7]['statistics'] or mm['statistics']['sst_anom']==ms[7]['statistics']['sst_anom']
            for n in p.BASE_DYNAMIC_VARS:assert mm['statistics'][n]==m['statistics'][n]
            scales.append(dict(ratio=ratio,seed=seed,lead=lead,target_mean=mm['statistics']['target'][0],target_std=mm['statistics']['target'][1],rhs_mse_kday2=mm['physics_rhs_scale']))
        group=[];stored={}
        active=np.logical_or.reduce([w>0 for d in ws.values() for w in d.values()])
        for name in p.BASE_DYNAMIC_VARS:
            mu,sd=m['statistics'][name];assert np.isfinite([mu,sd]).all() and sd>1e-6
            values=rebased.dynamic_arrays[name][:,ocean];assert np.isfinite(values).all()
            z=((values-mu)/sd).astype(np.float32);assert np.isfinite(z).all()
            affected=empty if name in c.ANOMALIES else np.zeros(nt,dtype=bool)
            for lead in (7,14):
                for split,w in ws[lead].items():
                    row=dict(ratio=ratio,seed=seed,lead=lead,split=split,variable=name,normalization_mean=mu,normalization_std=sd,**describe(z,w,affected))
                    details.append(row);group.append(row)
            # All >5 coordinates on dates actually used by an input window; no duplicates.
            ti,cell=np.nonzero((np.abs(z)>5)&active[:,None])
            stored[name+'_time_index']=ti.astype(np.int32);stored[name+'_ocean_node']=cell.astype(np.int16);stored[name+'_z']=z[ti,cell]
            # Top 20 plus global extrema locations, with per-split membership saved.
            flat=np.where(active[:,None],np.abs(z),-1).ravel();top=np.argpartition(flat,-20)[-20:];top=top[np.argsort(flat[top])[::-1]]
            for k in top:
                t,node=np.unravel_index(k,z.shape);iy,ix=ij[node]
                locations.append(dict(ratio=ratio,seed=seed,variable=name,date=dates[t],time_index=int(t),ocean_node=int(node),latitude=float(lats[iy]),longitude=float(lons[ix]),z=float(z[t,node]),raw_input=float(values[t,node]),empty_climatology_bin=bool(affected[t]),doy=int(bins[t]+1),occurrences={str(l):{s:int(w[t]) for s,w in ws[l].items()} for l in ws}))
        np.savez_compressed(OUT/f'outlier_coordinates_gt5_ratio{ratio}_seed{seed}.npz',ocean_grid_indices=ij,latitude=lats,longitude=lons,dates=dates,**stored)
        for lead in (7,14):
            rr=[x for x in group if x['lead']==lead];n=sum(x['n'] for x in rr)
            winner=max(rr,key=lambda x:max(abs(x['min']),abs(x['max'])))
            summaries.append(dict(ratio=ratio,seed=seed,lead=lead,max_abs_z=max(abs(winner['min']),abs(winner['max'])),max_variable=winner['variable'],max_split=winner['split'],n=n,counts={str(t):sum(x['counts'][str(t)] for x in rr) for t in (5,10,20)},fractions={str(t):sum(x['counts'][str(t)] for x in rr)/n for t in (5,10,20)},status='FINITE; EXTREME TAILS REQUIRE INTERPRETATION'))
        save(OUT/'input_summary.json',summaries)
        print(f'Analyzed ratio={ratio} seed={seed}; max|z|={summaries[-1]["max_abs_z"]:.5f}',flush=True)
        del rebased,z,stored
    save(OUT/'input_feature_split_details.json',details);save(OUT/'outlier_top_locations.json',locations);save(OUT/'target_rhs_scales.json',scales)
    assert all(p.sha256_file(f)==h for f,h in old.items())
    save(OUT/'input_analysis_verification.json',dict(status='PASS_NUMERICAL_FINITE_ONLY',source_hashes_verified=True,all_v1_v2_files_preserved=True,weighted_reference_test='PASS',variables=19,manifest_count=24,nonfinite_ocean=0,normalization_std_near_zero=0,denominator='Actual input tensor occurrences, counting repeated history dates; all three splits pooled only in summary. Feature/split details separate.',coordinates='NPZ stores all |z|>5 unique date/ocean coordinates across input dates; top JSON lists top20 per variable and group. Dates not used by any sample excluded.',training_executed=False,gpu_used=False))
    ds.close()

if __name__=='__main__':main()
