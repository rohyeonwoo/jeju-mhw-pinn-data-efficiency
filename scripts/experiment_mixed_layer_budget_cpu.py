#!/usr/bin/env python3
"""CPU-only offline mixed-layer heat-budget sensitivity audit for Jeju.

No model is loaded, trained, or inferred. It computes daily diagnostic closure
along the *observed* GLORYS mixed-layer temperature trajectory. The output is
NOT a forecast RMSE or a reproduction of a learned-we P0 experiment.

Requirements: numpy, xarray, a NetCDF backend (netcdf4 or h5netcdf on server).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

RHO = 1025.0
CP = 3990.0
DAY = 86400.0
EARTH_R = 6371000.0


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--budget2d', type=Path, required=True, help='Existing budget2d NetCDF, unchanged')
    p.add_argument('--glorys3d', type=Path, required=True, help='Existing 0.25-degree 3D thetao/uo/vo NetCDF')
    p.add_argument('--output', type=Path, required=True, help='Must NOT be an existing path')
    p.add_argument('--leads', default='7,14', help='Comma-separated lead days')
    p.add_argument('--targets', choices=['glorys_ml','noaa','both'], default='both',
                   help='GLORYS Tm is physically appropriate; NOAA is diagnostic cross-target comparison')
    p.add_argument('--history', type=int, default=30)
    p.add_argument('--split', choices=['train','val','test'], default='val')
    p.add_argument('--max-windows', type=int, default=48, help='Windows per lead; 0 means all')
    p.add_argument('--starts-json', type=Path, default=None,
                   help='Optional JSON list, or object with start_indices, to use exact EXISTING sample starts')
    p.add_argument('--kh-values', default='0,25,100,300', help='Horizontal eddy diffusivities m^2/s')
    p.add_argument('--kd', type=float, default=0.06, help='Simple single-band optical attenuation m^-1')
    p.add_argument('--penetration', choices=['off','on'], default='off',
                   help='If on: include both off/on arms; require explicit/recognized ssr units')
    p.add_argument('--ssr-units', choices=['auto','W_m2','J_m2_day'], default='auto')
    p.add_argument('--assume-qnet-wm2', action='store_true', help='Allow unknown qnet metadata; verify separately')
    p.add_argument('--min-ml-depth', type=float, default=5.0)
    p.add_argument('--max-ml-depth', type=float, default=None,
                   help='Optional cut to exclude very deep MLDs; default: 3D vertical coverage')
    return p.parse_args()


def detect_dims(ds, want3d=False):
    var = ds['thetao'] if want3d else ds['sst_mean']
    dims = var.dims
    la = next((d for d in dims if d.lower() in ('latitude','lat','nav_lat')), None)
    lo = next((d for d in dims if d.lower() in ('longitude','lon','nav_lon')), None)
    ti = next((d for d in dims if d.lower() in ('time','time_counter','valid_time')), None)
    if any(x is None for x in (la, lo, ti)):
        raise ValueError(f'Cannot detect time/lat/lon in {dims}. Inspect NetCDF dimensions.')
    depth = None
    if want3d:
        z = [d for d in dims if d not in (la,lo,ti)]
        if len(z)!=1:
            raise ValueError(f'Expected one depth dim in thetao: {dims}')
        depth=z[0]
    return {'time':ti,'lat':la,'lon':lo,'depth':depth}


def read2d(ds, name, dims, i=None):
    da=ds[name]
    idx={dims['time']:int(i)} if i is not None and dims['time'] in da.dims else {}
    arr=da.isel(idx).transpose(dims['lat'],dims['lon']).values
    return np.asarray(arr,dtype=np.float64)


def read3d(ds, name, dims, i):
    arr=ds[name].isel({dims['time']:int(i)}).transpose(dims['depth'],dims['lat'],dims['lon']).values
    return np.asarray(arr,dtype=np.float64)


def check_alignment(b, g, bd, gd):
    for tag in ('lat','lon'):
        x=np.asarray(b[bd[tag]].values)
        y=np.asarray(g[gd[tag]].values)
        if x.shape!=y.shape or not np.allclose(x,y,rtol=0,atol=2e-5):
            raise ValueError(f'{tag} mismatch between 2D and 3D; regridding is required, not silently performed')
        if x.ndim!=1 or not np.all(np.diff(x)>0):
            raise ValueError(f'{tag} should be increasing 1-D coordinates')
    bt=np.asarray(b[bd['time']].values)
    gt=np.asarray(g[gd['time']].values)
    if bt.shape!=gt.shape or not np.array_equal(bt,gt):
        raise ValueError('Dates differ between the two files; use matching time ranges/versions before auditing')
    if len(bt)<2:
        raise ValueError('Insufficient time data')
    # A daily interval must be exactly 1 day for the K/day conversion used here.
    d=np.diff(bt)
    if np.issubdtype(d.dtype,np.timedelta64) and not np.all(d==np.timedelta64(1,'D')):
        raise ValueError('Time spacing is not daily; this script assumes 1 day')
    z=np.asarray(g[gd['depth']].values,dtype=np.float64)
    if z.ndim!=1 or len(z)<2 or not np.all(z>=0) or not np.all(np.diff(z)>0):
        raise ValueError('GLORYS depth coordinate must be positive-down and strictly increasing')
    return z,bt


def vertical_edges(depth):
    # Approximate cell interfaces using layer-center midpoints.
    # Actual GLORYS layer bounds, if available, should replace this approximation.
    edges=np.empty(len(depth)+1,dtype=np.float64)
    edges[0]=0.0
    edges[1:-1]=0.5*(depth[:-1]+depth[1:])
    edges[-1]=depth[-1]+0.5*(depth[-1]-depth[-2])
    return edges


def layer_mean_and_below(theta, depth, edges, h, valid_ml):
    # theta [depth, lat, lon] ; h [lat, lon]
    weights=np.clip(np.minimum(h[None,:,:],edges[1:,None,None])-edges[:-1,None,None],0,None)
    use=weights>1e-10
    complete=np.all((~use)|np.isfinite(theta),axis=0)
    weight_sum=weights.sum(axis=0)
    deep_enough=h<=edges[-1]
    ok=valid_ml & (h>0) & complete & deep_enough & (weight_sum>0)
    tm=np.sum(np.where(use & np.isfinite(theta),weights*theta,0),axis=0)/np.maximum(weight_sum,1e-12)
    tm=np.where(ok,tm,np.nan)
    below_candidates=(depth[:,None,None]>h[None,:,:]) & np.isfinite(theta)
    has_below=np.any(below_candidates,axis=0)
    idx=np.argmax(below_candidates,axis=0)
    tb=np.take_along_axis(theta,idx[None,:,:],axis=0)[0]
    tb=np.where(ok&has_below,tb,np.nan)
    return tm,tb,ok


def spacing(lat,lon):
    if np.any(np.diff(lat)<=0) or np.any(np.diff(lon)<=0):
        raise ValueError('Latitude/longitude must be ascending')
    dy=EARTH_R*np.deg2rad(float(np.median(np.diff(lat))))
    dx=EARTH_R*np.cos(np.deg2rad(lat))[:,None]*np.deg2rad(float(np.median(np.diff(lon))))
    return dx,dy


def slopes_ocean(field,valid,dx,dy):
    # Central ocean-neighbor gradient, one-sided at coast; never treats land as 0C.
    f=np.where(valid,field,0.0)
    ddx=np.zeros_like(f)
    ddy=np.zeros_like(f)
    lv=np.zeros_like(valid);lv[:,1:]=valid[:,:-1]
    rv=np.zeros_like(valid);rv[:,:-1]=valid[:,1:]
    uv=np.zeros_like(valid);uv[1:,:]=valid[:-1,:]
    dv=np.zeros_like(valid);dv[:-1,:]=valid[1:,:]
    left=np.zeros_like(f);left[:,1:]=f[:,:-1]
    right=np.zeros_like(f);right[:,:-1]=f[:,1:]
    up=np.zeros_like(f);up[1:,:]=f[:-1,:]
    down=np.zeros_like(f);down[:-1,:]=f[1:,:]
    both=lv&rv; ddx=np.where(both,(right-left)/2,np.where(rv,right-f,np.where(lv,f-left,0)))/dx
    both=uv&dv; ddy=np.where(both,(down-up)/2,np.where(dv,down-f,np.where(uv,f-up,0)))/dy
    return np.where(valid,ddx,np.nan),np.where(valid,ddy,np.nan)


def horizontal_diffusion(field,valid,dx,dy):
    # Finite-difference no-flux on masked faces, constant Kh. Approximate on lat-lon grid.
    # Returns Laplacian (K / m2); NOT an exact conservative finite-volume coastal operator.
    f=np.where(valid,field,0.0)
    out=np.zeros_like(f)
    edge=valid[:,:-1]&valid[:,1:]
    delta=np.where(edge,(f[:,1:]-f[:,:-1])/(dx**2),0)
    out[:,:-1]+=delta
    out[:,1:]-=delta
    edge=valid[:-1,:]&valid[1:,:]
    delta=np.where(edge,(f[1:,:]-f[:-1,:])/(dy**2),0)
    out[:-1,:]+=delta
    out[1:,:]-=delta
    return np.where(valid,out,np.nan)


def units_mode(attrs,provided,field):
    if provided!='auto':return provided
    raw=str(attrs.get('units','')).strip().lower().replace(' ','').replace('**','^')
    if not raw:
        raise ValueError(f'{field} has no units metadata: specify --ssr-units after verifying original product')
    if 'w' in raw and ('m-2' in raw or 'm^-2' in raw or '/m2' in raw or '/m^2' in raw or 'm2' in raw): return 'W_m2'
    if ('j' in raw) and ('m-2' in raw or 'm^-2' in raw or '/m2' in raw or '/m^2' in raw or 'm2' in raw):
        return 'J_m2_day'
    raise ValueError(f'Unknown {field} units: {attrs.get("units")!r}')


def get_index_starts(total,history,lead,split,cap,custom):
    n=total-history-lead+1
    if n<=0:raise ValueError('History+lead too long for data')
    ntrain=int(n*.7)
    nval=int(n*.15)
    if custom is None:
        starts={'train':np.arange(0,ntrain),'val':np.arange(ntrain,ntrain+nval),
                'test':np.arange(ntrain+nval,n)}[split]
    else:
        parsed=json.loads(custom.read_text(encoding='utf8'))
        if isinstance(parsed,list): starts=np.asarray(parsed,dtype=int)
        elif isinstance(parsed,dict) and 'start_indices' in parsed:starts=np.asarray(parsed['start_indices'],dtype=int)
        else:raise ValueError('--starts-json accepts [indices] or {"start_indices": [indices]}')
        if starts.ndim!=1:raise ValueError('Index list must be 1-dimensional')
        starts=np.unique(starts)
        if np.any(starts<0) or np.any(starts>=n):raise ValueError('Index outside valid sample range')
        expected={'train':(0,ntrain),'val':(ntrain,ntrain+nval),'test':(ntrain+nval,n)}[split]
        if np.any(starts<expected[0]) or np.any(starts>=expected[1]):
            raise ValueError(f'Custom indices contain samples outside requested {split} split')
    if cap>0 and len(starts)>cap:
        starts=starts[np.unique(np.linspace(0,len(starts)-1,cap,dtype=int))]
    return starts


def metrics(error):
    if len(error)==0:return dict(n=0,rmse_k=None,mae_k=None,bias_k=None)
    a=np.asarray(error,dtype=np.float64)
    return dict(n=len(a),rmse_k=float(np.sqrt(np.mean(a*a))),mae_k=float(np.mean(abs(a))),bias_k=float(np.mean(a)))


def write_csv(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)


def main():
    a=arguments()
    if a.history<1 or a.max_windows<0 or a.kd<=0 or a.min_ml_depth<=0:raise ValueError('Invalid parameter')
    leads=[int(x) for x in a.leads.split(',')];khs=[float(x) for x in a.kh_values.split(',')]
    if any(x<=0 for x in leads) or any(k<0 for k in khs):raise ValueError('Invalid leads/Kh')
    if not a.budget2d.is_file() or not a.glorys3d.is_file():raise FileNotFoundError('Input NetCDF not found')
    a.output.mkdir(parents=True,exist_ok=False)
    with xr.open_dataset(a.budget2d) as b,xr.open_dataset(a.glorys3d) as g:
        bd=detect_dims(b);gd=detect_dims(g,True)
        required2=['sst_mean','qnet','ssr','mlotst','ocean_mask_static']
        required3=['thetao','uo','vo']
        for k in required2:
            if k not in b:raise KeyError(f'budget2d missing {k}')
        for k in required3:
            if k not in g:raise KeyError(f'GLORYS 3D missing {k}')
        depth,times=check_alignment(b,g,bd,gd)
        edges=vertical_edges(depth)
        lat=np.asarray(b[bd['lat']].values,dtype=float)
        lon=np.asarray(b[bd['lon']].values,dtype=float)
        dx,dy=spacing(lat,lon)
        m0=read2d(b,'ocean_mask_static',bd)
        ocean=np.isfinite(m0) & (m0>0.5)
        assert ocean.sum()>0
        q_units=str(b['qnet'].attrs.get('units','')).lower().replace(' ','')
        good_q=('w' in q_units and ('m-2' in q_units or 'm^-2' in q_units or '/m2' in q_units or '/m^2' in q_units or 'm2' in q_units))
        if not good_q and not a.assume_qnet_wm2:raise ValueError(f'Unverified qnet units={b["qnet"].attrs.get("units")!r}; use --assume-qnet-wm2 only after verifying W/m2')
        ssr_unit = units_mode(b['ssr'].attrs,a.ssr_units,'ssr') if a.penetration=='on' else None
        pen_modes=[False,True] if a.penetration=='on' else [False]
        # All source samples are diagnostically observed in hindsight. NO operational future forcing.
        per_window=[]
        term_rows=[]
        for lead in leads:
            starts=get_index_starts(len(times),a.history,lead,a.split,a.max_windows,a.starts_json)
            print(f'Lead{lead}: {len(starts)} windows ({a.split})',flush=True)
            for run_i,start in enumerate(starts):
                t0=int(start)+a.history-1
                cum={}
                valid_across=ocean.copy()
                term_sums={'flux':np.zeros_like(ocean,dtype=float),'adv':np.zeros_like(ocean,dtype=float),
                           'entrain':np.zeros_like(ocean,dtype=float),'penetration':np.zeros_like(ocean,dtype=float),
                           'laplacian':np.zeros_like(ocean,dtype=float)}
                tm_first=tm_last=None
                for day in range(lead):
                    t=t0+day
                    # Read one day at a time; no full 3D series is loaded in RAM.
                    h=read2d(b,'mlotst',bd,t)
                    hnext=read2d(b,'mlotst',bd,t+1)
                    q=read2d(b,'qnet',bd,t)
                    sw=read2d(b,'ssr',bd,t) if a.penetration=='on' else np.zeros_like(q)
                    if ssr_unit=='J_m2_day':sw=sw/DAY
                    th=read3d(g,'thetao',gd,t)
                    thnext=read3d(g,'thetao',gd,t+1)
                    u3=read3d(g,'uo',gd,t)
                    v3=read3d(g,'vo',gd,t)
                    h_good=np.isfinite(h)&(h>0)&np.isfinite(hnext)&(hnext>0)
                    if a.max_ml_depth:h_good&=(h<=a.max_ml_depth)&(hnext<=a.max_ml_depth)
                    h_use=np.maximum(h,a.min_ml_depth)
                    hnext_use=np.maximum(hnext,a.min_ml_depth)
                    tm,tb,ok=layer_mean_and_below(th,depth,edges,h_use,h_good)
                    tm_next,_,ok_next=layer_mean_and_below(thnext,depth,edges,hnext_use,h_good)
                    um,_,uok=layer_mean_and_below(u3,depth,edges,h_use,h_good)
                    vm,_,vok=layer_mean_and_below(v3,depth,edges,h_use,h_good)
                    grad_ok=ocean & ok & ok_next & uok & vok
                    # Tb is needed only where MLD actually deepens. Else entrainment=0.
                    deepening=np.isfinite(hnext_use-h_use)&(hnext_use>h_use)
                    tb_ok=(~deepening)|np.isfinite(tb)
                    valid=grad_ok&tb_ok&np.isfinite(q)&np.isfinite(sw)
                    gx,gy=slopes_ocean(tm,grad_ok,dx,dy)
                    lap=horizontal_diffusion(tm,grad_ok,dx,dy)
                    flux=DAY*q/(RHO*CP*h_use)
                    adv=-DAY*(um*gx+vm*gy)
                    # Using MLD growth as an entrainment proxy, NOT diagnosed vertical pumping.
                    # m/day * K / m = K/day (we=0 when MLD shoals).
                    entrain=-np.maximum(0.0,hnext_use-h_use)*np.where(deepening,tm-tb,0.0)/h_use
                    penetration=DAY*sw*np.exp(-a.kd*h_use)/(RHO*CP*h_use)
                    terms={'flux':flux,'adv':adv,'entrain':entrain,
                           'penetration':penetration,'laplacian':DAY*lap}
                    for name,arr in terms.items():
                        valid&=np.isfinite(arr)
                        term_sums[name]+=np.where(np.isfinite(arr),arr,0)
                    valid_across&=valid
                    if day==0:tm_first=tm
                    if day==lead-1:tm_last=tm_next
                if not valid_across.any():
                    print(f'WARNING: no valid cells at start={start} Lead={lead}',flush=True)
                    continue
                observed_glorys=tm_last-tm_first
                observed_noaa=read2d(b,'sst_mean',bd,t0+lead)-read2d(b,'sst_mean',bd,t0)
                requested_targets=({'glorys_ml':observed_glorys,'noaa':observed_noaa}
                                   if a.targets=='both' else
                                   {a.targets:observed_glorys if a.targets=='glorys_ml' else observed_noaa})
                for pen in pen_modes:
                    for kh in khs:
                        for ent in (False,True):
                            rhs=term_sums['flux']-(term_sums['penetration'] if pen else 0)+kh*term_sums['laplacian']
                            if ent:rhs=rhs+term_sums['entrain']
                            for target_name,observed in requested_targets.items():
                                eval_mask=valid_across & np.isfinite(observed)
                                if not np.any(eval_mask): continue
                                for adv in (False,True):
                                    total=rhs+(term_sums['adv'] if adv else 0)
                                    err=observed-total
                                    e=err[eval_mask]
                                    per_window.append(dict(lead=lead,target=target_name,start=int(start),current_index=t0,
                                        date=str(times[t0])[:10],split=a.split,
                                        penetration=pen,kh_m2s=kh,entrainment_proxy=ent,advection=adv,
                                        n_cells=len(e),sse_k2=float(np.sum(e*e)),sae_k=float(np.sum(abs(e))),
                                        bias_sum_k=float(np.sum(e)),obs2_sum=float(np.sum(observed[eval_mask]**2)),
                                        sum_a2=float(np.sum(term_sums['adv'][eval_mask]**2)),
                                        sum_e0a=float(np.sum((observed-rhs)[eval_mask]*term_sums['adv'][eval_mask]))))
                for term,value in term_sums.items():
                    vals=value[valid_across]
                    term_rows.append(dict(lead=lead,start=int(start),term=term,
                                          mean_k=float(np.mean(vals)),rms_k=float(np.sqrt(np.mean(vals**2))),
                                          n_cells=len(vals)))
                if run_i%20==19:print(f'  {run_i+1}/{len(starts)} processed',flush=True)
        if not per_window:raise RuntimeError('No valid windows; check MLD coverage, masks and depth coordinates')
        write_csv(a.output/'window_metrics.csv',per_window)
        write_csv(a.output/'term_diagnostics.csv',term_rows)
        # Aggregate by actual cell counts, always comparing same domain for each window.
        grouping={}
        for row in per_window:
            k=(row['lead'],row['target'],row['penetration'],row['kh_m2s'],row['entrainment_proxy'],row['advection'])
            t=grouping.setdefault(k,dict(n=0,n_windows=0,sse=0.,sae=0.,signed=0.,obs2=0.,a2=0.,e0a=0.))
            t['n']+=row['n_cells'];t['n_windows']+=1
            for a1,b1 in [('sse','sse_k2'),('sae','sae_k'),('signed','bias_sum_k'),('obs2','obs2_sum'),('a2','sum_a2'),('e0a','sum_e0a')]:t[a1]+=row[b1]
        rows=[]
        for key,t in sorted(grouping.items()):
            lead,target_name,pen,kh,ent,adv=key
            rows.append(dict(lead=lead,target=target_name,penetration=pen,kh_m2s=kh,entrainment_proxy=ent,advection=adv,
                 n_window_cells=t['n'],n_windows=t['n_windows'],
                 rmse_k=math.sqrt(t['sse']/t['n']),mae_k=t['sae']/t['n'],
                 bias_k=t['signed']/t['n'],zero_change_rmse_k=math.sqrt(t['obs2']/t['n']),
                 a2_mean_k2=t['a2']/t['n'],two_e0a_mean_k2=2*t['e0a']/t['n']))
        for r in rows:
            pair=(r['lead'],r['target'],r['penetration'],r['kh_m2s'],r['entrainment_proxy'],not r['advection'])
            match=grouping.get(pair)
            if match:
                this=grouping[(r['lead'],r['target'],r['penetration'],r['kh_m2s'],r['entrainment_proxy'],r['advection'])]
                rmse_delta=math.sqrt(this['sse']/this['n'])-math.sqrt(match['sse']/match['n'])
                r['rmse_vs_opposite_advection_k']=rmse_delta
                r['delta_mse_adv_on_minus_off_k2']=(this['a2']-2*this['e0a'])/this['n']
                if not r['advection']:r['delta_mse_adv_on_minus_off_k2']=None
        write_csv(a.output/'summary.csv',rows)
        note={
            'status':'diagnostic-only-no-training',
            'inputs':{'budget2d':str(a.budget2d),'glorys3d':str(a.glorys3d)},
            'setup':{'history':a.history,'split':a.split,'leads':leads,'targets':a.targets,'max_windows_per_lead':a.max_windows,
                     'kh_values_m2s':khs,'kd_m-1':a.kd,'penetration':a.penetration,
                     'ssr_units':ssr_unit,'qnet_units':b['qnet'].attrs.get('units'),
                     'mixed_layer_floor_m':a.min_ml_depth,'depth_levels_m':depth.tolist(),
                     'depth_interface_method':'midpoints-approximation'},
            'warnings':[
                'Results are observed-trajectory heat-budget closure (K), not model forecast skill or learned-P0 reproduction.',
                'For target=noaa, mixed-layer GLORYS physical terms are compared to NOAA SST change only as a cross-target diagnostic; it is NOT an exact NOAA SST conservation equation.',
                'Actual GLORYS future MLD, thetao, and currents are used for retrospective diagnosis, not deployable forecast features.',
                'Entrainment is the nonnegative MLD-deepening proxy; no real vertical pumping is included.',
                'Horizontal diffusion is a constant-Kh masked finite-difference sensitivity, not diagnosed physical Kh.',
                'Penetrative radiation is a single exponential approximation; optical Kd is a sensitivity parameter.',
                'Unresolved vertical mixing R_mix is defined as the remaining residual, not added back into the RHS.',
                'Layer interfaces are approximated from depth-center midpoints. 3D preprocessing fill masks are unavailable.',
                'Evaluation masks require all lead days valid and are shared across all physical variants at each sample.',
                'Use the exact v4/v5 sample indices and learned-we settings for strict numerical comparison with earlier audits.'
            ]}
        with (a.output/'manifest.json').open('x',encoding='utf8') as f:json.dump(note,f,ensure_ascii=False,indent=2,default=str)
    print('Saved:',a.output)
    for row in rows:
        if row['lead']==leads[0] and row['target']=='glorys_ml' and row['kh_m2s']==0 and not row['entrainment_proxy'] and not row['penetration']:
            print('Quick check:',{k:row[k] for k in ('lead','target','advection','rmse_k','zero_change_rmse_k','n_window_cells')})


if __name__=='__main__':
    main()
