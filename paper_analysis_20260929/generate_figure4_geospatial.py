#!/usr/bin/env python3
"""CPU-only Figure 4: validate saved native-grid predictions before geospatial plotting.
No torch import, checkpoints, training, inference, regridding, or source-result writes.
Dependencies: numpy, pandas, matplotlib, xarray, h5netcdf.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
import argparse,ast,hashlib,json,datetime,math,types
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_PROJECT=Path('/ceph_data/yeonwoo/MHW_subset')
SEEDS=[7,42,123]
LEADS=[7,14]
MODELS=['baseline','P0_PINN']
MODEL_DIR={'baseline':'jeju_hilam','P0_PINN':'pi_jeju_hilam'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,Path):return str(x)
 return x

def write_json(p,x):p.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def centers_to_edges(c):
 c=np.asarray(c,dtype=np.float64)
 assert c.ndim==1 and len(c)>1 and np.isfinite(c).all() and np.all(np.diff(c)>0)
 edges=np.r_[c[0]-(c[1]-c[0])/2,(c[:-1]+c[1:])/2,c[-1]+(c[-1]-c[-2])/2]
 assert len(edges)==len(c)+1 and np.all(np.diff(edges)>0)
 return edges

def safe_extract(path,names):
 """Execute only named pure NumPy helpers from original source, never module imports."""
 tree=ast.parse(path.read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 module=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
 env={'np':np,'dataclass':dataclass}
 exec(compile(module,str(path),'exec'),env)
 return env

def table(rows,cols):
 def fmt(v):
  if v is None:return 'NaN / masked'
  if isinstance(v,float):return f'{v:.12g}'
  return str(v)
 return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(fmt(r.get(c)) for c in cols)+' |' for r in rows])

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--project-root',type=Path,default=DEFAULT_PROJECT)
 ap.add_argument('--analysis-root',type=Path,default=Path(__file__).resolve().parent)
 ap.add_argument('--output-dir',type=Path,help='Must be a new directory under /ceph_data/yeonwoo')
 args=ap.parse_args();project=args.project_root.resolve();analysis=args.analysis_root.resolve()
 out=(args.output_dir or analysis/'results/figure4_geospatial').resolve()
 assert out.is_relative_to(Path('/ceph_data/yeonwoo')),'Ceph-only output policy'
 assert not out.exists(),'Refusing to overwrite: choose a new --output-dir'
 out.mkdir(parents=True);(out/'cache').mkdir()
 os.environ['MPLCONFIGDIR']=str(out/'cache/matplotlib');os.environ['XDG_CACHE_HOME']=str(out/'cache');os.environ['TMPDIR']=str(out/'cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import TwoSlopeNorm
 from matplotlib.ticker import MaxNLocator
 provenance={};sources=[];logs=[]
 def log(t):logs.append(t);print(t,flush=True)
 def use(p,role):
  p=Path(p).resolve()
  if p not in provenance:
   st=p.stat();provenance[p]={'path':str(p),'role':role,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256_before':sha(p)}
  return p
 def read(p,role):return json.loads(use(p,role).read_text())
 log('CPU-only saved-result audit and geospatial plotting; no training or inference.')
 mainroot=project/'outputs/data_efficiency/allocation_batch16_5plus5'
 manifest=read(mainroot/'_queue/batch_status.json','main batch16 completed-run manifest')
 assert manifest['status']=='completed' and manifest['batch_size']==16 and manifest['total_runs']==48
 selected_jobs=[j for j in manifest['jobs'] if j['ratio']==10 and j['lead'] in LEADS and j['seed'] in SEEDS]
 assert len(selected_jobs)==12 and all(j['status']=='completed' and j['returncode']==0 for j in selected_jobs)
 source_names=['src/mhw/datasets.py','src/mhw/jeju_hilam_dataset.py','src/mhw/jeju_hilam_graph.py','src/train_jeju_hilam.py','src/mhw/models/jeju_hilam.py']
 for name in source_names:use(project/name,'original model/data/graph/prediction code; matched against run config hashes')
 graph_env=safe_extract(project/'src/mhw/jeju_hilam_graph.py',['_as_2d_lat_lon','ocean_mask_from_dataset','_build_mapping','nodes_to_grid'])
 split_env=safe_extract(project/'src/mhw/datasets.py',['SplitIndices','compute_sample_count','make_time_splits','build_lat_lon_2d'])
 legacy_code=use(analysis/'analyze_data_scarcity_paper.py','existing verified index-grid metric and plotting implementation')
 assert 'spatial[k]=np.sqrt(np.mean(err**2,axis=0))' in legacy_code.read_text()
 legacy_csv=use(analysis/'results/analysis_data/spatial_cell_rmse.csv','previous verified per-cell RMSE values serialized at 12 significant digits')
 legacy=pd.read_csv(legacy_csv)
 legacy_meta=read(analysis/'results/paper_figures/figure_metadata.json','original index-grid figure metadata including common scale')
 for lead in LEADS:
  for ext in ['pdf','png']:use(analysis/f'results/paper_figures/fig_spatial_delta_rmse_10pct_lead{lead}.{ext}','original index-grid figure; preserved')
 firstcfg=read(mainroot/'ratio_10/jeju_hilam/lead7_seed7/jeju_hilam_config.json','run configuration')
 data_path=Path(firstcfg['data_path']).resolve();assert data_path.is_relative_to(project/'data')
 use(data_path,'exact NetCDF used by original runs; coordinate, target and mask source')
 assert provenance[data_path]['sha256_before']==firstcfg['data_sha256']
 log('NetCDF SHA-256 matches original run config.')
 subset={s:read(project/f'data/subsets/train_subsets_v2_win44_seed{s}.json','controlled block subset metadata') for s in SEEDS}
 runaudit=[];gridaudit=[];differences=[];cornerrows=[];anchors=[];perseed=[];allgrids={};netcdfrefs={};cfgrefs={};gridrefs={}
 with xr.open_dataset(data_path) as ds:
  lat_name='lat' if 'lat' in ds else 'latitude';lon_name='lon' if 'lon' in ds else 'longitude'
  lat=ds[lat_name].values.astype(np.float64);lon=ds[lon_name].values.astype(np.float64)
  assert lat.shape==(29,) and lon.shape==(33,) and ds['sst_mean'].dims==('time',lat_name,lon_name)
  assert np.all(np.diff(lat)>0) and np.all(np.diff(lon)>0)
  lat_edges=centers_to_edges(lat);lon_edges=centers_to_edges(lon)
  assert np.allclose(np.diff(lat),np.diff(lat)[0]) and np.allclose(np.diff(lon),np.diff(lon)[0])
  assert np.array_equal((lat_edges[:-1]+lat_edges[1:])/2,lat)
  assert np.array_equal((lon_edges[:-1]+lon_edges[1:])/2,lon)
  lat2d,lon2d=graph_env['_as_2d_lat_lon'](ds);lat2b,lon2b=split_env['build_lat_lon_2d'](ds)
  assert np.array_equal(lat2d,lat2b) and np.array_equal(lon2d,lon2b)
  assert np.array_equal(lat2d,np.broadcast_to(lat[:,None],(29,33)))
  assert np.array_equal(lon2d,np.broadcast_to(lon[None,:],(29,33)))
  mask=graph_env['ocean_mask_from_dataset'](ds)
  assert mask.shape==(29,33) and int(mask.sum())==661 and int((~mask).sum())==296
  assert np.array_equal(mask,np.isfinite(ds.ocean_mask_static.values))
  grid_to_node,ocean_flat,node_to_grid=graph_env['_build_mapping'](mask)
  assert np.array_equal(ocean_flat,np.flatnonzero(mask.ravel(order='C')))
  assert np.array_equal(node_to_grid,np.column_stack(np.unravel_index(ocean_flat,mask.shape,order='C')))
  graph=types.SimpleNamespace(height=29,width=33,ocean_flat_indices=ocean_flat)
  # Nonconstant marker audit avoids relying on masked corner values to detect a flip.
  markers=np.arange(29*33,dtype=np.float64).reshape(29,33,order='C')
  mapped=graph_env['nodes_to_grid'](markers.ravel(order='C')[ocean_flat],graph,fill_value=np.nan)
  assert np.array_equal(mapped[mask],markers[mask]) and np.isnan(mapped[~mask]).all()
  time=ds.time.values.astype('datetime64[D]')
  assert len(time)==10957 and np.all(np.diff(time)==np.timedelta64(1,'D'))
  coord={'path':str(data_path),'dataset_sha256':firstcfg['data_sha256'],'sst_dimensions':list(ds.sst_mean.dims),
   'latitude':{'name':lat_name,'shape':list(lat.shape),'min':float(lat.min()),'max':float(lat.max()),'values':lat,'attrs':dict(ds[lat_name].attrs),'actual_order':'strictly increasing: south to north','spacing_from_array':np.diff(lat)},
   'longitude':{'name':lon_name,'shape':list(lon.shape),'min':float(lon.min()),'max':float(lon.max()),'values':lon,'attrs':dict(ds[lon_name].attrs),'actual_order':'strictly increasing: west to east','spacing_from_array':np.diff(lon)},
   'metadata_conflict':'latitude stored_direction=decreasing contradicts actual ascending numeric array. Actual array and target equality used; no flip based on this stale attribute.',
   'center_evidence':'Dimension-coordinate arrays locate SST values at native grid points; source build_lat_lon_2d and graph meshgrid attach those coordinates directly to each sample. No cell-bound variables are declared. These are rendered as pixel centers, not coordinate bounds.',
   'bounds_status':{'latitude_bounds_attribute':ds[lat_name].attrs.get('bounds'),'longitude_bounds_attribute':ds[lon_name].attrs.get('bounds')},
   'edge_method':'Internal adjacent-center midpoints; outer half of nearest actual spacing. These are rendering bounds, not claimed original physical cell vertices.',
   'latitude_edges':lat_edges,'longitude_edges':lon_edges,'edge_extent':[lon_edges[0],lon_edges[-1],lat_edges[0],lat_edges[-1]],
   'mask_source':'isfinite(ocean_mask_static) & (ocean_mask_static > 0.5)','ocean_cells':661,'excluded_cells':296}
  write_json(out/'coordinate_provenance.json',coord)
  for lead in LEADS:
   splits=split_env['make_time_splits'](len(time),30,lead);expected_idx=splits.test+30-1+lead
   dates=time[expected_idx]
   assert len(dates)==({7:1639,14:1638}[lead])
   assert str(dates[0])==({7:'2018-07-07',14:'2018-07-08'}[lead]) and str(dates[-1])=='2022-12-31'
   rawtarget=ds.sst_mean.isel(time=expected_idx).values.astype(np.float32)
   assert np.isfinite(rawtarget[:,mask]).all()
   target_grid=np.nan_to_num(rawtarget,nan=0.0,posinf=0.0,neginf=0.0)
   pd.DataFrame({'target_index':expected_idx,'target_date':dates.astype(str)}).to_csv(out/f'target_dates_lead{lead}.csv',index=False)
   rmse={};independent={}
   for model in MODELS:
    for seed in SEEDS:
     rd=mainroot/f'ratio_10/{MODEL_DIR[model]}/lead{lead}_seed{seed}'
     jobs=[j for j in selected_jobs if Path(j['run_dir']).resolve()==rd.resolve()];assert len(jobs)==1
     cfg=read(rd/'jeju_hilam_config.json','run configuration');progress=read(rd/'progress.json','completion status')
     indices=read(rd/'selected_start_indices.json','actual training indices')
     assert cfg['subset_ratio']==10 and cfg['subset_seed']==seed and cfg['training_seed']==seed and cfg['seed']==seed
     assert cfg['lead']==lead and cfg['history']==30 and cfg['batch_size']==16 and cfg['train_samples']==376 and cfg['S']==478
     assert cfg['test_samples']==len(expected_idx) and cfg['val_samples']==len(splits.val)
     assert not cfg['smoke_only'] and progress['status']=='completed'
     assert cfg['data_sha256']==firstcfg['data_sha256'] and Path(cfg['data_path']).resolve()==data_path
     assert indices==subset[seed]['10']['start_indices']
     assert cfg['split_sha256']['test']==array_sha(splits.test)
     assert cfg['split_sha256']['val']==array_sha(splits.val)
     for name in source_names:assert cfg['source_sha256'][name]==provenance[(project/name).resolve()]['sha256_before'],(rd,name,'source hash mismatch')
     if model=='baseline':assert cfg['physics_mode']=='none' and cfg['lambda_physics']==0
     else:
      assert cfg['physics_mode']=='p0_original' and cfg['physics_components']=='original'
      assert cfg['physics_time_mode']=='daily_rollout' and cfg['lambda_physics']==.1 and cfg['lambda_rres']==10
     shared={k:cfg[k] for k in ['split_sha256','full_statistics','target_mean','target_std','source_sha256','include_residual_feature','batch_size','S','max_steps']}
     if lead not in cfgrefs:cfgrefs[lead]=shared
     assert cfgrefs[lead]==shared
     gpath=use(rd/'jeju_hilam_graph.npz','saved graph node-to-native-grid mapping')
     with np.load(gpath,allow_pickle=False) as g:
      for key,reference in [('ocean_mask',mask),('ocean_flat_indices',ocean_flat),('node_to_grid',node_to_grid),('grid_to_node',grid_to_node)]:assert np.array_equal(g[key],reference),(rd,key)
      assert np.array_equal(g['level_0_node_rc'],node_to_grid)
     fpaths=list(rd.glob('*_test_fields.npz'));assert len(fpaths)==1
     fpath=use(fpaths[0],'main ratio10 saved prediction/target/mask/index (no inference)')
     mpaths=list(rd.glob('*_metrics.json'));assert len(mpaths)==1
     metric=read(mpaths[0],'saved scalar metrics used only as run identity check')
     assert metric['seed']==seed and metric['lead']==lead and metric['subset_ratio']==10 and not metric['smoke_only']
     with np.load(fpath,allow_pickle=False) as z:
      pred=z['pred_raw'];obs=z['target_raw'];pmask=z['mask'];idx=z['target_index']
     assert pred.shape==obs.shape==pmask.shape==(len(expected_idx),1,29,33)
     assert np.array_equal(idx,expected_idx)
     assert np.array_equal(pmask,np.broadcast_to(mask.astype(np.float32),(len(idx),1,29,33)))
     # Compare every stored target cell/date to NetCDF, following source nan_to_num exactly.
     assert np.array_equal(obs[:,0],target_grid),(rd,'target is not the exact native NetCDF grid')
     assert np.isfinite(pred[:,0,mask]).all() and np.isfinite(obs[:,0,mask]).all()
     refhashes={'target':array_sha(obs),'mask':array_sha(pmask),'index':array_sha(idx)}
     if lead not in gridrefs:gridrefs[lead]=refhashes
     assert refhashes==gridrefs[lead]
     # Reproduce the previous analysis arithmetic/order verbatim on ocean arrays.
     err=pred[:,0][:,mask].astype(np.float64)-obs[:,0][:,mask].astype(np.float64)
     cell_rmse=np.sqrt(np.mean(err**2,axis=0))
     assert np.isclose(np.sqrt(np.mean(err**2)),metric['test_rmse'],atol=2e-6,rtol=0)
     grid=np.full((29,33),np.nan);grid[mask]=cell_rmse;rmse[model,seed]=grid
     # Independent native full-grid time reduction (not node-reordered).
     full_error=pred[:,0].astype(np.float64)-obs[:,0].astype(np.float64)
     native_rmse=np.sqrt(np.mean(full_error**2,axis=0));native_rmse[~mask]=np.nan
     assert np.allclose(grid,native_rmse,atol=1e-13,rtol=0,equal_nan=True)
     independent[model,seed]=native_rmse
     perp=out/f'cell_rmse_{model}_lead{lead}_seed{seed}.npy';np.save(perp,grid,allow_pickle=False)
     runaudit.append({'lead_day':lead,'seed':seed,'model':model,'source_path':str(rd),'prediction_file':str(fpath),'config_file':str(rd/'jeju_hilam_config.json'),
      'graph_file':str(gpath),'metric_file':str(mpaths[0]),'manifest_job_id':jobs[0]['id'],'batch_size':16,'training_fraction':10,'train_samples':376,
      'physics_mode':cfg['physics_mode'],'lambda_physics':cfg['lambda_physics'],'lambda_rres_config':cfg['lambda_rres'],
      'runner_sha256_recorded':cfg['runner_sha256'],'shape':list(pred.shape),'target_dates_first':str(dates[0]),'target_dates_last':str(dates[-1]),'n_dates':len(idx),
      'ocean_cells':661,'all_target_grid_values_match_netcdf':True,'target_max_abs_difference_netcdf':0.0,'finite_ocean_prediction_values':int(np.isfinite(pred[:,0,mask]).sum()),
      'expected_ocean_date_values':len(idx)*661,'every_ocean_cell_n_valid_dates':len(idx),'code_hashes_match_training_config':True,
      'independent_full_grid_rmse_max_abs_difference':float(np.max(np.abs(native_rmse[mask]-grid[mask]))),'target_sha256':refhashes['target']})
     log(f'Checked {model} lead{lead} seed{seed}: exact native targets, graph mapping, 661 cells × {len(idx)} dates.')
   # Deliberately mean of per-seed RMSE, not RMSE of mean predictions or pooled seeds.
   b=np.mean(np.stack([rmse['baseline',s][mask] for s in SEEDS]),axis=0)
   p=np.mean(np.stack([rmse['P0_PINN',s][mask] for s in SEEDS]),axis=0)
   legacy_grid=np.full((29,33),np.nan);legacy_grid[mask]=p-b
   geo_grid=legacy_grid.copy()  # only coordinate mapping changes
   allgrids[lead]=geo_grid
   native=np.mean([independent['P0_PINN',s][mask] for s in SEEDS],axis=0)-np.mean([independent['baseline',s][mask] for s in SEEDS],axis=0)
   old=legacy[(legacy.training_fraction==10)&(legacy.lead_day==lead)].sort_values(['grid_row','grid_column'])
   assert len(old)==661 and not old.duplicated(['grid_row','grid_column']).any()
   rr=old.grid_row.to_numpy(int);cc=old.grid_column.to_numpy(int)
   assert np.array_equal(np.column_stack([rr,cc]),node_to_grid)
   historical=np.full((29,33),np.nan);historical[rr,cc]=old.delta_rmse
   csv_diff=np.abs(geo_grid[mask]-historical[mask])
   # CSV was written %.12g, so unrounded floating values cannot compare bitwise to text.
   assert np.max(csv_diff)<=1e-12,('historical CSV mismatch',lead,float(csv_diff.max()))
   assert np.allclose(b,old.baseline_rmse_mean,atol=6e-12,rtol=0)
   assert np.allclose(p,old.pinn_rmse_mean,atol=6e-12,rtol=0)
   assert np.array_equal(geo_grid,legacy_grid,equal_nan=True)
   assert np.isfinite(geo_grid).sum()==661 and np.isnan(geo_grid[~mask]).all()
   np.save(out/f'legacy_index_grid_delta_rmse_lead{lead}.npy',legacy_grid,allow_pickle=False)
   np.save(out/f'geospatial_delta_rmse_lead{lead}.npy',geo_grid,allow_pickle=False)
   np.save(out/f'historical_csv_delta_rmse_lead{lead}.npy',historical,allow_pickle=False)
   differences.append({'lead_day':lead,'shape':'29 x 33','finite_cells':661,'min':float(geo_grid[mask].min()),'max':float(geo_grid[mask].max()),
    'n_improved_cells':int((geo_grid[mask]<0).sum()),'mean_seed_rmse_vs_pooled_seed_rmse':'mean of seed-specific RMSE only',
    'regenerated_legacy_vs_geospatial_max_abs_difference':float(np.abs(legacy_grid[mask]-geo_grid[mask]).max()),
    'regenerated_legacy_vs_geospatial_mean_abs_difference':float(np.abs(legacy_grid[mask]-geo_grid[mask]).mean()),
    'historical_csv_vs_recomputed_max_abs_difference':float(csv_diff.max()),'historical_csv_vs_recomputed_mean_abs_difference':float(csv_diff.mean()),
    'independent_native_grid_delta_max_abs_difference':float(np.max(np.abs(native-geo_grid[mask]))),
    'csv_rounding_tolerance':1e-12,'array_sha256':array_sha(geo_grid)})
   for i,j,pos in [(0,0,'SW'),(0,32,'SE'),(28,0,'NW'),(28,32,'NE')]:
    cornerrows.append({'lead_day':lead,'corner':pos,'row':i,'column':j,'latitude':float(lat[i]),'longitude':float(lon[j]),'in_ocean_mask':bool(mask[i,j]),'delta_rmse':float(geo_grid[i,j])})
    ii,jj=node_to_grid[np.argmin((node_to_grid[:,0]-i)**2+(node_to_grid[:,1]-j)**2)]
    anchors.append({'lead_day':lead,'nearest_corner':pos,'row':int(ii),'column':int(jj),'latitude':float(lat[ii]),'longitude':float(lon[jj]),'delta_rmse':float(geo_grid[ii,jj])})
   for k,(i,j) in enumerate(node_to_grid):
    for model in MODELS:
     for seed in SEEDS:perseed.append({'lead_day':lead,'model':model,'seed':seed,'node_index':k,'row':int(i),'column':int(j),'latitude':lat[i],'longitude':lon[j],'rmse':rmse[model,seed][i,j]})
 # All numerical, source and orientation checks finish before the first figure is written.
 legacy_scale=float(np.abs(legacy.delta_rmse).max())
 required_scale=max(float(np.abs(g[np.isfinite(g)]).max()) for g in allgrids.values())
 vlim=max(legacy_scale,required_scale)
 scale_captions=[x['caption'] for x in legacy_meta if x['figure'].startswith('fig_spatial_delta_rmse_')]
 assert all(f'{legacy_scale:.6f}' in x for x in scale_captions)
 # Validate input hashes after reading and before rendering, including the full NetCDF.
 for p,r in provenance.items():
  r['sha256_after']=sha(p);r['unchanged']=r['sha256_before']==r['sha256_after'];assert r['unchanged'],p
 pd.DataFrame(provenance.values()).to_csv(out/'input_provenance.csv',index=False)
 pd.DataFrame(runaudit).to_csv(out/'run_provenance.csv',index=False)
 pd.DataFrame(differences).to_csv(out/'numeric_equivalence.csv',index=False,float_format='%.17g')
 pd.DataFrame(perseed).to_csv(out/'cellwise_rmse_by_seed.csv',index=False,float_format='%.17g')
 pd.DataFrame(cornerrows).to_csv(out/'corner_checks.csv',index=False,na_rep='NaN')
 pd.DataFrame(anchors).to_csv(out/'ocean_anchor_checks.csv',index=False,float_format='%.17g')
 write_json(out/'run_provenance.json',runaudit)
 np.savez_compressed(out/'figure4_arrays.npz',latitude=lat,longitude=lon,latitude_edges=lat_edges,longitude_edges=lon_edges,
  ocean_mask=mask,node_to_grid=node_to_grid,ocean_flat_indices=ocean_flat,delta_rmse_lead7=allgrids[7],delta_rmse_lead14=allgrids[14])
 log('All pre-render checks passed, including historical CSV rounding equivalence and source hash preservation.')
 plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
  'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6})
 cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('white')
 norm=TwoSlopeNorm(vmin=-vlim,vcenter=0,vmax=vlim)
 figsize=(3.35,6.3)
 fig=plt.figure(figsize=figsize,layout='constrained')
 layout=fig.add_gridspec(2,2,width_ratios=[1,.045],wspace=.04,hspace=.12)
 axes=[fig.add_subplot(layout[i,0]) for i in range(2)];cax=fig.add_subplot(layout[:,1]);meshes=[]
 meshchecks=[]
 for lead,ax,label in zip(LEADS,axes,['(a) 7-day lead','(b) 14-day lead']):
  g=allgrids[lead]
  mesh=ax.pcolormesh(lon_edges,lat_edges,np.ma.masked_invalid(g),shading='flat',cmap=cmap,norm=norm,rasterized=True)
  plotted=mesh.get_array().reshape(29,33)
  assert np.array_equal(np.ma.getmaskarray(plotted),~mask)
  assert np.array_equal(np.ma.getdata(plotted)[mask],g[mask])
  coords=mesh.get_coordinates()
  assert coords.shape==(30,34,2)
  assert np.array_equal(coords[:,:,0],np.broadcast_to(lon_edges[None,:],(30,34)))
  assert np.array_equal(coords[:,:,1],np.broadcast_to(lat_edges[:,None],(30,34)))
  assert np.array_equal((coords[:-1,:-1,0]+coords[:-1,1:,0])/2,np.broadcast_to(lon[None,:],(29,33)))
  assert np.array_equal((coords[:-1,:-1,1]+coords[1:,:-1,1])/2,np.broadcast_to(lat[:,None],(29,33)))
  ax.set_xlim(lon_edges[0],lon_edges[-1]);ax.set_ylim(lat_edges[0],lat_edges[-1])
  # Constant reference-latitude aspect only; coordinates stay geographic (no regridding).
  ax.set_aspect(1/np.cos(np.deg2rad(lat.mean())))
  ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)');ax.set_title(label,loc='left',pad=4)
  ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
  meshchecks.append({'lead_day':lead,'plotted_array_max_abs_difference':0.0,'masked_cells':int(np.ma.getmaskarray(plotted).sum()),'mesh_coordinate_shape':list(coords.shape),'pixel_centers_match_netcdf_exactly':True,'no_transpose_or_flip':True})
  meshes.append(mesh)
 colorbar=fig.colorbar(meshes[0],cax=cax);colorbar.set_label('Seed-mean ΔRMSE (K)')
 fig.canvas.draw()
 assert axes[0].get_position().y0>axes[1].get_position().y0
 assert axes[0].get_xlim()==axes[1].get_xlim() and axes[0].get_ylim()==axes[1].get_ylim()
 assert all(not a.xaxis_inverted() and not a.yaxis_inverted() for a in axes)
 stem=out/'fig4_spatial_delta_rmse_10pct_geospatial'
 # Fixed canvas width, no bbox_inches='tight', so PDF remains exactly 3.35-inch single-column width.
 fig.savefig(stem.with_suffix('.pdf'),dpi=600)
 fig.savefig(stem.with_suffix('.png'),dpi=600)
 plt.close(fig)
 assert 'torch' not in sys.modules
 verdict={key:'Yes' for key in ['run provenance correct','cellwise metric correct','lat/lon mapping verified','no transpose/flip','center/edge handling correct','ocean mask semantics correct','test dates correct','existing spatial values numerically preserved','final figure reproducible']}
 figure={'pdf':str(stem.with_suffix('.pdf')),'png':str(stem.with_suffix('.png')),'size_inches':figsize,'png_dpi':600,'layout':'single-column 2x1',
  'colormap':'RdBu_r','color_scale':[-vlim,vlim],'center':0,'legacy_common_scale_from_four_original_maps':legacy_scale,
  'required_scale_of_two_ratio10_panels':required_scale,'geographic_extent_edges':coord['edge_extent'],
  'aspect':'1/cos(mean native latitude); geographic degree axes; not a projected/regridded map',
  'mask':'296 excluded cells blank; all 661 ocean cells have full finite test predictions and native targets',
  'caption':'Cellwise change in SST RMSE under the 10% controlled block allocation, P0 PINN minus baseline. Each model RMSE is calculated over held-out target dates separately for each seed and then averaged over seeds 7, 42 and 123. Negative (blue) values favor P0 PINN. (a) 7-day lead; (b) 14-day lead. Both panels use identical geographic bounds and a common zero-centered color scale. Blank cells lie outside the 661-cell evaluation mask.',
  'metric':'mean_seed(sqrt(mean_date((PINN-target)^2))) - mean_seed(sqrt(mean_date((baseline-target)^2))) per cell',
  'timestamp_kst':datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'script':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__)),
  'versions':{'numpy':np.__version__,'pandas':pd.__version__,'xarray':xr.__version__,'matplotlib':matplotlib.__version__}}
 write_json(out/'figure_metadata.json',figure)
 write_json(out/'verification.json',{'verdict':verdict,'numeric_equivalence':differences,'renderer_mesh_checks':meshchecks,
  'input_files_unchanged':True,'input_count':len(provenance),'gpu_used':False,'torch_imported':False,'training':False,'inference':False,
  'full_netcdf_targets_match_every_saved_run':True,'synthetic_node_grid_roundtrip_all_ocean_cells':True,'corner_checks':cornerrows,
  'lat_attribute_conflict_disclosed':True,'array_center_vs_declared_cell_bounds':'pixel centers verified; midpoint edges are derived rendering bounds, no original bounds available'})
 # Preserve exact source evidence snippets with line numbers for later review.
 code_evidence=[('src/mhw/datasets.py',44,62),('src/mhw/datasets.py',120,131),('src/mhw/datasets.py',246,254),('src/mhw/datasets.py',270,301),
  ('src/mhw/jeju_hilam_dataset.py',87,121),('src/mhw/jeju_hilam_graph.py',99,118),('src/mhw/jeju_hilam_graph.py',134,140),
  ('src/mhw/jeju_hilam_graph.py',465,484),('src/train_jeju_hilam.py',414,459)]
 snippets=[]
 for name,a,b in code_evidence:
  lines=(project/name).read_text().splitlines();snippets+=['## '+name+f':{a}','```python']+[f'{i+1:4}: {line}' for i,line in enumerate(lines) if a<=i+1<=b]+['```','']
 (out/'source_mapping_evidence.md').write_text('\n'.join(snippets))
 command=f'/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B {Path(__file__).resolve()} --project-root {project} --analysis-root {analysis} --output-dir {out.parent}/figure4_geospatial_reproduced'
 report=['# Figure 4 geospatial provenance and verification','', 'CPU-only; saved main-experiment predictions; no training or inference.','',
 '## Source files','',f'- Main experiment: `{mainroot}/ratio_10` only. 12 runs = 2 leads × 2 models × 3 seeds. Batch16, 376 training windows, original p0_original/daily_rollout, lambda_physics=0.1, lambda_rres=10. Full and corrected/follow-up runs are excluded.',
 f'- NetCDF: `{data_path}`; full SHA-256 `{firstcfg["data_sha256"]}` matches every run config.',
 f'- Previous verified cell values: `{legacy_csv}`; original index-grid figures and numerical CSV are preserved.',
 '- Every prediction/config/graph/metric source path is in `run_provenance.csv` and `run_provenance.json`; before/after SHA-256 for all used files is in `input_provenance.csv`.',
 table(runaudit,['lead_day','seed','model','source_path']),
 '', '## Coordinate provenance','',
 f'- `{lat_name}`: shape {lat.shape}, {lat[0]:g}–{lat[-1]:g}°N; `{lon_name}`: shape {lon.shape}, {lon[0]:g}–{lon[-1]:g}°E. Both actual coordinate arrays are strictly increasing. Measured spacing: latitude {np.diff(lat)[0]:g}°, longitude {np.diff(lon)[0]:g}°; neither is hard-coded.',
 '- Important metadata inconsistency: latitude `stored_direction="decreasing"` is stale and contradicts the actual ascending numeric values. It is not used to infer orientation. Every stored target grid exactly matches the native NetCDF grid at the recorded test index.',
 '- Longitude has no units attribute in this processed file. The longitude dimension/name and source meshgrid usage establish the E-coordinate mapping; missing attrs are recorded, not silently invented.',
 '', '## Grid mapping','',
 '- row 0 = south (30°N); row 28 = north (37°N). Column 0 = west (123°E); column 32 = east (131°E). `array[i,j] ↔ (latitude[i], longitude[j])` verified.',
 '- Source SST dimensions are `(time, latitude, longitude)`. Base target_array uses raw values in that dimension order; target_idx selects time only, followed by nan_to_num on excluded missing values.',
 '- Graph mapping is C-order: flat index = i*33+j. `flatnonzero(mask.reshape(-1))` defines ocean nodes; `unravel_index` creates node_to_grid. All 12 saved graphs match the original helper output exactly.',
 '- Dataset permute(2,0,1) changes ocean-node/time/channel axes after spatial flattening; it is not a latitude/longitude transpose. collect_predictions scatters nodes back using the same ocean_flat_indices then reshapes to [N,1,29,33]. No spatial transpose or flip.',
 '- Original `load_time_grid_array` can move time first but preserves spatial-dimension order; target_array uses ds.sst_mean directly. Pure NumPy mapping helpers were extracted from hash-matched source; torch modules and model inference were not imported or run.',
 '- Source locations and exact snippets: `source_mapping_evidence.md`. All-cell marker roundtrip verifies ordering even where geographic corners are masked.',
 table(clean(cornerrows),['lead_day','corner','row','column','latitude','longitude','in_ocean_mask','delta_rmse']),
 '- Masked corner NaNs are not evidence of coordinate failure; nearest valid ocean anchors are provided in `ocean_anchor_checks.csv`.',
 '', '## Cell center / edge handling','',
 '- Coordinate arrays locate native SST samples and are treated as pixel centers. No latitude/longitude bounds variable is declared. Derived edges are visualization bounds, not a claim about archived physical grid-cell vertices.',
 '- pcolormesh(shading="flat") uses adjacent-center midpoint edges and half-spacing extrapolation at each outer edge. The code computes spacing from arrays; no assumed 0.25° value or min/max-as-edge extent.',
 f'- Longitude edges: {lon_edges[0]:g} to {lon_edges[-1]:g}°E; latitude edges: {lat_edges[0]:g} to {lat_edges[-1]:g}°N. The actual QuadMesh vertices and center positions are asserted against these coordinate arrays.',
 '', '## Spatial metric','',
 'For every ocean cell: first sqrt(mean over held-out dates of squared prediction error) for each model and each seed; then arithmetic mean of seed RMSEs; finally P0 PINN minus baseline. Float64 error reductions, original float32 predictions/targets. Not ensemble-mean predictions, pooled-seed RMSE, global scalar RMSE, MSE or MAE difference.',
 '- Mask semantics: exactly 661 valid ocean cells, 296 excluded cells. Every valid cell has all test dates finite in saved predictions, saved targets and native SST targets. Blank cells are mask exclusions, not missing predictions/targets or insufficient valid dates.',
 '- 7-day: 2018-07-07–2022-12-31, 1639 dates. 14-day: 2018-07-08–2022-12-31, 1638 dates. Exact daily indices verified from the original split helper and native NetCDF time; baseline/P0 and all seeds match within each lead.',
 '', '## Numeric equivalence','',table(differences,['lead_day','shape','finite_cells','min','max','regenerated_legacy_vs_geospatial_max_abs_difference','regenerated_legacy_vs_geospatial_mean_abs_difference','historical_csv_vs_recomputed_max_abs_difference','historical_csv_vs_recomputed_mean_abs_difference','independent_native_grid_delta_max_abs_difference']),
 '- The earlier script did not archive an unrounded NPY grid. It saved a 12-significant-digit CSV. The old index-grid arithmetic was regenerated from the same predictions and passed unchanged to the geospatial renderer; its difference is exactly zero. Historical CSV differences are separately reported and must be ≤1e-12 to account only for text rounding. An independent full-native-grid time reduction is also checked (not the same-array identity check).',
 '- The exact arrays actually inside the Matplotlib QuadMesh are checked for zero value difference and identical NaN mask. New full-precision .npy/.npz arrays avoid future CSV rounding ambiguity. Plot generation is aborted before rendering on any source, mapping or numerical mismatch.',
 '', '## Figure output','',f'- [PDF]({stem.name}.pdf) and [600-dpi PNG]({stem.name}.png). Fixed canvas {figsize[0]} × {figsize[1]} inches; single-column 2×1; (a) 7-day above (b) 14-day.',
 f'- Colormap RdBu_r; common symmetric limits [{-vlim:.15g}, {vlim:.15g}] K, zero at white. Existing four-map common limit reconstructed from verified spatial CSV: {legacy_scale:.15g} K; target two-panel maximum magnitude: {required_scale:.15g} K.',
 '- Common geographic edge extent in both panels; longitude increases left→right and latitude bottom→top. Aspect uses 1/cos(mean latitude) as a fixed visual correction, with no coordinate transformation or regridding. No geographic basemap/coastline inference was added.',
 '- Colorbar: Seed-mean ΔRMSE (K). No grid-coordinate/unavailable-coordinate titles. No implication of statistical significance from color.',
 '', '## Reproducibility','', '```bash',command,'```','',
 '- Use a new output directory; existing results are never overwritten. The script reads original source/predictions plus the earlier spatial CSV and real NetCDF. All computation is CPU-only. Checkpoints are not loaded.',
 '- `figure4_arrays.npz`: actual coordinates, edges, mask, graph indexing, plotted ΔRMSE arrays. `cell_rmse_*`: 12 seed/model-specific cell grids. `cellwise_rmse_by_seed.csv`: full-precision cell-level calculation records. `figure_metadata.json`, `verification.json`, `numeric_equivalence.csv` record the audit.',
 '', '## Final verdict','',table([{'Check':k,'Yes/No':v} for k,v in verdict.items()],['Check','Yes/No']),'']
 (out/'provenance_report.md').write_text('\n'.join(report),encoding='utf-8')
 (out/'execution.log').write_text('\n'.join(logs)+'\n')
 write_json(out/'output_checksums.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and 'cache' not in p.relative_to(out).parts})
 log('Figure 4 complete: '+str(stem.with_suffix('.pdf')))
 print(json.dumps(clean(differences),indent=2),flush=True)

if __name__=='__main__':main()
