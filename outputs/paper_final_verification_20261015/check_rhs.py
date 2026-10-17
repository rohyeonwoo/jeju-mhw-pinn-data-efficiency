import os,sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';sys.path.insert(0,str(R/'src'));sys.dont_write_bytecode=True
import numpy as np,torch,xarray as xr,pandas as pd
from mhw.datasets import compute_grid_spacing_m
from mhw.subset_preprocessing import compute_subset_rhs_scale,p0_scale_args
import train_hybrid_jeju_forecast as h
torch.set_num_threads(1)
B=R/'data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc';d=xr.open_dataset(B);lat,lon=np.meshgrid(d.latitude.values,d.longitude.values,indexing='ij');dx,dy=compute_grid_spacing_m(lat,lon);ocean=np.isfinite(d.ocean_mask_static.values)
b=SimpleNamespace(history=30,target_array=d.sst_mean.values.astype('float32'),ocean_mask=ocean.astype('float32'),dx=dx,dy=dy,physics_arrays={k:d[k].values.astype('float32') for k in ['qnet','mlotst','u_ml','v_ml']})
rows=[]
for path in sorted((R/'data/subsets/preprocessing_subset_v3').glob('ratio_*.json')):
 m=json.loads(path.read_text());v=compute_subset_rhs_scale(b,m['train_start_indices'],m['lead']);old=m['physics_rhs_scale'];rows.append(dict(ratio=m['subset_ratio'],seed=m['subset_seed'],lead=m['lead'],stored=old,recomputed=v,abs_error=abs(v-old),status='PASS' if np.isclose(v,old,rtol=2e-6,atol=1e-9) else 'FAIL'));print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(O/'rhs_scale_recomputed.csv',index=False)
# Compare P0 runtime residual with _linear_rhs_per_day at identical settings/mask.
from mhw.subset_preprocessing import PhysicsScaleDataset
batch=next(iter(torch.utils.data.DataLoader(PhysicsScaleDataset(b,[0,90],7),batch_size=2)))
a=p0_scale_args();a.zero_rres_physics=True;a.physics_rhs_scale=rows[0]['stored'];a.physics_clip=10.;a.physics_normalize='rhs';a.physics_loss='mse';a.adaptive_sampling=False;a.adaptive_weighting=False
pred=(batch['Tm']+1).requires_grad_();we=torch.full_like(pred,1e-5);rhs,mask=h._linear_rhs_per_day(pred,batch['Tm'],batch['mask'],batch,a,7,we);capt={};original=h.compute_physics_loss
h.compute_physics_loss=lambda residual,mask,args,**kw:capt.update(residual=residual,mask=mask,**kw)
h.daily_rollout_physics_loss(pred,batch['Tm'],batch['y_raw'],batch['mask'],batch,a,7,we,torch.zeros_like(pred));h.compute_physics_loss=original
want=((pred-batch['Tm'])/7).repeat(7,1,1,1)-rhs;diff=(capt['residual']-want)[mask>0].abs().max().item();rhsdiff=(capt['rhs_per_day']-rhs)[mask>0].abs().max().item();loss=original(capt['residual'],capt['mask'],a,rhs_per_day=rhs);g=torch.autograd.grad(loss,pred)[0]
# Flux components, metadata and alignment; no original files changed.
q=b.physics_arrays['qnet'];components=sum(d[k].values.astype('float64') for k in ['ssr','str','slhf','sshf']);diffq=q[:,ocean].astype('float64')-components[:,ocean]
G=Path('/ceph_data/yeonwoo/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d.nc')
with xr.open_dataset(G) as g3:
 alignment={dim:np.array_equal(d[dim].values,g3[dim].values) for dim in ['time','latitude','longitude']};velocity={k:dict(dims=g3[k].dims,attrs=g3[k].attrs) for k in ['uo','vo']}
result=dict(p0_rhs_runtime_max_error_kday=rhsdiff,p0_residual_runtime_max_error_kday=diff,tolerance_kday=2e-6,prediction_gradient_norm=float(g.norm()),prediction_gradient_finite=bool(torch.isfinite(g).all()),runtime_consistency='PASS' if max(rhsdiff,diff)<2e-6 else 'FAIL',qnet_component_max_abs_difference_wm2=float(abs(diffq).max()),qnet_component_rmse_wm2=float(np.sqrt(np.mean(diffq**2))),alignment=alignment,velocity=velocity)
(O/'physics_numerical_checks.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
