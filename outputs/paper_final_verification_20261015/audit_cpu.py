import os,sys,json,hashlib,math,importlib.util
from pathlib import Path
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';sys.path.insert(0,str(R/'src'));sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd,torch,xarray as xr
torch.set_num_threads(1)
from mhw.ocean_stencil import ocean_neighbor_gradients as grad,ocean_neighbor_support as support
from mhw.datasets import compute_grid_spacing_m
J=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def save(name,x): (O/name).write_text(json.dumps(x,indent=2,default=str))
rows=[]
def check(name,actual,expected,atol,rtol=0,notes=''):
 a=np.asarray(actual);e=np.broadcast_to(np.asarray(expected),a.shape);err=np.abs(a-e);rel=err/np.maximum(abs(e),1e-30)
 rows.append(dict(test=name,expected_min=float(e.min()),expected_max=float(e.max()),calculated_min=float(a.min()),calculated_max=float(a.max()),max_abs_error=float(err.max()),max_rel_error=float(rel.max()),atol=atol,rtol=rtol,status='PASS' if np.isfinite(a).all() and np.allclose(a,e,atol=atol,rtol=rtol) else 'FAIL',notes=notes))
for dtype in [torch.float32,torch.float64]:
 tag=str(dtype);eps=torch.finfo(dtype).eps
 y,x=torch.meshgrid(torch.arange(7,dtype=dtype)*27000,torch.arange(9,dtype=dtype)*23000,indexing='ij');m=torch.ones_like(x);m[2:5,4]=0;m[0,0]=0
 dx=torch.full_like(x,23000);dy=torch.full_like(y,27000);a=2e-5;b=-1e-5;field=20+a*x+b*y;valid=support(m)>0
 tol=86400*eps*32*float(abs(field).max())/23000
 for label,f in [('constant',20*torch.ones_like(x)),('linear',field)]:
  gx,gy=grad(f,dx,dy,m);v=-86400*(.2*gx-.1*gy);expected=0 if label=='constant' else -86400*(.2*a-.1*b)
  check(tag+'_'+label,v[valid],expected,0 if label=='constant' else tol)
 gx,gy=grad(field,dx,dy,m);ad=-86400*(.2*gx-.1*gy)
 check(tag+'_velocity_sign',(-86400*(-.2*gx+.1*gy))[valid],-ad[valid],0)
 for fill in [0.,-100.,1000.,float('nan'),float('inf')]:
  land=torch.where(m>0,field,torch.full_like(field,fill));g=grad(land,dx,dy,m)
  check(tag+'_land_'+str(fill),torch.stack(g)[:,m>0],torch.stack([gx,gy])[:,m>0],0)
 # All 16 axis-neighbor configurations, including isolated center.
 for bits in range(16):
  mm=torch.zeros((3,3),dtype=dtype);mm[1,1]=1
  for k,ij in enumerate([(1,0),(1,2),(0,1),(2,1)]):mm[ij]=bool(bits&(1<<k))
  xx=torch.arange(3,dtype=dtype)[None,:]*23000;yy=torch.arange(3,dtype=dtype)[:,None]*27000
  g=grad(20+a*xx+b*yy,torch.full_like(mm,23000),torch.full_like(mm,27000),mm)
  check(tag+f'_neighbor{bits}',[g[0][1,1],g[1][1,1]],[a if bits&3 else 0,b if bits&12 else 0],tol/86400)
  check(tag+f'_support{bits}',support(mm)[1,1],float(bool(bits&3) and bool(bits&12)),0)
# Geographic spacing and descending arrays; field linear in degrees analytic geographic derivatives.
for descending in [False,True]:
 lat=np.arange(30,37.25,.25);lon=np.arange(122,130.25,.25)
 if descending:lat=lat[::-1];lon=lon[::-1]
 ll,oo=np.meshgrid(lat,lon,indexing='ij');dx,dy=compute_grid_spacing_m(ll,oo)
 f=.4*oo+.2*ll;gx,gy=grad(torch.tensor(f),torch.tensor(dx),torch.tensor(dy),torch.ones_like(torch.tensor(f)))
 check('geographic_dx_desc'+str(descending),gx.numpy(),.4/(6371000*np.cos(np.deg2rad(ll))*np.pi/180),1e-12,1e-7)
 check('geographic_dy_desc'+str(descending),gy.numpy(),.2/(6371000*np.pi/180),1e-12,1e-7)
check('coslat_distance_ratio',float(dx[-1,0]/dx[0,0]),float(np.cos(np.deg2rad(lat[-1]))/np.cos(np.deg2rad(lat[0]))),2e-7)
# Nonuniform interior: spacing is mean of both gaps. One-sided uses that mean too.
lon=np.array([120,120.25,120.75]);lat=np.array([30,30.25,30.5]);ll,oo=np.meshgrid(lat,lon,indexing='ij');dx,dy=compute_grid_spacing_m(ll,oo);f=torch.tensor(oo.copy());m=torch.ones_like(f);m[1,0]=0
g=grad(f,torch.tensor(dx),torch.tensor(dy),m)
expected=.5/dx[1,1];check('nonuniform_one_sided_documented_rule',g[0][1,1],expected,1e-12,1e-7,notes='Expected implementation uses averaged coordinate spacing .375deg; true derivative 1/(R coslat pi/180) differs by +33.333%. Not a general irregular-grid operator; production grid regular .25deg.')
# Archived coastal algorithm equivalence.
archive=Path('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py')
sys.path.append(str(archive.parent));spec=importlib.util.spec_from_file_location('archived_stencil',archive);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
print('archive functions', [n for n in vars(mod) if 'gradient' in n or 'neighbor' in n],flush=True)
# Metadata and actual ocean mask.
B=R/'data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d_budget2d.nc';ds=xr.open_dataset(B)
meta=dict(path=str(B),coords={n:dict(first=float(ds[n][0]),last=float(ds[n][-1]),steps=np.unique(np.diff(ds[n].values)).tolist(),attrs=ds[n].attrs) for n in ['latitude','longitude']},variables={n:dict(dims=ds[n].dims,attrs=ds[n].attrs) for n in ['sst_mean','qnet','ssr','u_ml','v_ml','uo','vo','mlotst'] if n in ds},attrs=ds.attrs)
sea=torch.tensor(np.isfinite(ds.ocean_mask_static.values),dtype=torch.float32);la,lo=np.meshgrid(ds.latitude.values,ds.longitude.values,indexing='ij');dx,dy=map(torch.tensor,compute_grid_spacing_m(la,lo));f=20+torch.randn(sea.shape,generator=torch.Generator().manual_seed(42));gg=grad(f,dx,dy,sea)
fn=getattr(mod,'ocean_neighbor_gradients',None)
if fn:
 old=fn(f,dx,dy,sea);check('archived_operator_equivalence',torch.stack(gg).numpy(),torch.stack(old).numpy(),0)
check('actual_mask_constant',torch.stack(grad(torch.full_like(sea,20),dx,dy,sea))[:,sea>0],0,0)
meta.update(ocean_cells=int(sea.sum()),support_cells=int(support(sea).sum()),unsupported_ocean_indices=torch.nonzero((sea>0)&(support(sea)==0)).tolist())
save('dataset_metadata.json',meta);pd.DataFrame(rows).to_csv(O/'stencil_tests.csv',index=False);save('stencil_tests.json',rows)
# Protocol/log/checkpoint hashes, all 120.
T=R/'outputs/v8_actual_epoch_oracle96_20261011';A=R/'outputs/v8_actual_epoch_auxiliary24_20261015';p=J(T/'experiment_protocol.json');q=J(A/'training_protocol.json');out=[];source=[]
for family,pr in [('Oracle96',p),('Aux24',q)]:
 for f,h in pr['code_sha256'].items():source.append(dict(family=family,path=f,expected=h,actual=sha(R/f),match=sha(R/f)==h))
refs={};configs={}
for family,pr,root in [('Oracle96',p,T),('Aux24',q,A)]:
 for j in pr['jobs']:
  d=Path(j['run_dir']);c=J(d/'config.json');h=J(d/'history.json');s=J(d/'status.json');m=J(j['manifest']);best=min(h,key=lambda e:e['val_rmse']);ca=J(d/'checkpoint_audit.json');ca=ca.get('best',ca)
  label=j.get('label','Auxiliary');key=(j['ratio'],j['lead'],j['seed']);errors=[]
  def req(ok,name):
   if not ok:errors.append(name)
  req(s['status']=='COMPLETED','completed');req(len(h)==80 and [e['epoch'] for e in h]==list(range(1,81)),'80epochs');req(c['settings']['batch_size']==16,'batch16');req(c['early_stopping_enabled']==False,'noearlystop')
  req(all(e['sampling_exact_once'] and e['train_samples']==len(m['train_start_indices']) for e in h),'sample_once');req(ca['epoch']==best['epoch'] and abs(ca['expected_val_rmse']-best['val_rmse'])<1e-12,'bestminimum');req(abs(ca['reloaded_val_rmse']-best['val_rmse'])<2e-6,'bestreload')
  cp=root/'checkpoints'/j['name']/'best.pt';req(sha(cp)==ca['sha256'],'checkpointSHA');req(sha(j['manifest'])==c['manifest_sha256'],'manifestSHA');req(m['schema']=='subset_normalization_v3','v3');req(c['stats']==m['statistics'],'statistics');req(c['rhs_scale']==m['physics_rhs_scale'] and c['rhs_scale']>0 and math.isfinite(c['rhs_scale']),'RHSscale')
  if label=='Baseline':refs[key]=(c,h,m)
  b,bh,bm=refs[key]
  for k in ['stats','rhs_scale','split_sha256','shared_initial_sha256','tensor_fingerprints','parameters']:req(c[k]==b[k],'paired_'+k)
  req([e['sample_order_sha256'] for e in h]==[e['sample_order_sha256'] for e in bh],'pairedshuffle');req([e['lr'] for e in h]==[e['lr'] for e in bh],'pairedLR')
  req(all(e['rres_calls']==(e['epoch_updates'] if label=='Auxiliary' else 0) for e in h),'auxbranch');req(all(e['physics_calls']==(e['epoch_updates'] if label in ['P0','NoAdv','Adv'] else 0) for e in h),'physicsbranch')
  req(max(m['target_time_indices'])<min(np.array(m['val_start_indices'])+29+j['lead']),'target_train_boundary');req(max(np.array(m['train_start_indices'])+29+j['lead']-1)<min(np.array(m['val_start_indices'])+29+j['lead']),'forcing_train_boundary')
  configs[j['name']]=c
  out.append(dict(name=j['name'],model=label,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],n_train=len(m['train_start_indices']),epochs=len(h),updates=h[-1]['steps'],best_epoch=best['epoch'],val_rmse=best['val_rmse'],rhs_scale=c['rhs_scale'],checkpoint_sha256=ca['sha256'],status='PASS' if not errors else 'FAIL',errors=';'.join(errors)))
  print(j['name'],out[-1]['status'],out[-1]['errors'],flush=True)
pd.DataFrame(out).to_csv(O/'protocol120.csv',index=False);save('source_hashes.json',source)
# v2 input/target/climatology preservation and v3 nested sets.
checks=[]
for key,(c,h,m) in refs.items():
 old=J(m['parent_manifest']);checks.append(dict(condition=key,stats_same=m['statistics']==old['statistics'],clim_same=m['climatology_sha256']==old['climatology_sha256'],indices_same=all(m[k]==old[k] for k in ['train_start_indices','val_start_indices','test_start_indices','input_time_indices','target_time_indices']),clim_hash_valid=sha(R/'data/subsets/preprocessing_subset_v3'/m['climatology_file'])==m['climatology_sha256']))
for seed in [42,7,123]:
 for lead in [7,14]:
  sets=[set(refs[r,lead,seed][2]['train_start_indices']) for r in [10,25,50,100]];assert all(a<=b for a,b in zip(sets,sets[1:]))
save('manifest_checks.json',checks)
# Auxiliary diagnostics, original reports recalculated.
diag=[]
for j in q['jobs']:
 for row in J(Path(j['run_dir'])/'diagnostics.json'):diag.append(dict(name=j['name'],**row))
df=pd.DataFrame(diag);df.to_csv(O/'aux_diagnostics_rechecked.csv',index=False)
save('audit_status.json',dict(protocol_pass=sum(x['status']=='PASS' for x in out),total=120,source_hashes=len(source),source_hash_mismatch=[x for x in source if not x['match']],stencil_pass=sum(x['status']=='PASS' for x in rows),stencil_total=len(rows),manifest_checks=checks,aux_diag_count=len(df),aux_diagnostics_preserved=bool(df[['model_state_unchanged','parameter_gradients_unchanged','RNG_unchanged','modes_unchanged']].all().all()),aux_median_gradient_ratio={e:float(df[df.epoch==e].weighted_gradient_ratio.median()) for e in [1,5,10,20,80]}))
print('CPU audit done',flush=True)
