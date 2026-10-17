"""Training-only native residual and no-step gradient diagnostics."""
from adaptive_common import *
from torch.utils.data import DataLoader,Subset
from mhw.mhw_skill import mhw_binary_field,day_of_year
from mhw.ocean_stencil import ocean_neighbor_support

def captured(endpoint,tm,mask,batch,pa,lead,we):
 captured_values={};original=h.compute_physics_loss
 def capture(res,m,args,**kw):
  captured_values.update(res=res.detach(),rhs=kw['rhs_per_day'].detach(),mask=m.detach())
  return original(res,m,args,**kw)
 h.compute_physics_loss=capture
 try:loss=h.daily_rollout_physics_loss(endpoint,tm,endpoint,mask,batch,pa,lead,we,torch.zeros_like(we))
 finally:h.compute_physics_loss=original
 return captured_values,loss

def describe(dt,rhs,res):
 dt=dt.astype(np.float64);rhs=rhs.astype(np.float64);res=res.astype(np.float64);n=len(res)
 if not n:return dict(count=0,reason='no valid entries')
 return dict(count=n,observed_mean=float(dt.mean()),observed_std=float(dt.std()),rhs_mean=float(rhs.mean()),rhs_std=float(rhs.std()),pearson=float(np.corrcoef(dt,rhs)[0,1]) if dt.std()>0 and rhs.std()>0 else None,rhs_minus_observed_bias=float((-res).mean()),residual_rmse=float(np.sqrt((res**2).mean())),abs_residual_median=float(np.median(abs(res))),abs_residual_p90=float(np.quantile(abs(res),.9)),abs_residual_p95=float(np.quantile(abs(res),.95)),clip_fraction=float(np.mean(abs(res)>10)),units='K/day')

def residual(jobid,cpu_reuse=False):
 d=jobdir(jobid,'diagnostics')
 if cpu_reuse:
  import xarray as xr
  from types import SimpleNamespace
  device=torch.device('cpu');torch.set_num_threads(2);j=protocol()['jobs'][jobid];cfg=json.loads((Path(j['reference_fixed_run'])/'config.json').read_text());args=SimpleNamespace(**cfg['settings']);m=json.loads(Path(j['manifest']).read_text());assert sha(j['manifest'])==j['manifest_sha256'];ds=xr.open_dataset(m['data_path']);base=p.make_raw_base(ds,30,j['lead']);pa=t.build_physics_args(args);p.apply_manifest_physics(args,pa,m)
  old=next(x for x in json.loads((OLD/'protocol.json').read_text())['jobs'] if x['run_dir']==j['reference_fixed_run']);met=json.loads((Path(old['evaluation_dir'])/'metrics.json').read_text());field=met['fields']['train'];before=sha(Path(j['reference_fixed_run'])/'best.pt');assert before==met['checkpoint_sha256'] and sha(field['path'])==field['sha256']
  with np.load(field['path']) as z:cached_we=z['we_raw'].copy();cached_indices=z['target_index'].copy()
  assert np.array_equal(cached_indices,np.asarray(m['train_start_indices'])+29+j['lead']);cached_pos={int(v):i for i,v in enumerate(cached_indices)};times=ds.time.values;starts=np.asarray(m['train_start_indices'])
 else:
  device=environment_device();gpu_identity(d);j,args,bundle,model,pa,cfg=load(jobid,device)
  cp=torch.load(Path(j['reference_fixed_run'])/'best.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model_state']);model.eval();before=state_digest(model,False);del cp
  base=bundle.train.base;times=bundle.ds.time.values;starts=np.asarray(bundle.train_indices)
 sea=(ocean_neighbor_support(torch.tensor(base.ocean_mask)[None,None])[0,0]>0).numpy();assert sea.sum()==660
 assert np.all(np.diff(times).astype('timedelta64[D]').astype(int)==1)
 current=starts+29;allidx=current[:,None]+np.arange(j['lead']+1)[None,:];valid=np.isfinite(base.target_array[allidx][:,:,base.ocean_mask>0]).all((1,2))
 for field in ['qnet','mlotst','u_ml','v_ml']:
  x=base.physics_arrays[field][allidx[:,:-1]][:,:,base.ocean_mask>0];valid &= np.isfinite(x).all((1,2))
  if field=='mlotst':valid &= (x>0).all((1,2))
 valid &= (allidx[:,-1]<len(times));assert valid.any()
 # Event labels use unchanged evaluation threshold and only pre-Test training dates.
 reference=ROOT/'outputs/data_efficiency/v3_baseline_p0_20261008_r1/analysis/mhw_reference.npz'
 with np.load(reference) as z:threshold=z['threshold']
 end=int(allidx.max())+1;events=mhw_binary_field(base.target_array[:end],day_of_year(times[:end]),threshold,5)
 modes=['native_linear_zero_we_resolved_only','native_linear_checkpoint_we','observed_daily_checkpoint_we']
 store={mode:dict(dt=[],rhs=[],res=[]) for mode in modes};dates=[];mhw=[];component={k:[] for k in ['surface_flux','horizontal_advection','learned_entrainment']};rhs_sq=0.;rhs_n=0;maxerr=0.
 loader=DataLoader(p.PhysicsScaleDataset(base,starts[valid],j['lead']) if cpu_reuse else Subset(bundle.train,np.flatnonzero(valid).tolist()),batch_size=16,shuffle=False)
 cursor=0
 with torch.no_grad():
  for bi,b in enumerate(loader):
   if cpu_reuse:
    n=len(b['Tm']);now_indices=current[valid][cursor:cursor+n];cursor+=n;b['current_index']=torch.tensor(now_indices);we=torch.tensor(cached_we[[cached_pos[int(x+j['lead'])] for x in now_indices]],device=device)
    if we.ndim==3:we=we[:,None]
   else:
    out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True);we=t.nodes_to_grid_torch(out['we'],bundle.graph)
   mask=b['mask'].to(device);tm=b['Tm'].to(device);target=b['y_raw'].to(device);lead=j['lead'];bsize=len(tm)
   zero=torch.zeros_like(we);records={}
   for mode,wv in [(modes[0],zero),(modes[1],we)]:
    rec,loss=captured(target,tm,mask,b,pa,lead,wv);assert torch.isfinite(loss)
    native_rhs,nmask=h._linear_rhs_per_day(target,tm,mask,b,pa,lead,wv);torch.testing.assert_close(rec['rhs'],native_rhs,rtol=2e-5,atol=2e-5)
    records[mode]=rec
   # Component extraction via unchanged RHS, isolating its actual forcing inputs.
   flux_batch=dict(b,u_seq=torch.zeros_like(b['u_seq']),v_seq=torch.zeros_like(b['v_seq']))
   flux,_=h._linear_rhs_per_day(target,tm,mask,flux_batch,pa,lead,zero)
   adv_batch=dict(b,qnet_seq=torch.zeros_like(b['qnet_seq']))
   adv,_=h._linear_rhs_per_day(target,tm,mask,adv_batch,pa,lead,zero)
   entr=records[modes[1]]['rhs']-records[modes[0]]['rhs']
   torch.testing.assert_close(flux+adv,records[modes[0]]['rhs'],rtol=2e-5,atol=2e-5)
   for k,v in [('surface_flux',flux),('horizontal_advection',adv),('learned_entrainment',entr)]:component[k].append(v[:,0,sea].cpu().numpy())
   daily={k:[] for k in ['dt','rhs','res']};now=b['current_index'].numpy();obs=torch.tensor(base.target_array[now[:,None]+np.arange(lead+1)[None,:]],device=device)
   for step in range(lead):
    bb={**b,**{k:b[k][:,step:step+1] for k in ['qnet_seq','hm_seq','u_seq','v_seq']}}
    rec,_=captured(obs[:,step+1:step+2],obs[:,step:step+1],mask,bb,pa,1,we)
    daily['rhs'].append(rec['rhs']);daily['res'].append(rec['res'])
   records[modes[2]]={k:torch.cat(daily[k]) for k in ['rhs','res']}
   for mode,rec in records.items():
    rr=rec['res'][:,0,sea].cpu().numpy();rh=rec['rhs'][:,0,sea].cpu().numpy();dt=rr+rh
    assert np.isfinite(rr).all() and np.isfinite(rh).all()
    for k,v in [('dt',dt),('rhs',rh),('res',rr)]:store[mode][k].append(v)
   rr=records[modes[0]]['rhs'];support=records[modes[0]]['mask'];rhs_sq+=float((rr.square()*support).sum());rhs_n+=int(support.sum())
   dayidx=(now[None,:]+np.arange(1,lead+1)[:,None]).reshape(-1);dates.append(dayidx);mhw.append(events[dayidx][:,sea])
   if bi%20==0:print('residual',jobid,bi,'/',len(loader),flush=True)
 dateidx=np.concatenate(dates);months=times[dateidx].astype('datetime64[M]').astype(int)%12+1;seasons=np.array(['DJF','MAM','JJA','SON'])[(months%12)//3];event=np.concatenate(mhw);summary=[];seasonal=[];spatial=[];meta=dict(job=jobid,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],split='selected_train',checkpoint='fixed_validation_best')
 for mode,values in store.items():
  dt,rhs,res=[np.concatenate(values[k]) for k in ['dt','rhs','res']];summary.append(dict(**meta,mode=mode,group='all',**describe(dt.ravel(),rhs.ravel(),res.ravel())))
  for label in ['DJF','MAM','JJA','SON']:
   ix=seasons==label;seasonal.append(dict(**meta,mode=mode,season=label,**describe(dt[ix].ravel(),rhs[ix].ravel(),res[ix].ravel())))
  for flag in [False,True]:
   ix=event==flag;summary.append(dict(**meta,mode=mode,group='MHW' if flag else 'non_MHW',**describe(dt[ix],rhs[ix],res[ix])))
  cells=np.argwhere(sea)
  for c,(iy,ix) in enumerate(cells):spatial.append(dict(**meta,mode=mode,iy=int(iy),ix=int(ix),latitude=float(base.lat2d[iy,ix]),longitude=float(base.lon2d[iy,ix]),**describe(dt[:,c],rhs[:,c],res[:,c])))
 comps=[]
 for k,arr in component.items():
  x=np.concatenate(arr).astype(float);comps.append(dict(**meta,component=k,mean=float(x.mean()),std=float(x.std()),rms=float(np.sqrt((x*x).mean())),absmean=float(abs(x).mean()),units='K/day'))
 scale=rhs_sq/rhs_n;assert abs(scale-pa.physics_rhs_scale)<=max(1e-7,abs(scale)*1e-5),(scale,pa.physics_rhs_scale)
 assert before==(sha(Path(j['reference_fixed_run'])/'best.pt') if cpu_reuse else state_digest(model,False))
 for name,rows in [('observed_residual_summary',summary),('residual_seasonal',seasonal),('residual_spatial',spatial),('residual_components',comps)]:csvsave(d/(name+'.csv'),rows)
 save(d/'residual_status.json',dict(status='PASS',job=jobid,valid_windows=int(valid.sum()),excluded_windows=int((~valid).sum()),excluded_candidate_cells=int((~valid).sum()*j['lead']*660),valid_physics_cells=660,unsupported_ocean_cells=1,non_daily_date_gaps=0,scale_recomputed=scale,scale_manifest=pa.physics_rhs_scale,native_equation_agreement=True,parameters_unchanged=True,observation_scope='Primary linear observed endpoint; daily observations secondary. full RHS uses checkpoint-learned we, not independent observations.',mhw_reference=str(reference),mhw_context='Native duration labeling on pre-Test training observation sequence; threshold unchanged',timestamp=stamp()))
 (ds if cpu_reuse else bundle.ds).close();print('RESIDUAL PASS',jobid,flush=True)

def gradient(jobid,cpu=False):
 device=torch.device('cpu') if cpu else environment_device();torch.set_num_threads(2);d=jobdir(jobid,'diagnostics')
 if not cpu:gpu_identity(d)
 j,args,bundle,model,pa,cfg=load(jobid,device)
 g=np.random.default_rng(20261011+args.seed);indices=[]
 while len(indices)<512:indices.extend(g.permutation(len(bundle.train)).tolist())
 indices=indices[:512];csvsave(d/'gradient_batch_indices.csv',[dict(batch=i//16,position=i%16,subset_position=v,start_index=int(bundle.train_indices[v])) for i,v in enumerate(indices)])
 rows=[];shared_params=shared(model);cpdir=Path(j['reference_fixed_run'])
 for which in ['best','last']:
  path=cpdir/(which+'.pt')
  if not path.exists():continue
  cp=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(cp['model_state']);epoch=cp.get('epoch',cp['best_epoch']);del cp;before=state_digest(model,False);model.train()
  for bi,b in enumerate(DataLoader(Subset(bundle.train,indices),batch_size=16,shuffle=False)):
   model.zero_grad(set_to_none=True);out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True);vals=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,.01,model);st=gradient_stats(vals[1],vals[2],shared_params)
   assert all(v.grad is None for v in model.parameters())
   rows.append(dict(job=jobid,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],state=which,checkpoint_epoch=epoch,batch=bi,data_loss=float(vals[1].detach()),physics_loss=float(vals[2].detach()),**st,lambda_diagnostic=.01,checkpoint_warmup_factor=h.physics_lambda_factor(epoch,5),weighted_ratio=.01*st['G_phys']/(st['G_data']+1e-12),rhs_scale=pa.physics_rhs_scale,target_std=bundle.stats['target'][1],device=str(device),scope='shared predictor, preclip, training batch; no optimizer or parameter .grad writes'))
   if bi%8==0:print('gradient',jobid,which,bi,flush=True)
  assert before==state_digest(model,False)
 csvsave(d/'gradient_diagnostics.csv',rows);alignment=[]
 for which in ['best','last']:
  rs=[r for r in rows if r['state']==which];valid=[r for r in rs if r['valid_direction']]
  alignment.append(dict(job=jobid,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],state=which,batches=len(rs),valid_batches=len(valid),conflict_fraction=sum(r['cosine']<0 for r in valid)/len(valid),weighted_ratio_median=float(np.median([r['weighted_ratio'] for r in rs])),weighted_ratio_p90=float(np.quantile([r['weighted_ratio'] for r in rs],.9)),cosine_mean=float(np.mean([r['cosine'] for r in valid])),rhs_scale=pa.physics_rhs_scale,target_std=bundle.stats['target'][1]))
 csvsave(d/'gradient_alignment.csv',alignment);save(d/'gradient_status.json',dict(status='PASS',job=jobid,batches_per_state=32,unique_sample_indices=len(set(indices)),repeated_samples=512-len(set(indices)),initial_checkpoint='N/A: not serialized; existing one-batch initial logs may be referenced separately',component_gradients='N/A: component losses are not separate native objectives; no alternative objective invented',optimizer_steps=0,parameters_unchanged=True,gradient_clipping='diagnostic is preclip; native training history supplies actual clipping frequency',timestamp=stamp()));bundle.ds.close();print('GRADIENT PASS',jobid,flush=True)
if __name__=='__main__':
 import argparse,traceback
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);ap.add_argument('--phase',choices=['residual','gradient'],required=True);ap.add_argument('--cpu-reuse',action='store_true');ap.add_argument('--cpu-gradient',action='store_true');o=ap.parse_args()
 try:
  if o.phase=='residual':residual(o.job,cpu_reuse=o.cpu_reuse)
  else:gradient(o.job,cpu=o.cpu_gradient)
 except Exception:save(jobdir(o.job,'diagnostics')/(o.phase+'_failure.json'),dict(error=traceback.format_exc(),timestamp=stamp()));raise
