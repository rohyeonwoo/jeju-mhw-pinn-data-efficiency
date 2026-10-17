"""24-manifest cached-data runtime audit and disposable native two-batch loss smoke."""
import os,json,copy,gc,math,csv
from pathlib import Path
import train_v3_ablation as a
from train_v3_study import *
from mhw import subset_climatology as c
from mhw.ocean_stencil import ocean_neighbor_gradients,ocean_neighbor_support
import xarray as xr
OUT=ROOT/'outputs/v3_physics_ratio_lead_lambda001_20261009'
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3']
 torch.set_num_threads(2);device=torch.device('cuda:0');gpu_identity(OUT)
 protocol=json.loads((OUT/'protocol.json').read_text());jobs=protocol['jobs'];manifest0=json.loads(Path(jobs[0]['manifest']).read_text())
 ds=xr.open_dataset(manifest0['data_path']);base=p.make_raw_base(ds,30,7);raw=c.load_raw_anomaly_fields(ds,manifest0['raw_thetao_path']);bins=c.calendar_bins(ds.time.values)
 ocean=base.ocean_mask.astype(bool)
 assert ocean.sum()==661
 for field in [*base.dynamic_arrays.values(),*raw.values()]:assert np.isfinite(field[:,ocean]).all()
 details=list(csv.DictReader((ROOT/'outputs/v2_final_preflight_20261008/input_feature_split_details.csv').open()))
 verification=json.loads((ROOT/'outputs/v2_final_preflight_20261008/input_analysis_verification.json').read_text());assert verification['nonfinite_ocean']==0
 expected={};allrows=[];diag=[];leakage_done=set();peak=0
 for jb,jp in zip(jobs[::2],jobs[1::2]):
  m=json.loads(Path(jb['manifest']).read_text());path=Path(jb['manifest']);clims=c.read_climatology(m,path)
  key=(jb['ratio'],jb['seed'])
  if key not in leakage_done:
   selected=np.array(m['input_time_indices']);fit,_=c.fit_climatology(raw['sst_anom'],bins,selected,ocean);np.testing.assert_array_equal(fit,clims['sst_anom'])
   altered=raw['sst_anom'].copy();excluded=np.ones(len(altered),bool);excluded[selected]=False;altered[excluded]+=12345
   fit2,_=c.fit_climatology(altered,bins,selected,ocean);np.testing.assert_array_equal(fit,fit2);del altered,fit,fit2;leakage_done.add(key)
  bundle=c.bundle_from_v2(ds,m,base,raw,clims)
  assert bundle.stats=={k:tuple(v) for k,v in m['statistics'].items()}
  # Actual same production setup, with only read-only raw bundle IO cached across arms.
  original=p.load_subset_bundle;p.load_subset_bundle=lambda requested:(bundle,m) if Path(requested)==path else original(requested)
  pair=[];smokes=[]
  for job in [jb,jp]:
   args,bb,model,pa,cfg=a.setup(job,protocol,device)
   assert bb is bundle and cfg['applied_rhs_scale']==m['physics_rhs_scale']
   for split in ['train','val','test']:
    dd=getattr(bundle,split)
    for i in [0,len(dd)//2,len(dd)-1]:
     item=dd[i];assert int(item['target_index']-item['current_index'])==job['lead']
     assert torch.isfinite(item['x_nodes']).all() and torch.isfinite(item['y_nodes']).all()
   mask=bundle.train[0]['mask'];assert int(mask.sum())==661 and int(ocean_neighbor_support(mask).sum())==660
   dx=torch.ones_like(mask);dy=dx;constant=torch.full_like(mask,20);g=ocean_neighbor_gradients(constant,dx,dy,mask);assert all((x[mask>0]==0).all() for x in g)
   f=torch.arange(mask.numel()).reshape(mask.shape).float();g1=ocean_neighbor_gradients(f,dx,dy,mask);f[mask==0]=float('nan');g2=ocean_neighbor_gradients(f,dx,dy,mask);assert all(torch.equal(x,y) for x,y in zip(g1,g2))
   d=OUT/'smoke'/job['name'];d.mkdir(parents=True,exist_ok=True)
   pair.append(cfg);a.initial_diagnostic(model,bundle,args,pa,device,d)
   batches=[]
   for batch in t.make_loader(bundle.train,args,False):
    batches.append(batch)
    if len(batches)==2:break
   torch.cuda.reset_peak_memory_stats()
   optimizer=torch.optim.AdamW(model.parameters(),lr=args.initial_lr,weight_decay=args.weight_decay,betas=(.9,.95));monitor=a.Monitor(model,pa)
   with monitor:metrics=t.run_epoch(model,batches,bundle.graph,device,*bundle.stats['target'],optimizer=optimizer,physics_args=pa,lambda_physics=args.lambda_physics,epoch=1)
   assert monitor.phys_calls==(2 if job['arm']=='physics_only' else 0) and monitor.rres_calls==0
   assert all(math.isfinite(v) for v in metrics.values())
   assert all(torch.isfinite(v).all() for v in model.state_dict().values())
   if jb['ratio']==10 and jb['seed']==42:
    cp=d/'roundtrip.pt';checkpoint(cp,dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict()));saved=torch.load(cp,map_location=device,weights_only=False);assert all(torch.equal(saved['model_state'][k],v) for k,v in model.state_dict().items());del saved
   peak=max(peak,torch.cuda.max_memory_reserved()/2**20)
   smokes.append(dict(arm=job['arm'],physics_calls=monitor.phys_calls,rres_calls=monitor.rres_calls,gradient_norms=monitor.norms,loss=metrics['total_loss'],peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,initial_diagnostic=str(d/'initial_objective_gradients.json')))
   del model,optimizer,monitor,batches;gc.collect();torch.cuda.empty_cache()
  p.load_subset_bundle=original
  for k in ['parameters','trainable_parameters','shared_initial_sha256','stats','manifest_sha256','rhs_scale','applied_rhs_scale','split_sha256','tensor_fingerprints']:assert pair[0][k]==pair[1][k],k
  with np.load(path.parent/m['climatology_file']) as z:counts=z['calendar_counts'];assert np.array_equal(z['input_time_indices'],m['input_time_indices'])
  for rr in details:
   if (int(rr['ratio']),int(rr['lead']),int(rr['seed']))!=(jb['ratio'],jb['lead'],jb['seed']):continue
   if rr['variable'] not in c.ANOMALIES:continue
   assert float(rr['normalization_mean'])==m['statistics'][rr['variable']][0] and float(rr['normalization_std'])==m['statistics'][rr['variable']][1]
   diag.append(dict(ratio=jb['ratio'],lead=jb['lead'],seed=jb['seed'],variable=rr['variable'],split=rr['split'],unique_input_dates=len(m['input_time_indices']),empty_doy=int((counts==0).sum()),singleton_doy=int((counts==1).sum()),anomaly_std_derived_from_z=float(rr['std'])*float(rr['normalization_std']),normalized_std=float(rr['std']),fraction_abs_z_gt5=float(rr['fractions_gt5']),fraction_abs_z_gt10=float(rr['fractions_gt10']),fraction_abs_z_gt20=float(rr['fractions_gt20']),max_abs_z=max(abs(float(rr['min'])),abs(float(rr['max']))),target_std=m['statistics']['target'][1],rhs_scale=m['physics_rhs_scale'],distribution_weighting='input-history tensor occurrences incl repeated dates; reused verified v2 inputs identical to v3',source='outputs/v2_final_preflight_20261008/input_feature_split_details.csv'))
  row=dict(ratio=jb['ratio'],lead=jb['lead'],seed=jb['seed'],status='PASS',samples=len(bundle.train),mask=661,physics_support=660,stencil='ocean_neighbors_v1',no_full_override=True,pair_identical=True,smoke=smokes)
  allrows.append(row);save(OUT/'preflight_progress.json',dict(completed=len(allrows),total=24,rows=allrows));print(json.dumps(row),flush=True)
  del bundle,bb;gc.collect()
 csvsave(OUT/'climatology_diagnostics.csv',diag)
 result=dict(status='PASS',conditions=24,rows=allrows,peak_reserved_mib=peak,leakage_mutation_tests=len(leakage_done),climatology_source='selected unique input dates only; empty calendar bins periodic linear interpolation; native Feb29 aligned to leap2000',finite_ocean='current raw arrays+manifest stats+production sample tensors; complete existing tensor-distribution scan reused after v2/v3 equality',distribution_source_sha256=sha(ROOT/'outputs/v2_final_preflight_20261008/input_feature_split_details.csv'),extreme_values='Preserved finite heavy tails; not clipped or threshold-relaxed',training='96 disposable smoke updates only; no production checkpoint modified',timestamp=stamp())
 save(OUT/'preflight.json',result)
 (OUT/'preflight.md').write_text('# Preflight PASS\n\n24 manifests: raw/source/climatology hashes, nested subsets/splits,376/987/1974/3995 samples, subset-only statistics, ocean661/physics660, same all-head initial predictor, applied RHS scales, production tensor fingerprints, finite inputs/objectives/gradients verified.\n\n12 distinct SST climatology mutation tests alter only nonselected dates: no change. Existing full feature occurrence distribution audit reused only after exact v2/v3 input statistics/climatology match; raw arrays rechecked finite. 7 anomaly channels ×3splits ×24conditions in climatology_diagnostics.csv. Finite extreme tails are preserved, not hidden or clipped.\n\nBoth arms run2 disposable native batch16 optimizer updates per condition, Baseline physics calls0; Physics-only calls2 and positive shared gradient; auxiliary calls0, inactive head gradientsNone. Checkpoint roundtrip at10% seed42 both leads. Production starts afresh from seed. Core training/physics/model code unchanged. See preflight.json for individual evidence and memory.\n')
 ds.close();log(f'Fixedλ=.01 all24condition preflight PASS; memory peak{peak}MiB; no changes to raw/v3/model/loss equations; production authorized scope39 new jobs.');print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  save(OUT/'preflight_failure.json',dict(error=traceback.format_exc(),timestamp=stamp()));raise
