"""Fixed global checkpoint lock inference; reuse exact previous fields when proven."""
import os,argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import train_v3_study as w
np=w.np;torch=w.torch;t=w.t;p=w.p

def metrics(f):
 valid=f['mask'].astype(bool);e=f['pred_raw'].astype(float)-f['target_raw'].astype(float);assert np.isfinite(e[valid]).all()
 count=valid.reshape(len(e),-1).sum(1);sse=np.where(valid,e**2,0).reshape(len(e),-1).sum(1)
 f['daily_sse']=sse;f['daily_count']=count;f['daily_mse']=sse/count
 return dict(rmse=float(np.sqrt(sse.sum()/count.sum())),mae=float(np.abs(e[valid]).mean()),bias=float(e[valid].mean()))
def main(study,jobid):
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3']
 prot=json.loads((study/'protocol.json').read_text());job=prot['jobs'][jobid];lock=json.loads((study/'checkpoint_lock.json').read_text());assert lock['status']=='PASS' and len(lock['rows'])==48
 frozen=next(x for x in lock['rows'] if x['id']==jobid);rd=Path(job['run_dir']);dest=Path(job['evaluation_dir']);dest.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((rd/'config.json').read_text());assert w.sha(rd/'best.pt')==frozen['checkpoint_sha256'] and cfg['config_sha256']==frozen['config_sha256']
 for name,h in prot['code_sha256'].items():assert w.sha(w.ROOT/name)==h
 manifest=json.loads(Path(job['manifest']).read_text());assert w.sha(job['manifest'])==job['manifest_sha256'];split_fields={};result={};bundle=None
 prior=w.ROOT/'outputs/v3_generalization_20261009';old=json.loads((prior/'protocol.json').read_text())
 candidates=[r for r in old['rows'] if r['checkpoint_sha256']==frozen['checkpoint_sha256'] and Path(r['checkpoint'])==rd/'best.pt']
 if candidates:
  row=candidates[0];assert row['manifest_sha256']==job['manifest_sha256'];src=prior/'runs'/f"{row['arm']}_seed{row['seed']}"
  assert json.loads((prior/'final_integrity.json').read_text())['status']=='PASS'
  for split in ['val','train','test']:
   path=src/(split+'_fields.npz');meta=json.loads((src/(split+'_metrics.json')).read_text());assert meta['checkpoint_sha256']==frozen['checkpoint_sha256']
   with np.load(path) as z:f={k:z[k] for k in z.files}
   met=metrics(f);assert abs(meta['rmse']-met['rmse'])<1e-12
   starts=manifest[('val' if split=='val' else split)+'_start_indices'];assert np.array_equal(f['target_index'],np.array(starts)+29+job['lead']) and np.all(f['daily_count']==661)
   split_fields[split]=dict(path=str(path),sha256=w.sha(path),reused=True);result[split]=met
 else:
  torch.set_num_threads(2);device=torch.device('cuda:0');w.gpu_identity(dest)
  args=SimpleNamespace(**cfg['settings']);bundle,m=p.load_subset_bundle(job['manifest']);assert bundle.stats=={k:tuple(v) for k,v in cfg['stats'].items()}
  model=t.JejuHiLAM(graph=bundle.graph,input_dim=len(bundle.train.input_feature_names),hidden_dim=args.hidden_dim,temporal_hidden_dim=args.temporal_hidden_dim,processor_layers=args.processor_layers,mlp_hidden_layers=args.mlp_hidden_layers,prediction_mode=args.prediction_mode,physics_heads=True,we_nonneg=True,we_scale=args.we_scale,rres_scale=args.rres_scale,we_softplus_shift=args.we_softplus_shift).to(device)
  cp=torch.load(rd/'best.pt',map_location='cpu',weights_only=False);assert cp['config_sha256']==cfg['config_sha256'] and cp['best_epoch']==frozen['best_epoch'];model.load_state_dict(cp['model_state'],strict=True);model.eval()
  pa=t.build_physics_args(args);p.apply_manifest_physics(args,pa,m)
  for split in ['val','train','test']:
   with torch.inference_mode():f=t.collect_predictions(model,t.make_loader(getattr(bundle,split),args,False),bundle.graph,device,*bundle.stats['target'],precision='fp32',physics_args=pa)
   assert np.array_equal(f['target_index'],getattr(bundle,split+'_indices')+29+job['lead'])
   f['target_dates']=bundle.ds.time.values[f['target_index']];f['seed']=np.array(job['seed']);f['model']=np.array(job['arm']);f['ratio']=np.array(job['ratio']);f['lead']=np.array(job['lead'])
   met=metrics(f);assert np.all(f['daily_count']==661)
   if split=='val':assert abs(met['rmse']-frozen['best_val_rmse'])<2e-6
   path=dest/(split+'_fields.npz');np.savez_compressed(path,**f);split_fields[split]=dict(path=str(path),sha256=w.sha(path),reused=False);result[split]=met
 assert abs(result['val']['rmse']-frozen['best_val_rmse'])<2e-6
 assert w.sha(rd/'best.pt')==frozen['checkpoint_sha256']
 w.save(dest/'fields_index.json',split_fields);w.save(dest/'metrics.json',dict(status='completed',job=jobid,checkpoint_sha256=frozen['checkpoint_sha256'],metrics=result,fields=split_fields,fixed_eval_mode=True,timestamp=w.stamp()))
 if bundle:bundle.ds.close()
 print(json.dumps(dict(job=jobid,status='completed',metrics=result)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);o=ap.parse_args()
 try:main(o.study,o.job)
 except Exception:
  import traceback
  j=json.loads((o.study/'protocol.json').read_text())['jobs'][o.job];d=Path(j['evaluation_dir']);d.mkdir(parents=True,exist_ok=True);w.save(d/'failure.json',dict(error=traceback.format_exc(),timestamp=w.stamp()));raise
