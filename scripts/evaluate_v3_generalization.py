"""Frozen-checkpoint generalization audit; inference only, separate outputs."""
import os, sys, json, hashlib, math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v3_generalization_20261009'
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','MPLCONFIGDIR']:
 d=OUT/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import train_v3_study as w
np=w.np;torch=w.torch;t=w.t;p=w.p

def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sst_metrics(f):
 sea=f['mask'].astype(bool);err=f['pred_raw'].astype(np.float64)-f['target_raw'].astype(np.float64)
 assert np.isfinite(err[sea]).all()
 daily_count=sea.reshape(len(sea),-1).sum(1)
 daily_sse=np.where(sea,err**2,0).reshape(len(sea),-1).sum(1)
 f['daily_sse']=daily_sse;f['daily_count']=daily_count;f['daily_mse']=daily_sse/daily_count
 return dict(rmse=float(np.sqrt(daily_sse.sum()/daily_count.sum())),mae=float(np.abs(err[sea]).mean()),bias=float(err[sea].mean()))

def prepare():
 source=ROOT/'outputs/aux_weight_selection_20261009'
 final=json.loads((source/'final_results.json').read_text());assert final['status']=='COMPLETE'
 rows=[]
 for row in final['rows']:
  d=Path(row['run_dir']);cfg=json.loads((d/'config.json').read_text());h=json.loads((d/'history.json').read_text());best=min(h,key=lambda z:z['val_rmse'])
  assert best['epoch']==row['best_epoch'] and best['val_rmse']==row['val_rmse']
  protpath=ROOT/('outputs/p0_lambda_v3_20261009/protocol.json' if row['arm']=='physics_only' else 'outputs/aux_weight_selection_20261009/protocol.json')
  prot=json.loads(protpath.read_text());assert w.digest(prot)==cfg['protocol_sha256']
  for name,expected in prot['code_sha256'].items():assert w.sha(ROOT/name)==expected,name
  cc=dict(cfg);ch=cc.pop('config_sha256');assert w.digest(cc)==ch
  weights={'baseline':(0,0),'physics_only':(.01,0),'auxiliary_only':(0,.0001),'full_pinn':(.01,.0001)}[row['arm']]
  assert (cfg['settings']['lambda_physics'],cfg['settings']['lambda_rres'])==weights
  assert cfg['parameters']['we_head']>0 and cfg['parameters']['rres_head']>0
  rows.append(dict(**row,checkpoint=str(d/'best.pt'),checkpoint_sha256=w.sha(d/'best.pt'),config_file=str(d/'config.json'),config_file_sha256=w.sha(d/'config.json'),protocol_file=str(protpath),manifest=cfg['job']['manifest'],manifest_sha256=cfg['manifest_sha256']))
 for seed in [42,7,123]:
  cs=[json.loads(Path(r['config_file']).read_text()) for r in rows if r['seed']==seed]
  for key in ['manifest_sha256','stats','split_sha256','tensor_fingerprints','parameters','trainable_parameters','shared_initial_sha256','rhs_scale','applied_rhs_scale','evaluation_mask_cells']:
   assert all(c[key]==cs[0][key] for c in cs),key
 protocol=dict(created_at=w.stamp(),scope='Exploratory Generalization Evaluation; no training/tuning/checkpoint reselection',rows=rows,bootstrap=dict(primary_block_days=90,sensitivity_block_days=[30,180],replicates=4000,seed_rule='20260925 + lead(7) + block_days',method='paired circular time blocks; shared indices across arms and fitted seeds; seed SD separate',reason='Retain prior v3 analysis predeclared 90-day primary; longer than history30+lead7=37-day overlap. SST error dependence sensitivity at30/180; ACF reported descriptively, no outcome-dependent block selection.'),validation_reproduction_atol_K=2e-6,mhw=dict(clim_frac=.7,quantile=.9,window=5,min_duration=5,area_frac=.15,onset_tolerance=3,implementation='src/summarize_jeju_hilam_mhw_skill.py:row_from_arrays; src/mhw/mhw_skill.py',reference='unchanged full-record first round(N*.70) raw SST dates; separate from subset input anomaly climatology'),code_sha256={str(x.relative_to(ROOT)):w.sha(x) for x in [Path(__file__),ROOT/'src/train_jeju_hilam.py',ROOT/'src/mhw/subset_preprocessing.py',ROOT/'src/mhw/mhw_skill.py',ROOT/'src/summarize_jeju_hilam_mhw_skill.py']})
 path=OUT/'protocol.json'
 if path.exists():
  old=json.loads(path.read_text());protocol['created_at']=old['created_at'];assert protocol==old
 else:w.save(path,protocol)
 w.log('v3 generalization evaluation protocol frozen:12 Validation-best checkpoints; no training;90-day paired circular bootstrap4000,30/180 sensitivity; unchanged MHW; GPU1 inference.')
 return protocol

def evaluate(protocol):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
 torch.set_num_threads(2);device=torch.device('cuda:0');w.gpu_identity(OUT)
 bundles={};reference={};audits=[];modelsigs=[]
 def load(row):
  cfg=json.loads(Path(row['config_file']).read_text());a=SimpleNamespace(**cfg['settings']);seed=row['seed']
  if seed not in bundles:bundles[seed]=p.load_subset_bundle(row['manifest'])
  bundle,m=bundles[seed]
  assert w.sha(row['manifest'])==row['manifest_sha256']
  assert m['schema']=='subset_normalization_v3' and bundle.stats=={k:tuple(v) for k,v in cfg['stats'].items()}
  pa=t.build_physics_args(a);p.apply_manifest_physics(a,pa,m);assert pa.physics_rhs_scale==cfg['applied_rhs_scale']
  model=t.JejuHiLAM(graph=bundle.graph,input_dim=len(bundle.train.input_feature_names),hidden_dim=a.hidden_dim,temporal_hidden_dim=a.temporal_hidden_dim,processor_layers=a.processor_layers,mlp_hidden_layers=a.mlp_hidden_layers,prediction_mode=a.prediction_mode,physics_heads=True,we_nonneg=True,we_scale=a.we_scale,rres_scale=a.rres_scale,we_softplus_shift=a.we_softplus_shift).to(device)
  cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
  assert w.sha(row['checkpoint'])==row['checkpoint_sha256'] and cp['config_sha256']==cfg['config_sha256'] and cp['manifest_sha256']==row['manifest_sha256']
  assert cp['best_epoch']==row['best_epoch'] and cp['best_value']==row['val_rmse']
  assert {k:tuple(v) for k,v in cp['stats'].items()}==bundle.stats
  model.load_state_dict(cp['model_state'],strict=True);model.eval()
  sig={k:list(v.shape) for k,v in model.state_dict().items()}
  if modelsigs:assert sig==modelsigs[0]
  else:modelsigs.append(sig)
  for split in ['train','val','test']:
   assert w.digest(getattr(bundle,split+'_indices').tolist())==cfg['split_sha256'][split]
   ds=getattr(bundle,split);h=hashlib.sha256()
   for i in sorted(set([0,len(ds)//2,len(ds)-1])):
    sample=ds[i]
    for k in ['x_nodes','y_nodes','target_index']:h.update(k.encode());h.update(sample[k].numpy().tobytes())
   assert h.hexdigest()==cfg['tensor_fingerprints'][split]
  return cfg,a,bundle,pa,model
 def collect(row,split,a,bundle,pa,model):
  dest=OUT/'runs'/f"{row['arm']}_seed{row['seed']}";dest.mkdir(parents=True,exist_ok=True)
  path=dest/(split+'_fields.npz');meta=dest/(split+'_metrics.json')
  if path.exists() and meta.exists():
   info=json.loads(meta.read_text());assert info['checkpoint_sha256']==row['checkpoint_sha256'] and info['protocol_sha256']==w.digest(protocol)
   with np.load(path) as z:f={k:z[k] for k in z.files}
  else:
   with torch.inference_mode():f=t.collect_predictions(model,t.make_loader(getattr(bundle,split),a,False),graph=bundle.graph,device=device,target_mean=bundle.stats['target'][0],target_std=bundle.stats['target'][1],precision='fp32',physics_args=pa)
   f['target_dates']=bundle.ds.time.values[f['target_index']];f['seed']=np.array(row['seed']);f['model']=np.array(row['arm'])
   metrics=sst_metrics(f)
   info=dict(**metrics,checkpoint_sha256=row['checkpoint_sha256'],protocol_sha256=w.digest(protocol),split=split,n=len(f['target_index']),first_date=str(f['target_dates'][0]),last_date=str(f['target_dates'][-1]),eval_mode=True,optimizer_created=False)
   np.savez_compressed(path,**f);w.save(meta,info)
  assert np.array_equal(f['target_index'],getattr(bundle,split+'_indices')+29+7)
  assert np.all(f['mask'].reshape(len(f['mask']),-1).sum(1)==661)
  signatures=[arrhash(f[k]) for k in ['target_index','target_raw','mask']]
  key=(split,row['seed']) if split=='train' else (split,)
  if key in reference:assert signatures==reference[key],(row,split)
  else:reference[key]=signatures
  print(json.dumps(dict(arm=row['arm'],seed=row['seed'],**info)),flush=True)
  return info
 # Global gate: every Validation must reproduce before ANY Test inference.
 for row in protocol['rows']:
  cfg,a,bundle,pa,model=load(row);info=collect(row,'val',a,bundle,pa,model)
  diff=info['rmse']-row['val_rmse'];assert abs(diff)<=protocol['validation_reproduction_atol_K'],(row,diff)
  audits.append(dict(arm=row['arm'],seed=row['seed'],checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],reported_val_rmse=row['val_rmse'],reproduced_val_rmse=info['rmse'],difference=diff,status='PASS'))
  del model;torch.cuda.empty_cache()
 w.save(OUT/'checkpoint_audit.json',dict(status='PASS',rows=audits,model_state_shapes=modelsigs[0],all_12_validation_gate=True))
 for row in protocol['rows']:
  cfg,a,bundle,pa,model=load(row)
  for split in ['train','test']:collect(row,split,a,bundle,pa,model)
  assert w.sha(row['checkpoint'])==row['checkpoint_sha256']
  del model;torch.cuda.empty_cache()
 for bundle,_ in bundles.values():bundle.ds.close()
 w.save(OUT/'inference_status.json',dict(status='COMPLETE',models=12,splits=['train','val','test'],timestamp=w.stamp()))
 w.log('v3 generalization:12/12 checkpoints Validation reproduced; fixed eval-mode Train/Test inference complete; original checkpoints preserved.')
if __name__=='__main__':
 try:
  protocol=prepare();evaluate(protocol)
 except Exception:
  import traceback
  w.save(OUT/'failure.json',dict(timestamp=w.stamp(),error=traceback.format_exc()));raise
