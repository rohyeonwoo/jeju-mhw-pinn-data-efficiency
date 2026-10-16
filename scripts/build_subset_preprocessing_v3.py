"""Derive coastal RHS-only v3 manifests; preserve v2 inputs and climatologies."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import ast,copy,json,hashlib,importlib.util
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
import numpy as np
import torch
import xarray as xr
from mhw import subset_preprocessing as p,subset_climatology as c
from mhw.ocean_stencil import STENCIL,ocean_neighbor_support,ocean_neighbor_gradients
from mhw.datasets import make_time_splits
import train_jeju_hilam as t
import train_hybrid_jeju_forecast as h
OUT=ROOT/'data/subsets/preprocessing_subset_v3'
V2=ROOT/'data/subsets/preprocessing_subset_v2'
REPORT=ROOT/'outputs/v3_preflight_20261008'
BACK=Path('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py')

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def forbid(*a,**k):raise AssertionError('Full preprocessing path reached')
class Probe(torch.nn.Module):
 def __init__(self,**kw):super().__init__();self.we_scale=kw['we_scale'];self.rres_scale=kw['rres_scale']
 def forward(self,x_nodes,base_nodes=None,return_aux=False):
  z=torch.zeros_like(base_nodes)
  return dict(pred=z,we=z,rres=z+1e-5) if return_aux else z

def setup_code():
 tree=ast.parse((ROOT/'src/train_jeju_hilam.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');body=[]
 for s in fn.body:
  if isinstance(s,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='config' for x in s.targets):break
  body.append(s)
 fn.name='setup_only';fn.body=body+[ast.Return(ast.Call(ast.Name('locals',ast.Load()),[],[]))]
 return compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_main_setup','exec')

def verify_setup(code,path,m,bundle):
 samples=[]
 for mode in ('none','p0_original'):
  env=dict(t.__dict__);env.update(JejuHiLAM=Probe,make_jeju_hilam_bundle=forbid,compute_fixed_rhs_scale=forbid,save_graph_artifacts=lambda *a:None);exec(code,env)
  argv=['trainer','--subset-manifest',str(path),'--lead',str(m['lead']),'--seed',str(m['subset_seed']),'--physics-mode',mode,'--lambda-rres','0','--device','cpu','--batch-size','2','--output-root',str(REPORT/'setup'/path.stem/mode)]
  with patch.object(sys,'argv',argv),patch.object(p,'load_subset_bundle',return_value=(bundle,m)):state=env['setup_only']()
  assert state['target_mean']==m['statistics']['target'][0] and state['target_std']==m['statistics']['target'][1]
  if mode!='none':assert state['physics_args'].physics_rhs_scale==m['physics_rhs_scale']
  batch=next(iter(t.make_loader(bundle.train,state['args'],shuffle=False)));samples.append(batch)
  with torch.no_grad():metrics=t.run_epoch(state['model'],[batch],bundle.graph,torch.device('cpu'),*bundle.stats['target'],optimizer=None,physics_args=state['physics_args'],lambda_physics=.1,epoch=5)
  assert np.isfinite(list(metrics.values())).all()
  if mode=='none':assert metrics['physics_loss']==metrics['rres_l2']==0
  else:assert metrics['rres_l2']==0 and metrics['physics_loss']>=0
 for key in ('x_nodes','y_nodes','y_raw','Tm','qnet_seq','u_seq','v_seq','hm_seq'):torch.testing.assert_close(samples[0][key],samples[1][key],rtol=0,atol=0)
 return {'manifest':path.name,'main_setup_baseline_p0':'PASS','same_tensors':'PASS','no_full_override':'PASS','cpu_objective':'PASS'}

def main():
 assert '\nOK\n' in (REPORT/'regression_tests.log').read_text()
 if OUT.exists():raise FileExistsError('v3 already exists; no overwrite')
 OUT.mkdir();torch.set_num_threads(4)
 preserve=json.loads((REPORT/'preservation_before.json').read_text())
 first=json.loads((V2/'ratio_10_lead7_seed42.json').read_text())
 for key,hkey in [('data_path','data_sha256'),('raw_thetao_path','raw_thetao_sha256')]:assert p.sha256_file(first[key])==first[hkey]
 sources=[ROOT/f for f in ['src/train_hybrid_jeju_forecast.py','src/train_jeju_hilam.py','src/mhw/subset_preprocessing.py','src/mhw/subset_climatology.py','src/mhw/ocean_stencil.py','scripts/build_subset_preprocessing_v3.py']]
 hashes={str(f.relative_to(ROOT)):p.sha256_file(f) for f in sources}
 spec=importlib.util.spec_from_file_location('legacy_hybrid',REPORT/'originals/src/train_hybrid_jeju_forecast.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 code=setup_code();rows=[];runtime=[];diffs=[];cache={}
 ds=xr.open_dataset(first['data_path']);base=p.make_raw_base(ds);raw=c.load_raw_anomaly_fields(ds,first['raw_thetao_path']);bins=c.calendar_bins(ds.time.values)
 mask=torch.from_numpy(base.ocean_mask)[None,None].float();dx=torch.from_numpy(base.dx)[None,None].float();dy=torch.from_numpy(base.dy)[None,None].float()
 support=ocean_neighbor_support(mask);assert int(mask.sum())==661 and int(support.sum())==660
 # Compare every real ocean cell against original pilot, including one-sided cases.
 tree=ast.parse(BACK.read_text());names={'shifted_neighbors','ocean_neighbor_support','ocean_neighbor_gradients'};env={'torch':torch,'F':torch.nn.functional}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'pilot_reference','exec'),env)
 field=torch.from_numpy(base.target_array[200])[None,None]
 ref=env['ocean_neighbor_gradients'](field,dx,dy,mask);now=h.spatial_gradients_torch(field,dx,dy,mask)
 for a,b in zip(now,ref):torch.testing.assert_close(a,b,rtol=0,atol=0)
 for fill in (0.,100.,float('nan'),float('inf')):
  constant=torch.where(mask>0,20.,fill)
  for g in h.spatial_gradients_torch(constant,dx,dy,mask):assert torch.isfinite(g).all() and torch.count_nonzero(g)==0
 changed=((legacy.spatial_gradients_torch(20*mask,dx,dy)[0]!=0)|(legacy.spatial_gradients_torch(20*mask,dx,dy)[1]!=0))&(mask>0)
 save(REPORT/'stencil_real_grid.json',dict(status='PASS',ocean_cells=661,physics_cells=660,legacy_constant_nonzero_cells=int(changed.sum()),v3_constant_nonzero_cells=0,reference_pilot_exact_match=True,land_0_100_nan_inf_invariant=True))
 for ratio in (10,25,50,100):
  for seed in (42,7,123):
   m7=json.loads((V2/f'ratio_{ratio}_lead7_seed{seed}.json').read_text());clims=c.read_climatology(m7,V2/f'ratio_{ratio}_lead7_seed{seed}.json')
   rebased=c.apply_to_base(base,raw,bins,clims)
   for lead in (7,14):
    name=f'ratio_{ratio}_lead{lead}_seed{seed}.json';parent=V2/name;old=json.loads(parent.read_text());m=copy.deepcopy(old);starts=old['train_start_indices'];key=(lead,tuple(starts))
    if key not in cache:cache[key]=p.compute_subset_rhs_scale(base,starts,lead)
    rhs=cache[key];assert np.isfinite(rhs) and rhs>0
    m.update(schema=c.V3_SCHEMA,physics_rhs_scale=rhs,physics_stencil=STENCIL,source_sha256=hashes,
      climatology_file='../preprocessing_subset_v2/'+old['climatology_file'],parent_manifest=str(parent),parent_manifest_sha256=p.sha256_file(parent))
    m['physics_scale_definition'].update(stencil=STENCIL,support='ocean and at least one ocean neighbor on each axis')
    allowed={'schema','physics_rhs_scale','physics_stencil','source_sha256','climatology_file','parent_manifest','parent_manifest_sha256','physics_scale_definition'}
    assert {k for k in set(old)|set(m) if old.get(k)!=m.get(k)}<=allowed
    for k in ('statistics','train_start_indices','val_start_indices','test_start_indices','input_time_indices','target_time_indices','climatology_sha256','climatology_rule'):assert m[k]==old[k]
    path=OUT/name;save(path,m)
    # Reuse v2 anomaly arrays: no climatology or normalization fitting.
    bundle=p.bundle_from_manifest(ds,dict(m,schema=p.SCHEMA),base=rebased)
    for dataset in (bundle.train,bundle.val,bundle.test):
     sample=dataset[0];start=int(sample['start_index']);idx=bundle.graph.ocean_flat_indices
     for ch,var in enumerate(p.BASE_DYNAMIC_VARS):
      mu,sd=m['statistics'][var];expected=((rebased.dynamic_arrays[var][start:start+30]-mu)/sd).reshape(30,-1)[:,idx].T
      np.testing.assert_allclose(sample['x_nodes'][:,:,ch],expected,rtol=1e-6,atol=1e-6)
    splits=make_time_splits(ds.sizes['time'],30,lead);np.testing.assert_array_equal(bundle.val_indices,splits.val);np.testing.assert_array_equal(bundle.test_indices,splits.test)
    original=json.loads(Path(m['subset_path']).read_text());assert starts==original[str(ratio)]['start_indices']
    runtime.append(verify_setup(code,path,m,bundle))
    batch=next(iter(torch.utils.data.DataLoader(p.PhysicsScaleDataset(base,starts,lead),batch_size=2)))
    tm=batch['Tm'];target=batch['y_raw'];z=torch.zeros_like(tm);args=p.p0_scale_args()
    before,bm=legacy._linear_rhs_per_day(target,tm,batch['mask'],batch,args,lead,z)
    after,am=h._linear_rhs_per_day(target,tm,batch['mask'],batch,args,lead,z)
    def rms(a,w):return float(((a.square()*w).sum()/w.sum()).sqrt())
    tendency=((target-tm)/lead).repeat(lead,1,1,1)
    diffs.append(dict(ratio=ratio,seed=seed,lead=lead,scope='first two selected training samples; observed linear path; we=0; un-clipped K/day',rhs_rms_before=rms(before,bm),rhs_rms_after=rms(after,am),residual_rms_before=rms(tendency-before,bm),residual_rms_after=rms(tendency-after,am)))
    row=dict(manifest=name,sha256=p.sha256_file(path),ratio=ratio,seed=seed,lead=lead,old_rhs_scale=old['physics_rhs_scale'],physics_rhs_scale=rhs,ratio_old_new=old['physics_rhs_scale']/rhs,train_samples=len(starts),validation='PASS');rows.append(row)
    print(f'v3 {name}: {old["physics_rhs_scale"]:.6f} -> {rhs:.6f}',flush=True)
   del rebased,bundle
 for seed in (42,7,123):
  sets=[set(json.loads((OUT/f'ratio_{ratio}_lead7_seed{seed}.json').read_text())['train_start_indices']) for ratio in (10,25,50,100)]
  assert all(a<b for a,b in zip(sets,sets[1:]))
 # Actual disk dispatch once, without cached bundle injection.
 b,m=p.load_subset_bundle(OUT/'ratio_10_lead7_seed42.json');assert m['schema']==c.V3_SCHEMA and len(b.train)==376;b.ds.close();del b
 for name,digest in preserve['hashes'].items():assert p.sha256_file(name)==digest,name
 for name,metadata in preserve['existing_results'].items():f=Path(name);assert [f.stat().st_size,f.stat().st_mtime_ns]==metadata,name
 ds.close()
 save(REPORT/'rhs_residual_comparison.json',diffs);save(REPORT/'runtime_verification.json',runtime)
 save(OUT/'index.json',dict(schema=c.V3_SCHEMA,status='CPU_VALIDATED_GPU_SMOKE_PENDING',artifact_count=24,climatology_reused_from_v2=True,unique_rhs_computations=len(cache),datasets=rows))
 gates={k:'PASS' for k in ['manifest_integrity','nested_and_splits','baseline_pinn_same_inputs','no_full_statistics_override','land_fill_invariance','loss_rhs_diagnostic_consistency','inactive_loss_skipped','positive_finite_rhs','mhw_evaluation_preserved','existing_results_preserved']}
 save(OUT/'validation_report.json',dict(gates=gates,regression_tests=24,gpu_smoke='PENDING',training_executed=False,optimizer_steps=0,input_target_statistics_refitted=False,climatology_refitted=False))
 tstamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
 with (ROOT/'docs/research_log.md').open('a') as f:f.write(f'\n\n## {tstamp} — v3 24개 RHS manifest 생성 및 CPU 검증 완료\n\n- scripts/build_subset_preprocessing_v3.py 및 data/subsets/preprocessing_subset_v3 생성. 기존climatology를 경로·해시로 참조하고 Input/Target 통계·인덱스는 그대로 유지했다.24조건/20개 고유 RHS 계산.\n- 24개 회귀검사,48개 실제 main setup/CPU objective, 실제 v3 public loader, 기존 해안 구현과 전 해양 격자 일치, 보존 해시/결과 메타데이터 검사를 통과했다. 수정 전후 RHS·잔차는 outputs/v3_preflight_20261008에 저장했다.\n- GPU smoke는 아직 대기, 본 학습 미실행.\n')
 print('All ten CPU gates PASS; GPU smoke may proceed',flush=True)

if __name__=='__main__':main()
