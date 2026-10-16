"""CPU-only final audit; no optimizer, backward, model training or source edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TMPDIR'] = '/ceph_data/yeonwoo/MHW_subset/docs/pretraining_audit_20261008'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from unittest.mock import patch
import ast, copy, hashlib, json, textwrap
import setproctitle
setproctitle.setproctitle('yeonwoo')
import numpy as np
import pandas as pd
import torch
import xarray as xr
ROOT = Path('/ceph_data/yeonwoo/MHW_subset')
OUT = ROOT/'docs/pretraining_audit_20261008'
sys.path.insert(0, str(ROOT/'src'))
import train_jeju_hilam as trainer
import train_hybrid_jeju_forecast as physics
import mhw.subset_preprocessing as prep
from mhw.datasets import make_time_splits

torch.set_num_threads(4)
def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')
def forbidden(*a, **kw):
    raise AssertionError('Forbidden Full-statistics fitting path reached')

man_dir = ROOT/'data/subsets/preprocessing_subset_v1'
index = json.loads((man_dir/'index.json').read_text())
first = json.loads((man_dir/index['datasets'][0]['manifest']).read_text())
data_path = Path(first['data_path'])
data_hash = prep.sha256_file(data_path)
assert data_hash == first['data_sha256']
raw_ds = xr.open_dataset(data_path)
metadata = {'global':dict(raw_ds.attrs), 'anomalies':{n:dict(raw_ds[n].attrs) for n in raw_ds if 'anom' in n}, 'data_sha256':data_hash}
save('netcdf_metadata.json',metadata)
print('Metadata saved',flush=True)

# Re-run the exact anomaly block recovered from the preprocessing source.
source = Path('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/lab/MHW/preprocess_glorys3d_budget_terms.py')
code_lines = source.read_text().splitlines()
clim_code = compile(textwrap.dedent('\n'.join(code_lines[140:159])), str(source), 'exec')
precursor = Path('/ceph_data/yeonwoo/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d.nc')
clim_result = {'source':str(source),'source_sha256':prep.sha256_file(source),'precursor':str(precursor),'depths':[]}
with xr.open_dataset(precursor) as old:
    np.testing.assert_array_equal(old.time.values,raw_ds.time.values)
    np.testing.assert_array_equal(old.sst_anom.values,raw_ds.sst_anom.values)
    clim_result['sst_anom_preserved_from_precursor_exactly'] = True
    depths=old.depth.values
    ntrain=int(round(len(old.time)*0.7)); dates=old.time.values.astype('datetime64[D]')
    clim_result.update(ntrain=ntrain,reference_start=str(dates[0]),reference_end=str(dates[ntrain-1]))
    for name in [n for n in raw_ds if n.startswith('thetao_') and n.endswith('_anom')]:
        z=int(name.split('_z')[1].split('m')[0]); zi=int(np.argmin(abs(depths-z)))
        arr=old.thetao.isel(depth=zi).values.astype(np.float32)
        env={'np':np,'pd':pd,'nt':len(dates),'args':SimpleNamespace(train_ratio=0.7),
             'ds':old,'Y':29,'X':33,'lvl_raw':{('thetao',zi):arr}}
        exec(clim_code,env)
        rebuilt=env['lvl_anom'][('thetao',zi)]; actual=raw_ds[name].values
        finite=np.isfinite(actual)&np.isfinite(rebuilt)
        exact=np.array_equal(actual,rebuilt,equal_nan=True)
        row={'variable':name,'depth_index':zi,'depth_m':float(depths[zi]),'exact_equal':exact,
             'max_abs_error':float(np.max(abs(actual[finite]-rebuilt[finite]))),
             'finite_pattern_equal':bool(np.array_equal(np.isfinite(actual),np.isfinite(rebuilt)))}
        clim_result['depths'].append(row)
        print('Anomaly comparison',row,flush=True)
save('anomaly_reconstruction.json',clim_result)

# Execute actual main() setup statements until config creation. Replace model
# construction with a zero-output CPU probe; never execute optimizer/training.
module_ast=ast.parse((ROOT/'src/train_jeju_hilam.py').read_text())
main=next(n for n in module_ast.body if isinstance(n,ast.FunctionDef) and n.name=='main')
body=[]
for stmt in main.body:
    if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in stmt.targets):
        break
    body.append(copy.deepcopy(stmt))
body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])))
main.name='audit_setup';main.body=body
compiled=compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'actual_main_setup','exec')
class Probe(torch.nn.Module):
    def __init__(self,**kw):
        super().__init__();self.we_scale=kw['we_scale'];self.rres_scale=kw['rres_scale']
    def forward(self,x_nodes,base_nodes=None,return_aux=False):
        p=torch.zeros_like(base_nodes)
        return {'pred':p,'we':p,'rres':p} if return_aux else p

# Cache immutable raw arrays and the verified hash, not any preprocessing state.
needed=list(dict.fromkeys([*prep.BASE_DYNAMIC_VARS,'qnet','ocean_mask_static',*[n for n in ('u_mld','v_mld','vadv_kday') if n in raw_ds]]))
ds=raw_ds[needed].load();base=prep.make_raw_base(ds)
real_bundle=prep.bundle_from_manifest
cached_load=lambda path,data_path=None:(real_bundle(ds,json.loads(Path(path).read_text()),base=base),json.loads(Path(path).read_text()))
# Independently exercise unmodified disk loader once.
b,m=prep.load_subset_bundle(man_dir/index['datasets'][0]['manifest']); assert len(b.train)==376
b.ds.close();del b
result={'timestamp_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'training_executed':False,'gpu_used':False,
        'setup_method':'actual main AST prefix before config/optimizer; cached immutable raw arrays; CPU zero-output probe; run_epoch optimizer=None',
        'disk_loader_roundtrip':'PASS','datasets':[]}
setmap={}
for row in index['datasets']:
    path=man_dir/row['manifest'];m=json.loads(path.read_text());lead=m['lead'];seed=m['subset_seed'];ratio=m['subset_ratio']
    assert prep.sha256_file(path)==row['sha256']
    for name,sha in m['source_sha256'].items(): assert prep.sha256_file(ROOT/name)==sha
    original=json.loads(Path(m['subset_path']).read_text());starts=np.array(m['train_start_indices'])
    np.testing.assert_array_equal(starts, original[str(ratio)]['start_indices'])
    splits=make_time_splits(ds.sizes['time'],30,lead)
    assert np.isin(starts,splits.train).all()
    for split in ('val','test'): np.testing.assert_array_equal(m[f'{split}_start_indices'],getattr(splits,split))
    setmap[(lead,seed,ratio)]=set(starts.tolist())
    fresh=prep.fit_subset_stats(ds,starts,30,lead)
    for name,pair in fresh.items(): np.testing.assert_allclose(pair,m['statistics'][name],rtol=0,atol=0)
    rhs=prep.compute_subset_rhs_scale(base,starts,lead)
    np.testing.assert_allclose(rhs,m['physics_rhs_scale'],rtol=0,atol=0)
    arms={}; losses={}
    for mode in ('none','p0_original'):
        argv=['train_jeju_hilam.py','--subset-manifest',str(path),'--lead',str(lead),'--seed',str(seed),
              '--device','cpu','--physics-mode',mode,'--output-root',str(OUT/'setup_scratch'/row['manifest'][:-5]/mode)]
        env=dict(trainer.__dict__)
        env.update(JejuHiLAM=Probe,make_jeju_hilam_bundle=forbidden,compute_fixed_rhs_scale=forbidden,save_graph_artifacts=lambda *a:None)
        exec(compiled,env)
        with patch.object(sys,'argv',argv),patch.object(prep,'load_subset_bundle',cached_load): state=env['audit_setup']()
        bundle=state['bundle'];pa=state['physics_args']
        np.testing.assert_array_equal(bundle.train_indices,starts)
        assert state['target_mean']==m['statistics']['target'][0] and state['target_std']==m['statistics']['target'][1]
        sample=bundle.train[0]
        arms[mode]={'indices':bundle.train_indices.tolist(),'stats':bundle.stats,'sample':sample}
        if mode=='p0_original': assert pa.physics_rhs_scale==m['physics_rhs_scale']
        loader=trainer.make_loader(bundle.train,state['args'],shuffle=False)
        metrics=trainer.run_epoch(state['model'],loader,bundle.graph,torch.device('cpu'),state['target_mean'],state['target_std'],
                                  optimizer=None,max_batches=1,physics_args=pa,lambda_physics=0.1,epoch=5)
        expected=float((next(iter(loader))['y_nodes']**2).mean())
        np.testing.assert_allclose(metrics['data_loss'],expected,rtol=1e-6)
        losses[mode]={k:metrics[k] for k in ('data_loss','physics_loss','rres_l2')}
    assert arms['none']['indices']==arms['p0_original']['indices'] and arms['none']['stats']==arms['p0_original']['stats']
    for key in ('x_nodes','y_nodes','y_raw','Tm','qnet_seq','u_seq','v_seq','hm_seq'):
        torch.testing.assert_close(arms['none']['sample'][key],arms['p0_original']['sample'][key],rtol=0,atol=0)
    inputs=np.array(m['input_time_indices']);window=np.unique((starts[:,None]+np.arange(30+lead)).ravel())
    targets=np.concatenate([splits.val,splits.test])+29+lead
    climate_days=np.arange(ntrain)
    entry={'manifest':row['manifest'],'checks_3_4_5_6':'PASS','loss_probe':losses,
           'theta_climatology_days_outside_input_fit':int(len(np.setdiff1d(climate_days,inputs))),
           'theta_climatology_days_outside_selected_sample_footprints':int(len(np.setdiff1d(climate_days,window))),
           'theta_climatology_heldout_target_dates':int(len(np.intersect1d(climate_days,targets)))}
    result['datasets'].append(entry)
    print('Dataset verified',row['manifest'],flush=True)
for lead in (7,14):
    for seed in (42,7,123):
        assert setmap[lead,seed,10]<setmap[lead,seed,25]<setmap[lead,seed,50]<setmap[lead,seed,100]
for seed in (42,7,123):
    for ratio in (10,25,50,100):assert setmap[7,seed,ratio]==setmap[14,seed,ratio]
result['nested_and_cross_lead_indices']='PASS'
result['all_datasets_verified']=len(result['datasets'])
# A conflicting Full scale must fail before it can be used.
args=SimpleNamespace(history=30,lead=7,physics_mode='p0_original',hm_min=5.,entrainment_dt=1.,physics_rhs_scale=9.736722977796306)
try:prep.apply_manifest_physics(args,SimpleNamespace(),first)
except ValueError:result['conflicting_full_scale_rejected']='PASS'
else:raise AssertionError('Full RHS override was accepted')
save('runtime_verification.json',result)
raw_ds.close()
print('All 24 dataset checks and 48 CPU evaluation probes passed. No training.',flush=True)
