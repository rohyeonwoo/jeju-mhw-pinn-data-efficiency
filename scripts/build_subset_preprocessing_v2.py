#!/usr/bin/env python3
"""Build/verify v2 input climatologies on CPU; never train a model."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import ast, copy, json, math
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
import numpy as np
import torch
import xarray as xr
import setproctitle
setproctitle.setproctitle('yeonwoo')
from mhw import subset_preprocessing as p
from mhw import subset_climatology as c
from mhw.datasets import make_time_splits
import train_jeju_hilam as trainer
from train_hybrid_jeju_forecast import daily_rollout_physics_loss

OUT=ROOT/'data/subsets/preprocessing_subset_v2'
V1=ROOT/'data/subsets/preprocessing_subset_v1'
RAW3D=Path('/ceph_data/yeonwoo/MHW/data/merged_sst_cmems_era5_wind10_qnet_jeju_1993_2022_FINAL_staticmask_filled_glorys3d.nc')

def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def log(text):
    t=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M')
    with (ROOT/'docs/research_log.md').open('a') as f:f.write(f'\n\n## {t} KST — v2 데이터셋 구축·검증\n\n{text}\n')

def digest_tree(directory):
    return {str(f.relative_to(directory)):p.sha256_file(f) for f in directory.rglob('*') if f.is_file()}

def no_full(*a,**kw):raise AssertionError('Full fitting path reached')

class Probe(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__();self.we_scale=kwargs['we_scale'];self.rres_scale=kwargs['rres_scale']
    def forward(self,x_nodes,base_nodes=None,return_aux=False):
        z=torch.zeros_like(base_nodes)
        return {'pred':z,'we':z,'rres':torch.full_like(z,1e-5)} if return_aux else z

def setup_code():
    tree=ast.parse((ROOT/'src/train_jeju_hilam.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=[]
    for s in fn.body:
        if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='config' for t in s.targets):break
        body.append(s)
    body.append(ast.Return(ast.Call(ast.Name('locals',ast.Load()),[],[])))
    fn.name='setup_only';fn.body=body
    return compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_trainer_setup','exec')

def verify_entrypoint(code, path, manifest, bundle):
    arms={}
    for mode in ('none','p0_original'):
        env=dict(trainer.__dict__)
        env.update(JejuHiLAM=Probe,make_jeju_hilam_bundle=no_full,compute_fixed_rhs_scale=no_full,save_graph_artifacts=lambda *a:None)
        exec(code,env)
        argv=['trainer','--subset-manifest',str(path),'--lead',str(manifest['lead']),
              '--seed',str(manifest['subset_seed']),'--physics-mode',mode,'--device','cpu',
              '--batch-size','2','--virtual-epoch-steps','478',
              '--output-root',str(OUT/'verification_setup'/path.stem/mode)]
        with patch.object(sys,'argv',argv),patch.object(p,'load_subset_bundle',lambda *a:(bundle,manifest)):
            state=env['setup_only']()
        np.testing.assert_array_equal(state['bundle'].train_indices,manifest['train_start_indices'])
        assert state['bundle'].stats==bundle.stats
        assert [state['target_mean'],state['target_std']]==list(bundle.stats['target'])
        batch=next(iter(trainer.make_loader(bundle.train,state['args'],shuffle=False)))
        pa=state['physics_args']
        metrics=trainer.run_epoch(state['model'],[batch],bundle.graph,torch.device('cpu'),
            state['target_mean'],state['target_std'],optimizer=None,max_batches=1,
            physics_args=pa,lambda_physics=.1,epoch=5)
        np.testing.assert_allclose(metrics['data_loss'],float((batch['y_nodes']**2).mean()),rtol=1e-6)
        if pa is not None:
            assert pa.physics_rhs_scale==manifest['physics_rhs_scale']
            np.testing.assert_allclose(metrics['rres_l2'],(.864**2)/pa.physics_rhs_scale,rtol=1e-5)
            bare=copy.copy(pa);bare.physics_normalize='none'
            pred=trainer.nodes_to_grid_torch(torch.full_like(batch['y_nodes'],state['target_mean']),bundle.graph)
            z=torch.zeros_like(batch['Tm'])
            bare_loss=daily_rollout_physics_loss(pred,batch['Tm'],batch['y_raw'],batch['mask'],batch,bare,manifest['lead'],z,z)
            np.testing.assert_allclose(metrics['physics_loss'],float(bare_loss)/pa.physics_rhs_scale,rtol=1e-5)
        arms[mode]=batch
    for key in ('x_nodes','y_nodes','y_raw','Tm','qnet_seq','u_seq','v_seq','hm_seq'):
        torch.testing.assert_close(arms['none'][key],arms['p0_original'][key],rtol=0,atol=0)
    return 'PASS: actual main setup + CPU optimizer-free loss; same Baseline/PINN tensors; no Full fit'

def quality(name,raw,anom,clim,counts,inputs,ocean,old_std):
    train=anom[inputs][:,ocean].astype(np.float64)
    all_values=anom[:,ocean].astype(np.float64)
    mu,sd=p._mean_std(train)
    raw_selected=raw[inputs][:,ocean]
    # A convex interpolation cannot create climatology outside selected raw bounds.
    assert np.all(clim[:,ocean]>=raw_selected.min(axis=0)[None]-1e-5)
    assert np.all(clim[:,ocean]<=raw_selected.max(axis=0)[None]+1e-5)
    assert np.isfinite(all_values).all() and float(train.std())>1e-6
    assert (anom[:,~ocean]==0).all() and (clim[:,~ocean]==0).all()
    zmax=float(np.max(np.abs((all_values-mu)/sd)))
    train_zmax=float(np.max(np.abs((train-mu)/sd)))
    ratio=float(all_values.std()/sd)
    # Declared numerical QC gates; not a claim that all real-world SST tails are Gaussian.
    passed=train_zmax<=20 and zmax<=30 and .1<=ratio<=10
    return {'variable':name,'status':'PASS' if passed else 'REVIEW_REQUIRED',
        'train_mean':mu,'train_std':sd,'all_dates_std':float(all_values.std()),
        'all_dates_abs_max_K':float(np.max(np.abs(all_values))),
        'train_abs_zmax':train_zmax,'all_dates_abs_zmax':zmax,
        'all_to_train_std_ratio':ratio,'v2_to_v1_train_std_ratio':sd/old_std,
        'all_dates_abs_quantiles_K':np.quantile(np.abs(all_values),[.5,.95,.99,.999]).tolist(),
        'nonfinite_ocean':int((~np.isfinite(all_values)).sum()),'land_neutral_zero':True,
        'climatology_convex_bounds':True,'ocean_support_count_min':int(counts[:,ocean].min()),
        'ocean_support_count_max':int(counts[:,ocean].max())}

def execution_plan(rows):
    common={'batch_size':16,'history':30,'lr':.0002,'initial_lr':.00002,'weight_decay':.1,
        'precision':'fp32','virtual_epoch_steps':478,'max_virtual_epochs':80,'max_optimizer_steps':38240,
        'warmup_virtual_epochs':5,'warmup_steps':2390,'validation_interval_steps':478,
        'early_stopping_patience_validations':12,'selection_metric':'val_rmse','prediction_mode':'direct',
        'zero_rres_physics':True,'lambda_we_tie':0.,'MHW_evaluation':'unchanged original reference/threshold code'}
    jobs=[]
    for row in rows:
        for arm,mode,lp,lr in [('baseline','none',0.,0.),('p0','p0_original',.1,10.)]:
            jobs.append({'experiment':'data_efficiency','arm':arm,'manifest':row['manifest'],
                'lead':row['lead'],'seed':row['seed'],'ratio':row['ratio'],'physics_mode':mode,
                'lambda_physics':lp,'lambda_rres':lr,'physics_heads':mode!='none'})
        if row['ratio']==10:
            for arm,lp,lr in [('baseline',0.,0.),('physics_only',.1,0.),('auxiliary_only',0.,10.),('full_pinn',.1,10.)]:
                jobs.append({'experiment':'ablation10','arm':arm,'manifest':row['manifest'],
                    'lead':row['lead'],'seed':row['seed'],'ratio':10,'physics_mode':'p0_original',
                    'lambda_physics':lp,'lambda_rres':lr,'physics_heads':True})
    for job in jobs:
        output=ROOT/'outputs/subset_v2'/job['experiment']/f"ratio_{job['ratio']}"/job['arm']
        job['argv']=['python','-B',str(ROOT/'src/train_jeju_hilam.py'),'--subset-manifest',str(OUT/job['manifest']),
            '--lead',str(job['lead']),'--seed',str(job['seed']),'--physics-mode',job['physics_mode'],
            '--lambda-physics',str(job['lambda_physics']),'--lambda-rres',str(job['lambda_rres']),
            '--batch-size','16','--virtual-epoch-steps','478','--output-root',str(output)]
    return {'status':'PLAN_ONLY_NOT_LAUNCHED','common':common,'jobs':jobs,'main_count':48,'ablation_count':24,
        'ablation_initialization':'All four arms use identical architecture, seed, minibatch schedule and optimizer; only loss weights change.',
        'diagnostics':['SST RMSE/MAE','existing MHW evaluator unchanged','physics residual'],
        'gpu_launch_requirement':'Future launcher must allocate GPUs and verify PID appears as yeonwoo in nvidia-smi; never launch all jobs at once.'}

def main():
    if OUT.exists():raise FileExistsError('v2 output exists; never overwrite an existing version')
    OUT.mkdir(parents=True);os.environ['TMPDIR']=str(OUT)
    torch.set_num_threads(4)
    oldhash=digest_tree(V1)
    evaluation_hash=p.sha256_file(ROOT/'src/mhw/mhw_skill.py')
    first=json.loads((V1/'ratio_10_lead7_seed42.json').read_text())
    data=Path(first['data_path']);assert p.sha256_file(data)==first['data_sha256']
    raw_hash=p.sha256_file(RAW3D)
    source_hash={str(f.relative_to(ROOT)):p.sha256_file(f) for f in (Path(__file__),ROOT/'src/mhw/subset_climatology.py',ROOT/'src/mhw/subset_preprocessing.py',ROOT/'src/train_jeju_hilam.py')}
    code=setup_code();rows=[];group_reports=[];qc_pass=True;runtime=[]
    with xr.open_dataset(data) as ds:
        needed=list(dict.fromkeys([*p.BASE_DYNAMIC_VARS,'qnet','ocean_mask_static',*[n for n in ('u_mld','v_mld','vadv_kday') if n in ds]]))
        ds=ds[needed].load();base=p.make_raw_base(ds);ocean=base.ocean_mask.astype(bool)
        raw=c.load_raw_anomaly_fields(ds,RAW3D);bins=c.calendar_bins(ds.time.values)
        for seed in (42,7,123):
            previous=set()
            for ratio in (10,25,50,100):
                m7=json.loads((V1/f'ratio_{ratio}_lead7_seed{seed}.json').read_text())
                inputs=np.array(m7['input_time_indices']);assert len(inputs)=={10:608,25:1596,50:3192,100:6460}[ratio]
                starts=np.array(m7['train_start_indices']);assert previous<set(starts);previous=set(starts)
                day_counts=np.bincount(bins[inputs],minlength=366)
                archive={'input_time_indices':inputs,'calendar_counts':day_counts,'ocean_mask':ocean}
                clims={};counts={};mutations=[]
                outside=np.setdiff1d(np.arange(len(bins)),inputs)
                print(f'Fitting v2 ratio={ratio} seed={seed}; {len(inputs)} input days',flush=True)
                for name in c.ANOMALIES:
                    clim,count=c.fit_climatology(raw[name],bins,inputs,ocean)
                    altered=raw[name].copy();altered[outside]=12345.0
                    alt,ac=c.fit_climatology(altered,bins,inputs,ocean)
                    np.testing.assert_array_equal(alt,clim);np.testing.assert_array_equal(ac,count)
                    original_a=c.anomaly(raw[name][inputs],bins[inputs],clim,ocean)
                    altered_a=c.anomaly(altered[inputs],bins[inputs],alt,ocean)
                    np.testing.assert_array_equal(original_a,altered_a)
                    assert p._mean_std(original_a[:,ocean])==p._mean_std(altered_a[:,ocean])
                    mutations.append(name);del altered
                    clims[name]=clim;counts[name]=count
                    archive[name]=clim;archive[name+'_valid_counts']=count
                fname=f'climatology_ratio_{ratio}_seed{seed}.npz'
                np.savez_compressed(OUT/fname,**archive)
                rebased=c.apply_to_base(base,raw,bins,clims)
                new_ds=ds.assign({n:(('time','latitude','longitude'),rebased.dynamic_arrays[n]) for n in c.ANOMALIES})
                qcs=[quality(n,raw[n],rebased.dynamic_arrays[n],clims[n],counts[n],inputs,ocean,m7['statistics'][n][1]) for n in c.ANOMALIES]
                qc_pass=qc_pass and all(q['status']=='PASS' for q in qcs)
                report={'ratio':ratio,'seed':seed,'input_unique_days':len(inputs),
                    'calendar_counts':day_counts.tolist(),'empty_bins':(np.flatnonzero(day_counts==0)+1).tolist(),
                    'single_observation_bins':int((day_counts==1).sum()),
                    'selected_dates_in_single_observation_bins':int(day_counts[day_counts==1].sum()),
                    'feb29_count':int(day_counts[59]),'feb29_method':'selected mean' if day_counts[59] else 'periodic linear interpolation',
                    'mutation_test':'PASS: every unselected raw date replaced by 12345; clims, selected anomalies and input stats identical',
                    'variables':qcs}
                group_reports.append(report)
                for lead in (7,14):
                    old=json.loads((V1/f'ratio_{ratio}_lead{lead}_seed{seed}.json').read_text())
                    np.testing.assert_array_equal(old['input_time_indices'],inputs)
                    np.testing.assert_array_equal(old['train_start_indices'],starts)
                    original=json.loads(Path(old['subset_path']).read_text())
                    np.testing.assert_array_equal(starts,original[str(ratio)]['start_indices'])
                    fresh=p.fit_subset_stats(new_ds,starts,30,lead)
                    for name,pair in fresh.items():
                        if name not in c.ANOMALIES:np.testing.assert_array_equal(pair,old['statistics'][name])
                    rhs=p.compute_subset_rhs_scale(base,starts,lead)
                    np.testing.assert_array_equal(rhs,old['physics_rhs_scale'])
                    m=copy.deepcopy(old)
                    m.update(schema=c.SCHEMA,statistics=fresh,source_sha256=source_hash,
                        raw_thetao_path=str(RAW3D),raw_thetao_sha256=raw_hash,raw_thetao_depth_indices=c.DEPTH_INDICES,
                        climatology_file=fname,climatology_sha256=p.sha256_file(OUT/fname),climatology_rule=c.RULE,
                        physics_rhs_scale=rhs,statistics_basis='subset-only anomaly climatology + selected unique history stats; selected target stats',
                        upstream_features='SST and six thetao input anomalies rebuilt from raw temperatures; v1 anomalies never used as input',
                        validation='See validation_report.json; construction checks completed before training')
                    name=f'ratio_{ratio}_lead{lead}_seed{seed}.json';path=OUT/name
                    save(path,m)
                    # Read the serialized climatology again and construct real v2 datasets.
                    loaded=c.read_climatology(m,path)
                    bundle=c.bundle_from_v2(ds,m,base,raw,loaded)
                    for dataset in (bundle.train,bundle.val,bundle.test):
                        assert dataset.base.stats is bundle.stats
                        sample=dataset[0];a=int(sample['start_index']);idx=bundle.graph.ocean_flat_indices
                        for ch,n in enumerate(p.BASE_DYNAMIC_VARS):
                            values=rebased.dynamic_arrays[n][a:a+30]
                            mu,sd=fresh[n];expected=((values-mu)/sd).reshape(30,-1)[:,idx].T
                            np.testing.assert_allclose(sample['x_nodes'][:,:,ch],expected,rtol=1e-6,atol=1e-6)
                    check=verify_entrypoint(code,path,m,bundle);runtime.append({'manifest':name,'result':check})
                    splits=make_time_splits(ds.sizes['time'],30,lead)
                    np.testing.assert_array_equal(bundle.val_indices,splits.val);np.testing.assert_array_equal(bundle.test_indices,splits.test)
                    rows.append({'manifest':name,'sha256':p.sha256_file(path),'ratio':ratio,'seed':seed,'lead':lead,
                        'train_samples':len(starts),'input_unique_days':len(inputs),'target_mean':fresh['target'][0],
                        'target_std':fresh['target'][1],'physics_rhs_scale':rhs})
                    del bundle
                    print(f'Verified {name}',flush=True)
                del rebased,new_ds
    # Exercise the public v1/v2 dispatch, hashes and full disk loading once.
    b,m=p.load_subset_bundle(OUT/'ratio_10_lead7_seed42.json');assert len(b.train)==376
    assert m['schema']==c.SCHEMA;b.ds.close();del b
    assert digest_tree(V1)==oldhash
    assert p.sha256_file(ROOT/'src/mhw/mhw_skill.py')==evaluation_hash
    save(OUT/'climatology_qc.json',{'calendar_rule':c.RULE,'qc_gates':{
        'finite_ocean':True,'land_zero':True,'positive_train_std_min':1e-6,'convex_climatology':True,
        'train_abs_zmax_upper':20,'all_dates_abs_zmax_upper':30,'all_to_train_std_ratio_bounds':[.1,10]},'groups':group_reports})
    save(OUT/'runtime_verification.json',{'training_executed':False,'gpu_used':False,'checks':runtime,
        'public_disk_loader':'PASS','v1_files_preserved':'PASS','MHW_evaluator_sha256':evaluation_hash})
    save(OUT/'execution_plan.json',execution_plan(rows))
    gates={'subset_only_invariance':'PASS','baseline_pinn_equal':'PASS',
        'coverage_missing_extremes':'PASS' if qc_pass else 'REVIEW_REQUIRED','actual_v2_entrypoint_no_full_override':'PASS'}
    save(OUT/'validation_report.json',{'timestamp_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'gates':gates,'training_executed':False,'gpu_used':False,'artifact_count':len(rows),
        'target_stats_unchanged':'24/24 PASS','rhs_unchanged':'24/24 PASS','v1_unchanged':True})
    save(OUT/'index.json',{'schema':c.SCHEMA,'status':'validated' if qc_pass else 'review_required',
        'artifact_count':len(rows),'training_executed':False,'datasets':rows})
    log('- scripts/build_subset_preprocessing_v2.py로24개 manifest와12개 climatology를 별도 생성했다.\n'
        '- 선택되지 않은 raw 날짜 변경 불변성, target/RHS 불변성, 실제 실행 진입점/CPU loss 및 Baseline/PINN 동일성을 검증했다.\n'
        f'- QC 결과: {gates}. v1 파일/원본 subset 보존 및 MHW 평가 함수 해시 유지. 본 학습 미실행;48개 비교+24개 ablation은 계획만 저장했다.')
    print(json.dumps(gates),flush=True)

if __name__=='__main__':main()
