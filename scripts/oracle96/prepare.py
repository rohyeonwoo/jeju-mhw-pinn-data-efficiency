"""CPU preflight and immutable 96-condition protocol; never starts a GPU worker."""
from mixed_budget_common import *
import ast,copy
import numpy as np
import torch
import xarray as xr
import pandas as pd
import mixed_budget_physics as mb
import train_mixed_budget_gpu as worker
import train_v3_study as w

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 assert not (STUDY/'experiment_protocol.json').exists(),'Existing frozen protocol is preserved'
 STUDY.mkdir(parents=True,exist_ok=True)
 # Preserve all previously protected inputs plus the previous v7 experiment tree.
 old=ROOT/'outputs/v3_cpu_diagnostics_20261011/protected_input_hashes.json';protected=json.loads(old.read_text())
 for path,h in protected.items():assert sha(path)==h,path
 for f in (ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011').rglob('*'):
  if f.is_file() and 'runtime' not in f.parts:protected[str(f)]=sha(f)
 save(STUDY/'protected_input_hashes.json',protected);guard_hashes('before')
 # Existing tested physical equations are imported unchanged; test outputs belong to v8.
 import validate_mixed_budget_training as unit
 unit.unit_tests()
 assert mb.KD==.06 and mb.RHO==1025 and mb.CP==3990 and mb.DAY==86400
 fm=json.loads((FEATURES/'manifest.json').read_text())
 for path,h in fm['sha256'].items():assert sha(path)==h,path
 jobs=[];boundaries=[]
 conditions=[(10,7)]+[(r,L) for r in [10,25,50,100] for L in [7,14] if (r,L)!=(10,7)]
 with xr.open_dataset(BUDGET) as ds:
  times=ds.time.values;N=len(times)
  for ratio,L in conditions:
   for seed in [42,7,123]:
    mf=ROOT/f'data/subsets/preprocessing_subset_v3/ratio_{ratio}_lead{L}_seed{seed}.json';m=json.loads(mf.read_text());s=np.array(m['train_start_indices']);v=np.array(m['val_start_indices']);te=np.array(m['test_start_indices']);spl=w.p.make_time_splits(N,30,L)
    assert np.array_equal(v,spl.val) and np.array_equal(te,spl.test) and np.isin(s,spl.train).all()
    end=int(v.min()+29+L-1);targets=s+29+L;forcing=s[:,None]+29+np.arange(L)
    assert np.all(targets<=end) and np.all(forcing<=end)
    assert np.array_equal(np.unique(s[:,None]+np.arange(30)),m['input_time_indices']) and np.array_equal(targets,m['target_time_indices'])
    orig=json.loads(Path(m['subset_path']).read_text());assert s.tolist()==orig[str(ratio)]['start_indices']
    for lower in [r for r in [10,25,50,100] if r<ratio]:assert set(orig[str(lower)]['start_indices'])<=set(s)
    clim=mf.parent/m['climatology_file'];assert sha(clim)==m['climatology_sha256']
    with np.load(clim) as z:assert np.array_equal(z['input_time_indices'],m['input_time_indices'])
    assert np.isfinite(m['physics_rhs_scale']) and m['physics_rhs_scale']>0
    boundaries.append(dict(ratio=ratio,lead=L,seed=seed,n_train=len(s),train_calendar_end=str(times[end])[:10],latest_target=str(times[targets.max()])[:10],latest_forcing=str(times[forcing.max()])[:10],first_val_target=str(times[v.min()+29+L])[:10],first_test_target=str(times[te.min()+29+L])[:10],excluded_cross_boundary=0,status='PASS',rhs_scale=m['physics_rhs_scale'],stats_sha256=digest(m['statistics'])))
    for model,label in [('B0','Baseline'),('B1','P0'),('A','NoAdv'),('B','Adv')]:
     name=f'{label}_r{ratio}_lead{L}_seed{seed}'
     jobs.append(dict(id=len(jobs),name=name,model=model,label=label,arm='p0',ratio=ratio,lead=L,seed=seed,manifest=str(mf),manifest_sha256=sha(mf),run_dir=str(STUDY/'runs'/name)))
  # Unchanged MHW evaluation reference; fitting period predates every held-out target.
  from summarize_mixed_budget_gpu import prepare_threshold
  prepare_threshold(ds)
 save(STUDY/'boundary_audit.json',boundaries);pd.DataFrame(boundaries).to_csv(STUDY/'boundary_audit.csv',index=False)
 base=json.loads((ROOT/'outputs/v7_gpu_mixed_budget_comparison_20261011/experiment_protocol.json').read_text())
 files={str(ROOT/name):h for name,h in base['code_sha256'].items()}
 for path,h in files.items():assert sha(path)==h,path
 for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text());files[str(f)]=sha(f)
 files[str(Path(__file__).parent/'README_ko.md')]=sha(Path(__file__).parent/'README_ko.md')
 p={**base,'version':2,'allowed_physical_gpus':[7,8,9],'created_at':stamp(),'jobs':jobs,'automatic_expansion':True,'training_policy':'Training-only Oracle Physics Supervision','early_stopping':False,'patience_validations':81,'fixed_updates':38240,'selection':'Validation SST RMSE, physics/forcing absent in validation selection','claim':'SST-label efficiency; not equal total-information efficiency','reuse_previous_runs':False,'expansion_gates':['CPU','smoke','representative four + trained Validation forcing invariance','remaining10lead7seeds','remaining84','all96 trained','Test then CPU report'],'code_sha256':{str(Path(f).relative_to(ROOT)):h for f,h in files.items()}}
 save(STUDY/'experiment_protocol.json',p)
 # Actual native entrypoint, no optimizer, on both leads; all four losses share inputs/weights.
 worker.install();torch.set_num_threads(1);entry=[]
 for L in [7,14]:
  j=next(j for j in jobs if j['ratio']==10 and j['lead']==L and j['seed']==42 and j['model']=='A')
  a,b,model,pa,cfg=worker.setup(j,p,torch.device('cpu'));a.batch_size=2;batch=next(iter(w.t.make_loader(b.train,a,False)));model.eval();out=model(batch['x_nodes'],base_nodes=batch['tm_norm_nodes'],return_aux=True)
  pars=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))]
  for arm in ['B0','B1','A','B']:
   phys=copy.copy(pa);phys.mixed_variant=arm;phys.lambda_physics=0 if arm=='B0' else .1
   vals=w.t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],None if arm=='B0' else phys,phys.lambda_physics,model)
   assert all(torch.isfinite(v).all() for v in vals)
   gd=torch.autograd.grad(vals[1],pars,retain_graph=True,allow_unused=True);nd=sum(float(g.double().square().sum()) for g in gd if g is not None)**.5
   gp=[] if arm=='B0' else torch.autograd.grad(vals[2],pars,retain_graph=True,allow_unused=True);ng=sum(float(g.double().square().sum()) for g in gp if g is not None)**.5
   assert arm=='B0' or ng>0
   vb=next(iter(w.t.make_loader(b.val,a,False)));assert not any(k.startswith('mixed_') or k.endswith('_seq') for k in vb)
   with torch.no_grad():vm=w.t.run_epoch(model,[vb],b.graph,torch.device('cpu'),*b.stats['target'],physics_args=None,lambda_physics=0.)
   assert vm['physics_loss']==0
   entry.append(dict(lead=L,model=arm,data_loss=float(vals[1].detach()),physics_loss=float(vals[2].detach()),gradient_ratio=.1*ng/nd,val_no_forcing=True,status='PASS'))
  b.ds.close();print('CPU native entry PASS lead',L,flush=True)
 save(STUDY/'cpu_entrypoint_check.json',entry)
 verify_code(p);guard_hashes('after_cpu')
 save(STUDY/'preflight_report.json',dict(status='PASS',protocol_sha256=digest(p),conditions=96,manifests=24,boundaries='PASS',unit_tests='PASS',entrypoint='PASS',GPU_smoke='PENDING',timestamp=stamp()))
 save(STUDY/'training_status.json',dict(status='ready_for_smoke',timestamp=stamp()))
 log('scripts/oracle96/ 별도 96조건 실행기 생성. 원본/v7 보존, Train-only oracle·검증 forcing 제거·동일38240 updates·λ=.1·Kd=.06 고정. CPU 단위/24개 manifest 경계/양리드 실제 loss·gradient 검사 PASS. GPU는 7/8/9만, 1프로세스/GPU. 순차 gate 후 확대하도록 사용자 승인 반영.')
 print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
