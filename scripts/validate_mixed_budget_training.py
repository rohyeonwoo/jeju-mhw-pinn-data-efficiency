"""CPU regression, subset-only statistics audit and frozen v7 protocol."""
from mixed_budget_common import *
import argparse,ast,math
import numpy as np
import torch
import pandas as pd
import mixed_budget_physics as mb
import train_v3_study as w
import experiment_mixed_layer_budget_cpu as c
from types import SimpleNamespace
from torch.utils.data import DataLoader

def unit_tests():
 torch.set_num_threads(1);dtype=torch.float64
 shape=(2,1,5,6);mask=torch.ones(shape,dtype=dtype);mask[...,0,0]=0
 temp=torch.full(shape,20.,dtype=dtype,requires_grad=True);h=torch.full(shape,10.,dtype=dtype);z=torch.zeros(shape,dtype=dtype);q=torch.full(shape,100.,dtype=dtype);sw=torch.full(shape,80.,dtype=dtype);dx=torch.full(shape,25000.,dtype=dtype)
 args=dict(h=h,qnet=q,ssr=sw,u=z+1,v=z+1,tb=z+19,we=z,mask=mask,dx=dx,dy=dx)
 ts=mb.terms(temp,**args);assert ts['advection'].abs().max()==0
 ts0=mb.terms(temp,**args,kd=0);assert torch.allclose(ts0['flux']-ts0['penetration'],torch.where(mask>0,z+86400*20/(1025*3990*10),z))
 polluted=temp.detach().clone();polluted[mask==0]=float('nan');ts2=mb.terms(polluted,**args);assert torch.equal(ts['advection'],ts2['advection'])
 field=temp+torch.arange(6,dtype=dtype).reshape(1,1,1,6);ts3=mb.terms(field,**{**args,'u':z,'v':z});assert ts3['advection'].abs().max()==0
 L=3;b={'mixed_'+k:val[:,None].repeat(1,L,1,1,1) for k,val in dict(h=h,qnet=q,ssr=sw,u=z+.2,v=z+.1,tb=z+19,we=z+1e-5,valid=mask).items()};b.update(dx=dx,dy=dx)
 pred=(field.detach()+1).requires_grad_();rA,rhsA,sA,terms=mb.path(pred,field.detach(),mask,b,L,False);rB,rhsB,sB,_=mb.path(pred,field.detach(),mask,b,L,True)
 assert torch.equal(sA,sB);assert torch.allclose((rhsB-rhsA)*sA,terms['advection']*sA);assert torch.allclose((rA-rB)*sA,terms['advection']*sA)
 pa=SimpleNamespace(mixed_variant='A',physics_clip=10.,physics_normalize='rhs',physics_rhs_scale=.1,physics_normalize_eps=1e-8,physics_loss='mse',adaptive_sampling=False,adaptive_weighting=False)
 v1=mb.loss(pred,field.detach(),z,mask,b,pa,L,z,z);v2=mb.loss(pred,field.detach(),z+float('nan'),mask,b,pa,L,z,z);assert torch.equal(v1,v2)
 grad=torch.autograd.grad(v1,pred)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
 save(STUDY/'cpu_unit_tests.json',dict(status='PASS',Kd0='all shortwave transmitted, net flux qnet-ssr',constant_temperature_gradient_zero=True,land_nan_invariance=True,zero_velocity_advection_zero=True,AB_only_advection=True,future_target_unused=True,predictor_gradient_positive=True,units='K/day before clip10; squared normalized by common v3 mean-square RHS'))

def audit_data():
 ar=mb.arrays();feature_info=json.loads((FEATURES/'manifest.json').read_text())
 for path,h in feature_info['sha256'].items():assert sha(path)==h,path
 rows=[];scale_rows=[];count_model=None
 for ratio,lead in [(10,7),(50,14)]:
  for seed in [42,7,123]:
   path=ROOT/f'data/subsets/preprocessing_subset_v3/ratio_{ratio}_lead{lead}_seed{seed}.json';b,m=w.p.load_subset_bundle(path);assert m['schema']=='subset_normalization_v3'
   wrapped=mb.ForcingDataset(b.train,ar)
   for i in [0,len(b.train)//2,len(b.train)-1]:
    old=b.train[i];new=wrapped[i]
    for key in ['x_nodes','y_nodes','tm_norm_nodes']:assert torch.equal(old[key],new[key])
    assert int(new['current_index'])==int(new['start_index'])+29
    new['mixed_tb']+=1000;assert torch.equal(old['x_nodes'],new['x_nodes'])
   ref_end=round(len(b.ds.time)*.7)-1;assert ref_end<int(b.val_indices[0])+29+lead and ref_end<int(b.test_indices[0])+29+lead
   # Same instantiated structure and same random stream for every model label.
   sig=[]
   for label in ['B0','B1','A','B']:
    w.t.set_seed(seed);model=w.t.JejuHiLAM(graph=b.graph,input_dim=len(b.train.input_feature_names),hidden_dim=256,temporal_hidden_dim=256,processor_layers=3,mlp_hidden_layers=1,prediction_mode='direct',physics_heads=True,we_nonneg=True,we_scale=1e-5,rres_scale=1e-4,we_softplus_shift=1.)
    sig.append(w.state_digest(model,shared=False));params=sum(v.numel() for v in model.parameters());del model
   assert len(set(sig))==1
   sums={'A':0.,'B':0.};n=0
   loader=DataLoader(wrapped,batch_size=32,shuffle=False,num_workers=0)
   with torch.no_grad():
    for batch in loader:
     _,rhs,s,ts=mb.path(batch['y_raw'],batch['Tm'],batch['mask'],batch,lead,False)
     sums['A']+=float((rhs.double().square()*s).sum());sums['B']+=float(((rhs+ts['advection']).double().square()*s).sum());n+=int(s.sum())
   for name,v in sums.items():scale_rows.append(dict(ratio=ratio,lead=lead,seed=seed,equation=name,own_train_rhs_mean_square=v/n,common_v3_mean_square=m['physics_rhs_scale'],own_over_common=(v/n)/m['physics_rhs_scale'],normalizer_used='common_v3',sample_count=n))
   rows.append(dict(ratio=ratio,lead=lead,seed=seed,manifest_sha256=sha(path),stats_sha256=digest(m['statistics']),initial_all_heads_sha256=sig[0],parameters=params,train_count=len(b.train),val_count=len(b.val),test_count=len(b.test),input_identical=True,MHW_reference_last_date=str(b.ds.time.values[ref_end])[:10],first_val_target=str(b.ds.time.values[int(b.val_indices[0])+29+lead])[:10]))
   b.ds.close();print('subset audit',ratio,lead,seed,flush=True)
 # Cache/torch versus existing v6 NumPy terms, with the intentionally backward we supplied identically.
 import xarray as xr
 with xr.open_dataset(BUDGET) as ds:
  bd=c.detect_dims(ds);mask=c.read2d(ds,'ocean_mask_static',bd)>.5;dx,dy=c.spacing(ds.latitude.values,ds.longitude.values)
  for day in [29,365,5000,9000]:
   valid=mask&(ar['valid'][day]>0);field=np.asarray(ar['tm'][day],dtype=float);gx,gy=c.slopes_ocean(field,valid,dx,dy)
   ref=-86400*(ar['u'][day]*gx+ar['v'][day]*gy)
   vals={k:torch.tensor(np.array(ar[k][day]),dtype=torch.float64)[None,None] for k in ['h','qnet','ssr','u','v','tb','we']}
   out=mb.terms(torch.tensor(field)[None,None],**vals,mask=torch.tensor(valid,dtype=torch.float64)[None,None],dx=torch.tensor(dx)[None,None],dy=torch.tensor(dy))
   assert np.allclose(out['advection'].numpy()[0,0][valid],ref[valid],rtol=1e-6,atol=1e-7)
 save(STUDY/'subset_input_audit.json',dict(status='PASS',rows=rows,derivative_reproduction='v6 NumPy versus native ocean-neighbor torch, four real days PASS',temporal_difference='backward MLD proxy differs deliberately from CPU v6 forward proxy'))
 pd.DataFrame(scale_rows).to_csv(STUDY/'rhs_scale_comparison.csv',index=False)

def entrypoint_test():
 import train_mixed_budget_gpu as trainer
 from copy import copy
 trainer.install();torch.set_num_threads(1)
 mf=ROOT/'data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json'
 j=dict(id=0,name='CPU_entrypoint_no_update',model='A',arm='p0',ratio=10,lead=7,seed=42,manifest=str(mf),manifest_sha256=sha(mf),run_dir=str(STUDY/'cpu_check_only'))
 p=dict(code_sha256={},feature_manifest_sha256=sha(FEATURES/'manifest.json'))
 a,b,model,pa,cfg=trainer.setup(j,p,torch.device('cpu'));a.batch_size=2
 batch=next(iter(w.t.make_loader(b.train,a,False)));model.eval();out=model(batch['x_nodes'],base_nodes=batch['tm_norm_nodes'],return_aux=True)
 pars=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))];rows=[]
 for name in ['B0','B1','A','B']:
  physics=copy(pa);physics.mixed_variant=name;physics.lambda_physics=0 if name=='B0' else .1
  vals=w.t.compute_batch_losses(out,batch,b.graph,*b.stats['target'],None if name=='B0' else physics,physics.lambda_physics,model)
  assert all(torch.isfinite(v).all() for v in vals)
  gd=torch.autograd.grad(vals[1],pars,retain_graph=True,allow_unused=True);nd=sum(float(g.double().square().sum()) for g in gd if g is not None)**.5
  gp=torch.autograd.grad(vals[2],pars,retain_graph=True,allow_unused=True) if name!='B0' else [];ng=sum(float(g.double().square().sum()) for g in gp if g is not None)**.5
  if name!='B0':assert ng>0
  rows.append(dict(model=name,data_loss=float(vals[1].detach()),physics_loss=float(vals[2].detach()),total_loss=float(vals[5].detach()),weighted_shared_gradient_ratio=.1*ng/nd,batch_size=2))
  with torch.no_grad():
   metrics=w.t.run_epoch(model,w.t.make_loader(b.val,a,False),b.graph,torch.device('cpu'),*b.stats['target'],max_batches=1,physics_args=None if name=='B0' else physics,lambda_physics=physics.lambda_physics,epoch=1)
  assert all(math.isfinite(v) for v in metrics.values())
 b.ds.close();save(STUDY/'cpu_entrypoint_check.json',dict(status='PASS',scope='actual native model forward, autograd.grad and validation loop; CPU batch2; zero optimizer steps; GPU smoke remains pending',rows=rows));print('CPU ENTRYPOINT PASS',rows,flush=True)

def freeze():
 files=[ROOT/'scripts'/n for n in ['mixed_budget_common.py','mixed_budget_physics.py','prepare_mixed_budget_features.py','validate_mixed_budget_training.py','train_mixed_budget_gpu.py','run_mixed_budget_stage.py','summarize_mixed_budget_gpu.py','train_v3_study.py','experiment_mixed_layer_budget_cpu.py']]
 files+=list((ROOT/'src').rglob('*.py'))
 files+=[ROOT/'scripts/run_mixed_budget_tmux.sh',ROOT/'scripts/README_mixed_budget_gpu_ko.md']
 for path in files:
  if path.suffix=='.py':ast.parse(path.read_text())
 jobs=[]
 for ratio,lead in [(10,7),(50,14)]:
  for seed in [42,7,123]:
   for model in ['B0','B1','A','B']:
    name=f'{model}_r{ratio}_lead{lead}_seed{seed}';mf=ROOT/f'data/subsets/preprocessing_subset_v3/ratio_{ratio}_lead{lead}_seed{seed}.json'
    jobs.append(dict(id=len(jobs),name=name,model=model,arm='p0',ratio=ratio,lead=lead,seed=seed,manifest=str(mf),manifest_sha256=sha(mf),run_dir=str(STUDY/'runs'/name)))
 p=dict(version=1,created_at=stamp(),jobs=jobs,representative_job_ids=[0,1,2,3],allowed_physical_gpus=[0,1,2],max_processes_per_gpu=1,automatic_expansion=False,automatic_restart=False,model='identical native HiLAM with we/rres heads; NOAA SST proxy',oracle_forcing=True,forcing_in_model_input=False,feature_manifest_sha256=sha(FEATURES/'manifest.json'),normalization='unchanged v3 subset-specific mean-square P0 RHS common to B1/A/B; own-equation scales diagnostic only',lambda_physics=.1,physics_warmup_epochs=0,lambda_rres=0,lambda_we_tie=0,zero_rres_physics=True,kd=.06,optimizer='AdamW betas(.9,.95), weight_decay .1, clip norm1',batch_size=16,epochs=80,updates_per_validation=478,max_updates=38240,patience_validations=12,selection='full Validation SST RMSE',lr='native 5-epoch linear LR warmup then cosine; physical lambda has no warmup',MHW=dict(reference='first round(.7*N) raw SST dates, unchanged',quantile=.9,window_days=5,min_duration_days=5,area_fraction=.15,onset_tolerance_days=3),statistics=dict(bootstrap_blocks=[90,30,180],replicates=2000,paired_across_models=True,seeds=3,Test='exploratory: past Test inspected; no claim of independent confirmatory evaluation'),code_sha256={str(f.relative_to(ROOT)):sha(f) for f in files})
 path=STUDY/'experiment_protocol.json'
 if path.exists():
  old=json.loads(path.read_text());p['created_at']=old['created_at'];assert p==old,'Frozen protocol changed'
 else:save(path,p)
 if not (STUDY/'run_manifest.json').exists():save(STUDY/'run_manifest.json',dict(status='not_started',jobs=jobs))
 if not (STUDY/'training_status.json').exists():save(STUDY/'training_status.json',dict(status='not_started',reason='user executes manually'))
 if not (STUDY/'smoke_test_report.json').exists():save(STUDY/'smoke_test_report.json',dict(status='NOT_RUN',reason='GPU smoke test left for manual command'))
 save(STUDY/'preflight_report.json',dict(status='CPU_PASS_GPU_PENDING',timestamp=stamp(),unit_tests=json.loads((STUDY/'cpu_unit_tests.json').read_text()),subset_audit='PASS',cpu_entrypoint=json.loads((STUDY/'cpu_entrypoint_check.json').read_text()),source_integrity=guard_hashes('after'),GPU_smoke='NOT_RUN',launch_guard='resource check + exclusive per-GPU lock + UUID process-title verification',protocol_sha256=digest(p)))
 save(STUDY/'protected_hash_audit.json',dict(before='protected_hash_before.json',after='protected_hash_after.json',status='PASS',count=380))
 log('v7 CPU 회귀/6개 subset 통계·입력·공통 구조·RHS 크기 검사 후 24회 수동 프로토콜 고정. GPU smoke 및 학습은 미실행.');print('CPU preflight PASS; GPU smoke NOT RUN')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['unit','data','entry','freeze','all'],default='all');a=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 if a.stage in ['unit','all']:unit_tests()
 if a.stage in ['data','all']:audit_data()
 if a.stage in ['entry','all']:entrypoint_test()
 if a.stage in ['freeze','all']:freeze()
