from adaptive_common import *
import xarray as xr,inspect
from mhw import datasets
from mhw.models.pinn_model_alpha import compute_residual
from mhw.ocean_stencil import ocean_neighbor_gradients

def main():
 OUT.mkdir(exist_ok=False)
 for name in ['figures','runs','logs','diagnostics','preflight','runtime']:(OUT/name).mkdir()
 old=json.loads((OLD/'protocol.json').read_text());assert json.loads((OLD/'final_integrity.json').read_text())['status']=='PASS'
 jobs=[];reuse=[]
 for ratio,lead in [(50,14),(10,7)]:
  for seed in [42,7,123]:
   pair=[next(j for j in old['jobs'] if (j['ratio'],j['lead'],j['seed'],j['arm'])==(ratio,lead,seed,arm)) for arm in ['baseline','physics_only']]
   configs=[json.loads((Path(j['run_dir'])/'config.json').read_text()) for j in pair]
   for k in ['parameters','trainable_parameters','shared_initial_sha256','stats','rhs_scale','applied_rhs_scale','manifest_sha256','split_sha256','tensor_fingerprints']:assert configs[0][k]==configs[1][k],k
   histories=[json.loads((Path(j['run_dir'])/'history.json').read_text()) for j in pair]
   assert all(x['batch_order_sha256']==y['batch_order_sha256'] for x,y in zip(*histories))
   for j,c in zip(pair,configs):
    d=Path(j['run_dir']);status=json.loads((d/'status.json').read_text());assert status['status']=='trained'
    assert c['settings']['lambda_physics']==(0 if j['arm']=='baseline' else .01) and c['settings']['lambda_rres']==0
    for n in ['best.pt','last.pt']:
     cp=torch.load(d/n,map_location='cpu',weights_only=False);assert cp['config_sha256']==c['config_sha256'] and all(torch.isfinite(v).all() for v in cp['model_state'].values())
     if n=='last.pt':assert cp['history']==json.loads((d/'history.json').read_text()) and cp['stream']['total']==status['steps']
     del cp
    reuse.append(dict(ratio=ratio,lead=lead,seed=seed,arm=j['arm'],run_dir=str(d),best_sha256=sha(d/'best.pt'),last_sha256=sha(d/'last.pt'),config_sha256=c['config_sha256'],status='PASS',validation_rmse=status['best_val_rmse'],best_epoch=status['best_epoch']))
   j=dict(pair[1]);j.update(id=len(jobs),name=f'ratio_{ratio}/lead{lead}_seed{seed}/adaptive',run_dir=str(OUT/'runs'/f'ratio_{ratio}/lead{lead}_seed{seed}/adaptive'),reuse=False,reference_fixed_run=pair[1]['run_dir'],reference_baseline_run=pair[0]['run_dir']);j.pop('evaluation_dir',None);j.pop('reuse_evidence',None);jobs.append(j)
 code=dict(old['code_sha256'])
 for n,hsh in code.items():assert sha(ROOT/n)==hsh
 # Freeze these new files after all helpers exist, before any diagnostics/training.
 for n in ['adaptive_common.py','diagnose_adaptive_pilot.py','preflight_adaptive_pilot.py','train_adaptive_pilot.py']:
  assert (ROOT/'scripts'/n).exists();code['scripts/'+n]=sha(ROOT/'scripts'/n)
 pr=dict(created_at=stamp(),jobs=jobs,code_sha256=code,adaptive=SPEC,resources=dict(gpu_ids=[1,2,3],max_processes_per_gpu=4,planned_training_per_gpu=2,reserve_mib=4096,memory_factor=1.2),validation_only=True,no_new_test=True,diagnosis_scope='Selected training windows only; native linear endpoint path plus observed daily path; learned entrainment explicitly distinguished',reuse=reuse,original_protocol_sha256=sha(OLD/'protocol.json'))
 save(OUT/'protocol.json',pr);save(OUT/'reuse_audit.json',dict(status='PASS',rows=reuse));csvsave(OUT/'run_manifest.csv',[dict(job=j['id'],name=j['name'],status='pending_diagnosis',run_dir=j['run_dir']) for j in jobs])
 (OUT/'reuse_audit.md').write_text('# Reuse audit PASS\n\n12 existing Baseline/Fixed checkpoints strict-loaded; model/config/stats/masks/manifests/splits/shared initialization/batch chains/native478-step validation and12-check patience verified. New training limited to6 Adaptive runs. best/last SHA and exact original paths: reuse_audit.json. Prior initial gradient logs exist; no serialized initialization checkpoint, so initialization-state diagnostics will be N/A rather than invented.\n')
 ds=xr.open_dataset(json.loads(Path(jobs[0]['manifest']).read_text())['data_path'])
 attrs={k:dict(ds[k].attrs) for k in ['sst_mean','qnet','mlotst','u_ml','v_ml']};save(OUT/'source_units.json',attrs);ds.close()
 audit='''# P0 equation and implementation audit\n\nT_j = T_0 + j/h (T_hat_h - T_0); d_j = (T_{j+1}-T_j)/86400 (K/s).\nF_j = Q_j/(1025*3990*max(H_j,5)) - u_j D_x T_j - v_j D_y T_j - w_e * 1K/max(H_j,5).\nr_j = 86400*(d_j-F_j) (K/day); Rres=0. Lphys = masked_mean(clip(r_j,-10,10)^2)/S_RHS.\n\nT in source is degrees Celsius (see source_units.json), with differences numerically equal to kelvin. Qnet is net heat input in W/m², H metres, velocity m/s; gradients K/m. dx/dy use Earth radius and radian latitude/longitude differences. Each RHS term K/s is multiplied by86400 before residual clipping. SST input is observational/merged product; forcing is reanalysis, not independent in-situ truth. Exact metadata are preserved in source_units.json.\n\nEntrainment is NOT an observed flux: we = 1e-5 * softplus(we_head -1), interpreted m/s, one predicted spatial map shared across all h daily steps. DeltaT is prescribed1K; dataset Td_glorys is not used by this P0 loss. Native positive we only cools under this assumed deltaT. Rres head exists but zero_rres_physics=True and lambda_rres=lambda_we_tie=0, no auxiliary penalty gradient. Vertical advection, diffusion, variable subsurface contrast, and other unresolved processes are absent. Lack of closure by itself is not a coding error.\n\nPerfect forcing uses reanalysis dates current ... current+h-1 during the physics loss. The endpoint target is observed; daily intermediate states in the native loss are linearly interpolated, not observed daily states. Primary observed-endpoint diagnostic must retain that linear path; actual daily observations are a separately labelled comparator using the same native one-day function.\n\nOcean-neighbor stencil: two ocean neighbors central difference; one ocean neighbor one-sided; no ocean neighbors zero derivative and unsupported cell excluded. Land zeroed before arithmetic, independent of NaN/Inf land fill. SST data mask661 cells; physics support660 cells with a neighbor on each axis.\n\nS_RHS = mean_training_samples,steps,physics_cells(F_observed_linear,we=0²), units(K/day)²; it is not std or RMS. Manifest fixes this subset-only quantity, residual divided by sqrt(S_RHS). Clipping precedes normalization and MSE. Loss is dimensionless. Data loss is normalized node MSE with manifest target mean/std. Same normalization is retained for Fixed and Adaptive.\n\nNative warmup f(epoch)=min(1,epoch/5), applied once. Fixed base lambda=.01; effective .002 ... .01. AdamW betas(.9,.95), weight_decay .1; LR2e-5 to2e-4 native5-epoch warmup; clip_grad_norm_(all parameters,1). fp32 (no GradScaler). Validation478updates; max80*478=38240; patience12; checkpoint min Validation RMSE. Adaptive estimates unweighted shared-predictor gradient norms only, excludes both heads, then applies the identical warmup once.\n\nNo fatal inconsistency identified in this static audit; native-function numerical agreement, finite data, gradients and resume remain runtime gates.\n\n## Code locations\n'''
 for obj in [h.daily_rollout_physics_loss,h._linear_rhs_per_day,h.compute_fixed_rhs_scale,h.compute_physics_loss,h.physics_lambda_factor,h.physics_budget_diagnostics,compute_residual,ocean_neighbor_gradients,datasets.compute_grid_spacing_m,t.compute_batch_losses,t.run_epoch,t.set_epoch_lr,a.worker]:
  file=Path(inspect.getsourcefile(obj));source,line=inspect.getsourcelines(obj);audit+=f'- {file.relative_to(ROOT)}:{line}-{line+len(source)-1} — {obj.__name__}\n'
 audit+='- src/mhw/datasets.py:246-354 — raw SST/forcing, sequence dates and input construction\n- src/mhw/models/jeju_hilam.py:341-344 — head output scales\n- src/mhw/subset_preprocessing.py:187-207 — mandatory v3 physics/stencil/RHS application\n'
 (OUT/'physics_equation_audit.md').write_text(audit)
 log('Adaptive pilot initialized: new outputs/v3_adaptive_physics_pilot_20261011; two cases50%L14/10%L7,3seeds;12checkpoint reuse strict PASS; source/physics audit recorded. New files only; native Fixed unchanged. GPU1/2/3 per latest attachment, max4 but plan2 training each subject to smoke memory. No Test/8-condition expansion.')
 print('PREPARE PASS',flush=True)
if __name__=='__main__':main()
