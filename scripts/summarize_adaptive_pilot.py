"""Phase gates and Validation-only pilot reports; never selects by Test."""
from adaptive_common import *

def readcsv(path):return list(csv.DictReader(Path(path).open()))
def merge(names,folder):
 for name in names:
  rows=[]
  for j in protocol()['jobs']:rows+=readcsv(OUT/folder/str(j['id'])/(name+'.csv'))
  csvsave(OUT/(name+'.csv'),rows)
def phase_report(phase):
 jobs=protocol()['jobs']
 if phase=='residual':
  results=[json.loads((OUT/'diagnostics'/str(j['id'])/'residual_status.json').read_text()) for j in jobs];assert all(r['status']=='PASS' for r in results)
  merge(['observed_residual_summary','residual_components','residual_seasonal','residual_spatial'],'diagnostics')
  lines=['# Physics residual diagnosis','', 'Selected training windows only. Native linear observed-endpoint diagnostic matches the training time discretization; observed daily path is separately labelled. Learned entrainment comes from Fixed Validation-best checkpoint and is not independent observed RHS. No Test prediction/evaluation.', '', '| Ratio | Lead | Seed | Mode | Bias RHS−dT | RMSE K/day | Corr | Clip fraction |','|---|---|---|---|---|---|---|']
  for r in readcsv(OUT/'observed_residual_summary.csv'):
   if r['group']=='all':lines.append('| '+' | '.join(r[k] for k in ['ratio','lead','seed','mode','rhs_minus_observed_bias','residual_rmse','pearson','clip_fraction'])+' |')
  lines+=['','Pooled scale recomputation agrees with v3 manifests. Native daily function versus RHS helper agrees at floating-point tolerances. No fatal implementation inconsistency found. Residual magnitude/correlation is a model adequacy diagnostic, not proof of a coding error. Prescribed1K contrast and positive, horizon-constant learned we restrict closure.','', 'Exclusions and MHW label context: diagnostics/{job}/residual_status.json. Per-season and660-cell summaries retain negative findings. Raw dataset finite values may include upstream nearest-neighbor filling per metadata; this is not proof of absence of observation error.']
  (OUT/'physics_residual_report.md').write_text('\n'.join(lines)+'\n');save(OUT/'residual_gate.json',dict(status='PASS',rows=results,scope='code consistency and finite selected observations; not physical closure adequacy',timestamp=stamp()))
 elif phase=='gradient':
  results=[json.loads((OUT/'diagnostics'/str(j['id'])/'gradient_status.json').read_text()) for j in jobs];assert all(r['status']=='PASS' for r in results);merge(['gradient_diagnostics','gradient_alignment'],'diagnostics')
  data=readcsv(OUT/'gradient_diagnostics.csv');align=readcsv(OUT/'gradient_alignment.csv');lines=['# Data / Physics gradient diagnosis','', 'Unweighted native objectives; shared predictor excludes we_head/rres_head.32 batches of16 training windows at each saved Fixed best/last checkpoint; identical batches between states, deterministic seed-specific permutations.10% has376 samples, so512 diagnostic draws explicitly repeat136 samples. No initial checkpoint was saved: N/A; existing smoke one-batch initialization logs are distinct and not fabricated checkpoints. No optimizer.step and parameter SHA unchanged.','', '| Ratio | Lead | Seed | State | Ratio median | Cosine mean | Conflict |','|---|---|---|---|---|---|---|']
  for r in align:lines.append('| '+' | '.join(r[k] for k in ['ratio','lead','seed','state','weighted_ratio_median','cosine_mean','conflict_fraction'])+' |')
  clips=[]
  for j in jobs:
   for label,key in [('baseline','reference_baseline_run'),('fixed','reference_fixed_run')]:
    history=json.loads((Path(j[key])/'history.json').read_text());clips.append(dict(job=j['id'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=label,updates=len(history)*478,clipped_fraction=sum(r['gradient_clipped_batches'] for r in history)/(478*len(history)),max_gradient=max(r['gradient_norm_max'] for r in history)))
  csvsave(OUT/'historical_gradient_clipping.csv',clips);lines+=['','Historical actual training clipping frequencies: historical_gradient_clipping.csv. Measured ratio uses nominal .01; checkpoint warmup factor is separately recorded. Component gradients N/A: components are not independent native objectives. Magnitude imbalance, direction conflict and residual adequacy are distinct; no causal contribution percentage is inferred.']
  (OUT/'gradient_report.md').write_text('\n'.join(lines)+'\n')
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axs=plt.subplots(1,2,figsize=(10,4))
  for ratio,lead in [(50,14),(10,7)]:
   r=[x for x in data if int(x['ratio'])==ratio and int(x['lead'])==lead and x['state']=='best'];axs[0].hist([float(x['weighted_ratio']) for x in r],bins=20,alpha=.5,label=f'{ratio}% L{lead}');axs[1].hist([float(x['cosine']) for x in r if x['cosine']],bins=20,alpha=.5,label=f'{ratio}% L{lead}')
  axs[0].set_xlabel('||0.01 g_phys|| / ||g_data||');axs[1].set_xlabel('Gradient cosine');axs[1].axvline(0,color='k');axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(OUT/'figures/gradient_distributions.png',dpi=180);plt.close(fig)
  assert json.loads((OUT/'residual_gate.json').read_text())['status']=='PASS';save(OUT/'diagnosis_gate.json',dict(status='PASS',gradient=results,residual_gate='residual_gate.json',meaning='No fatal implementation inconsistency; physical/gradient limitations documented',timestamp=stamp()))
 elif phase=='preflight':
  results=[json.loads((OUT/'preflight'/str(j['id'])/'status.json').read_text()) for j in jobs];assert all(r['status']=='PASS' for r in results);peak=max(r['peak_reserved_mib'] for r in results)
  resources=[dict(job=r['job'],**x) for r in results for x in r['benchmark']];csvsave(OUT/'resource_usage.csv',resources)
  save(OUT/'preflight.json',dict(status='PASS',rows=results,peak_reserved_mib=peak,timestamp=stamp()))
  (OUT/'preflight.md').write_text(f'# Adaptive preflight PASS\n\n6conditions CPU analytic3-update tests; GPU native3-update smoke; Fixed loss/shared gradient exact at equal coefficient; ocean mask/stencil; parameter+optimizer invariance; native clipping; inactiveRres; adaptive lambda/EMA checkpoint next-update exact replay; native batch-stream replay. Peak {peak:.1f}MiB.100-update paired disposable native Fixed/Adaptive timing for seed42 both cases in resource_usage.csv. No Test, no smoke weights used in production.\n')
  (OUT/'adaptive_implementation.md').write_text('# Adaptive implementation\n\nNative P0, shared predictor, optimizer and model unchanged. New isolated adaptive_common.Controller intercepts scalar loss coefficient only. Unweighted data/physics gradients via autograd.grad(retain_graph=True), excluding we_head/rres_head; no .grad accumulation or optimizer mutation. First measurement at update100; initial base lambda=.01 for updates1–99. EMA first measurement initializes to measured norms, subsequent EMA=.9 old+.1 new. lambda=clip(.25 EMA_Gdata/(EMA_Gphys+1e-12),.0001,.03); Python float detaches coefficient. Effective lambda multiplies native min(epoch/5,1) exactly once. New coefficient applies on measured update. Validation never updates EMA/counter. fp32 matches originals; unsupported precision fails rather than silently using unscaled AMP gradients. Zero/invalid direction recorded; nonfinite aborts. No PCGrad. Adaptive state, full trajectory and RNG/optimizer/stream are saved in checkpoints.\n\nFiles: scripts/adaptive_common.py, train_adaptive_pilot.py, preflight_adaptive_pilot.py. Fixed src and trainers are protected by original SHA checks in protocol.json. Synthetic smoke starts controller counter98 to cross100 boundary in3updates; production counter always0 or restored. Smoke benchmark is separately discarded. Training logs include preclip gradient statistics, actual warmup-weighted lambda, loss and clipping.\n')
 log(f'Adaptive phase {phase} gate PASS; outputs {OUT}.')

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['residual','gradient','preflight']);o=ap.parse_args();phase_report(o.phase)
