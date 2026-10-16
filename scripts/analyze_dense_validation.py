"""Summarize dense checkpoint selection and common physical diagnostics separately."""
import json,statistics
from pathlib import Path
import train_v3_study as w

def summarize(out,require_complete=False):
 out=Path(out);p=json.loads((out/'protocol.json').read_text());trained=[];diagnostics=[];pending=[]
 for j in p['jobs']:
  d=Path(j['run_dir']);sf=d/'status.json'
  if not sf.exists() or json.loads(sf.read_text())['status']!='trained':pending.append(j['name']);continue
  if j['kind']=='diagnostics':
   result=json.loads((d/'diagnostics.json').read_text());assert result['status']=='complete'
   diagnostics.append(dict(lead=j['lead'],seed=j['seed'],**result));continue
  s=json.loads(sf.read_text());hist=json.loads((d/'history.json').read_text());c=json.loads((d/'config.json').read_text());ref=json.loads((Path(j['reference_run'])/'status.json').read_text())
  assert c['protocol_sha256']==w.digest(p) and c['manifest_sha256']==w.sha(j['manifest'])
  assert s['best_step']==min(hist,key=lambda r:r['val_rmse'])['steps']
  assert s['steps']<=38240 and (s['steps']==38240 or s['steps']-s['best_step']==5736)
  assert all(r['rres_calls']==0 for r in hist)
  matched=[r['reference_val_rmse_difference'] for r in hist if 'reference_val_rmse_difference' in r];assert matched and max(matched)<2e-6
  trained.append(dict(lead=j['lead'],seed=j['seed'],arm=j['arm'],lambda_physics=j['lambda_physics'],best_step=s['best_step'],stopped_step=s['steps'],dense_val_rmse=s['best_val_rmse'],original_val_rmse=ref['best_val_rmse'],original_best_step=ref['best_epoch']*478,improvement_vs_original_percent=100*(ref['best_val_rmse']-s['best_val_rmse'])/ref['best_val_rmse'],first_interval_best=s['best_step']<478,max_common_trajectory_difference=max(matched),diagnostics=json.loads((d/'best_diagnostics.json').read_text())))
 if require_complete:assert not pending,pending
 groups=[]
 if len(trained)==12:
  for lead in [7,14]:
   rows=[r for r in trained if r['lead']==lead];b={r['seed']:r for r in rows if r['arm']=='baseline'};ph=[r for r in rows if r['arm']!='baseline']
   group=dict(lead=lead,paired_improvement_percent=statistics.mean(100*(b[r['seed']]['dense_val_rmse']-r['dense_val_rmse'])/b[r['seed']]['dense_val_rmse'] for r in ph),improved_seeds=sum(r['dense_val_rmse']<b[r['seed']]['dense_val_rmse'] for r in ph))
   for arm,rs in [('baseline',list(b.values())),('p0',ph)]:
    group[arm]=dict(rmse_mean=statistics.mean(r['dense_val_rmse'] for r in rs),rmse_sd=statistics.stdev(r['dense_val_rmse'] for r in rs),original_rmse_mean=statistics.mean(r['original_val_rmse'] for r in rs),common_zero_we_residual_mean=statistics.mean(r['diagnostics']['zero_we']['rmse_kday'] for r in rs),best_steps=[r['best_step'] for r in rs])
   groups.append(group)
 result=dict(timestamp=w.stamp(),status='COMPLETE' if not pending else 'PENDING',training_completed=len(trained),diagnostic_groups_completed=len(diagnostics),pending=pending,training=trained,existing_checkpoint_diagnostics=diagnostics,by_lead=groups,test_evaluated=False)
 w.save(out/'results.json',result)
 lines=['# Dense validation and shared residual diagnostic','',f"Training {len(trained)}/12; existing checkpoint groups {len(diagnostics)}/6. Status {result['status']}.",'', '24-update validation plus original478-update boundaries. Original schedule and38240-update budget retained; patience5736 updates after the best validation. Evaluation preserves RNG; matching original boundaries are checked to2e-6K. No Test inference.','', 'Common zero-we residual omits entrainment for every model and is a resolved flux/advection diagnostic, not a complete heat-budget claim. Learned-we residual is reported separately for P0. Observed-target zero-we residual is a reference, not a zero-error expectation.','', '## Existing checkpoints','', '| Lead | Seed | lambda | SST RMSE | common we=0 residual K/day | learned-we residual K/day |','|---|---|---|---|---|---|']
 for group in diagnostics:
  for item in group['results']:
   r=item['diagnostics'];learned=r['learned_we']
   lines.append(f"| {group['lead']} | {group['seed']} | {item['source']['lambda_physics']} | {r['val_rmse']:.6f} | {r['zero_we']['rmse_kday']:.6f} | {learned['rmse_kday'] if learned else 'N/A'} |")
 lines+=['','## Dense validation comparison','',json.dumps(groups,indent=2),'', 'Selection on more validation checkpoints can increase validation selection bias. Treat as exploratory; a small validation gain is not proof of generalization. The finite candidate lambda=.01 was selected earlier; no new lambda search in this study. No automatic extension to other ratios.']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 return result
