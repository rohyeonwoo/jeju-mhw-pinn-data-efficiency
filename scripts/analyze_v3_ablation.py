"""Validation-only four-arm summary; never reads test metrics."""
from pathlib import Path
import json,csv,statistics
import train_v3_study as w

def summarize(out,require_complete=False):
 out=Path(out);p=json.loads((out/'protocol.json').read_text());rows=[];pending=[]
 for j in p['references']+p['jobs']:
  d=Path(j['run_dir']);sf=d/'status.json'
  if not sf.exists() or json.loads(sf.read_text())['status']!='trained':pending.append(j['name']);continue
  cfg=json.loads((d/'config.json').read_text());history=json.loads((d/'history.json').read_text());s=json.loads(sf.read_text())
  if j in p['jobs']:assert cfg['protocol_sha256']==w.digest(p)
  assert cfg['manifest_sha256']==w.sha(j['manifest'])==j['manifest_sha256']
  assert cfg['settings']['lambda_physics']==j['lambda_physics'] and cfg['settings']['lambda_rres']==j['lambda_rres']
  ref=json.loads((Path(j['reference_p0_run'])/'config.json').read_text());rh=json.loads((Path(j['reference_p0_run'])/'history.json').read_text())
  for key in ['shared_initial_sha256','stats','manifest_sha256','split_sha256','tensor_fingerprints','parameters','trainable_parameters','rhs_scale','applied_rhs_scale']:assert cfg[key]==ref[key],key
  for a,b in zip(history,rh):assert a['batch_order_sha256']==b['batch_order_sha256']
  for h in history:
   assert bool(h['physics_calls'])==bool(j['lambda_physics']) and bool(h['rres_calls'])==bool(j['lambda_rres'])
  best=min(history,key=lambda r:r['val_rmse']);assert best['epoch']==s['best_epoch'] and best['val_rmse']==s['best_val_rmse']
  rows.append(dict(seed=j['seed'],arm=j['arm'],lambda_physics=j['lambda_physics'],lambda_rres=j['lambda_rres'],best_epoch=s['best_epoch'],epochs=s['epochs'],val_rmse=best['val_rmse'],val_data_loss=best['val_data_loss'],val_physics_loss=best['val_physics_loss'] if j['lambda_physics'] else None,val_rres_loss=best['val_rres_l2'] if j['lambda_rres'] else None,run_dir=str(d),reused=j in p['references']))
 if require_complete:assert not pending
 groups=[]
 if not pending:
  for arm in ['baseline','physics_only','auxiliary_only','full_pinn']:
   rs=[r for r in rows if r['arm']==arm];assert len(rs)==3
   ratios=[r['val_rmse']/next(b['val_rmse'] for b in rows if b['arm']=='baseline' and b['seed']==r['seed']) for r in rs]
   groups.append(dict(arm=arm,mean=statistics.mean(r['val_rmse'] for r in rs),sd=statistics.stdev(r['val_rmse'] for r in rs),mean_paired_improvement_percent=100*(1-statistics.mean(ratios)),improved_seeds=sum(v<1 for v in ratios)))
 result=dict(status='COMPLETE' if not pending else 'PENDING',completed=len(rows),total=12,pending=pending,runs=rows,groups=groups,test_evaluated=False)
 w.save(out/'results.json',result)
 lines=['# v3 10% Lead7 four-arm ablation','',f"{result['status']}: {len(rows)}/12 conditions (3 Physics-only reused; 9 matched-head runs).",'', 'Validation only. Physics lambda=.01 selected from completed sensitivity; auxiliary lambda=10 fixed. Same v3 manifests, all heads, batch stream, 478-update validation, optimizer and schedules. No dense-validation mixing.','', '| Seed | Arm | Best epoch | Validation RMSE |','|---|---|---|---|']
 for r in rows:lines.append(f"| {r['seed']} | {r['arm']} | {r['best_epoch']} | {r['val_rmse']:.6f} |")
 lines+=['', '| Arm | Mean ± SD | Paired improvement | Improved seeds |','|---|---|---|---|']
 for g in groups:lines.append(f"| {g['arm']} | {g['mean']:.6f} ± {g['sd']:.6f} | {g['mean_paired_improvement_percent']:+.3f}% | {g['improved_seeds']}/3 |")
 lines+=['','Inactive loss placeholders are not measured physical residuals. Auxiliary is the existing zero-target unresolved-residual penalty, independent of the physics residual. Validation-selected exploratory comparison; no claim of independent Test improvement.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');return result
