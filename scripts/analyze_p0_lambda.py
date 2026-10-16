"""Validation-only comparison; no test files or guarded experiment results read."""
import csv,json,math,statistics
from pathlib import Path
import train_v3_study as w

def summarize(out,require_complete=False):
 out=Path(out);p=json.loads((out/'protocol.json').read_text());rows=[];pending=[]
 for j in p['references']+p['jobs']:
  d=Path(j['run_dir']);sf=d/'status.json'
  if not sf.exists() or json.loads(sf.read_text())['status']!='trained':pending.append(j['name']);continue
  cfg=json.loads((d/'config.json').read_text());hist=json.loads((d/'history.json').read_text());s=json.loads(sf.read_text())
  if j in p['jobs']:assert cfg['protocol_sha256']==w.digest(p)
  assert cfg['settings']['lambda_physics']==j['lambda_physics']
  assert cfg['manifest_sha256']==w.sha(j['manifest'])==j['manifest_sha256']
  best=min(hist,key=lambda r:r['val_rmse']);assert best['epoch']==s['best_epoch'] and best['val_rmse']==s['best_val_rmse']
  for ref in [j['reference_baseline_run'],j['reference_p0_run']]:
   rc=json.loads((Path(ref)/'config.json').read_text());rh=json.loads((Path(ref)/'history.json').read_text())
   for k in ['shared_initial_sha256','stats','manifest_sha256','split_sha256','tensor_fingerprints']:assert cfg[k]==rc[k],k
   for x,y in zip(hist,rh):assert x['batch_order_sha256']==y['batch_order_sha256']
   for k in cfg['settings']:
    if k not in ['lambda_physics','physics_mode','output_root'] and not (k=='physics_rhs_scale' and (cfg['settings']['physics_mode']=='none' or rc['settings']['physics_mode']=='none')):assert cfg['settings'][k]==rc['settings'][k],k
  assert all(r['rres_calls']==0 for r in hist)
  if j['lambda_physics']>0:
   assert cfg['rhs_scale']==cfg['applied_rhs_scale']
   for r in hist:
    assert abs(r['train_lambda_physics_effective']-j['lambda_physics']*min(r['epoch']/5,1))<1e-12
  fields=dict(lead=j['lead'],seed=j['seed'],lambda_physics=j['lambda_physics'],best_epoch=s['best_epoch'],epochs=s['epochs'],val_rmse=best['val_rmse'],val_bias=best['val_bias'],val_physics_loss=best['val_physics_loss'] if j['lambda_physics'] else None,val_residual_rmse_kday=best.get('val_common_res_rmse_kday'),run_dir=str(d),reused=j in p['references'])
  assert all(math.isfinite(v) for v in fields.values() if isinstance(v,(int,float)))
  rows.append(fields)
 if require_complete:assert not pending,pending
 groups=[];selection=None
 if not pending:
  baseline={(r['lead'],r['seed']):r['val_rmse'] for r in rows if r['lambda_physics']==0}
  for r in rows:r['relative_rmse_to_baseline']=r['val_rmse']/baseline[r['lead'],r['seed']]
  for lead in [7,14]:
   for lam in p['candidate_lambdas']:
    rs=[r for r in rows if r['lead']==lead and r['lambda_physics']==lam];assert len(rs)==3
    residual=[r['val_residual_rmse_kday'] for r in rs if r['val_residual_rmse_kday'] is not None]
    groups.append(dict(lead=lead,lambda_physics=lam,val_rmse_mean=statistics.mean(r['val_rmse'] for r in rs),val_rmse_sd=statistics.stdev(r['val_rmse'] for r in rs),mean_paired_improvement_percent=statistics.mean(100*(1-r['relative_rmse_to_baseline']) for r in rs),improved_baseline_seeds=sum(r['relative_rmse_to_baseline']<1 for r in rs),residual_rmse_kday_mean=statistics.mean(residual) if residual else None))
  scores=[dict(lambda_physics=lam,mean_relative_validation_rmse=statistics.mean(r['relative_rmse_to_baseline'] for r in rows if r['lambda_physics']==lam)) for lam in p['candidate_lambdas']]
  winner=min(scores,key=lambda r:(r['mean_relative_validation_rmse'],r['lambda_physics']))
  selection=dict(scope='10% only; finite candidate set, not a global optimum or independent test proof',shared_lambda=winner['lambda_physics'],scores=scores,claim='No physics benefit established' if winner['lambda_physics']==0 else 'Candidate selected on validation; generalization unverified')
 result=dict(timestamp=w.stamp(),status='COMPLETE' if not pending else 'PENDING',completed=len(rows),total=30,pending=pending,runs=rows,by_lead=groups,selection=selection,test_evaluated=False,physics_comparison='Native P0 residual with learned nonnegative we. Baseline has no physics diagnostic: null, never zero. Compare positive lambdas on the same subset and mask.')
 w.save(out/'validation_result.json',result)
 if rows:
  with (out/'validation_comparison.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# Native P0 lambda sensitivity — validation only','',f"Status: {result['status']}; {len(rows)}/30 conditions available (12 reused,18 new).",'', 'No gradient surgery. All native settings except lambda/output path are checked against the original study. Exact common batch-order prefixes, initial predictor, v3 statistics and data splits are checked.','', '| Lead | lambda | Validation RMSE mean ± SD (K) | vs baseline (%) | Improved seeds | P0 residual RMSE (K/day) |','|---|---|---|---|---|---|']
 for r in groups:lines.append(f"| {r['lead']} | {r['lambda_physics']} | {r['val_rmse_mean']:.6f} ± {r['val_rmse_sd']:.6f} | {r['mean_paired_improvement_percent']:+.3f} | {r['improved_baseline_seeds']}/3 | {r['residual_rmse_kday_mean']} |")
 lines+=['',json.dumps(selection,indent=2),'','Baseline physics diagnostics are unavailable, not zero. P0 raw residual includes its learned we and is distinct from clipped/normalized training loss. RMSE selection and physics trade-offs must both be reported. The existing Test results were seen before this exploratory study; no new Test inference is run. A 10% winner is not automatically optimal for 25/50/100%. No automatic rollout beyond these 18 jobs.']
 (out/'validation_report.md').write_text('\n'.join(lines)+'\n')
 return result
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);opt=ap.parse_args();print(summarize(opt.study)['status'])
