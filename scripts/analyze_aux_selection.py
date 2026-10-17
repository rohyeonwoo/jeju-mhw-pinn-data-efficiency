"""Predeclared validation-only auxiliary selection and matched final comparison."""
import statistics,json
from pathlib import Path
import train_v3_study as w

def history_verified(j,protocol=None):
 d=Path(j['run_dir']);cfg=json.loads((d/'config.json').read_text());h=json.loads((d/'history.json').read_text())
 assert cfg['manifest_sha256']==w.sha(j['manifest'])==j['manifest_sha256']
 if protocol is not None:assert cfg['protocol_sha256']==w.digest(protocol)
 assert cfg['settings']['lambda_physics']==j['lambda_physics'] and cfg['settings']['lambda_rres']==j['lambda_rres']
 ref=Path(j['reference_p0_run']);rc=json.loads((ref/'config.json').read_text());rh=json.loads((ref/'history.json').read_text())
 for key in ['stats','shared_initial_sha256','parameters','manifest_sha256','split_sha256','tensor_fingerprints','rhs_scale','applied_rhs_scale']:assert cfg[key]==rc[key],key
 for x,y in zip(h,rh):assert x['batch_order_sha256']==y['batch_order_sha256']
 for x in h:
  assert bool(x['physics_calls'])==bool(j['lambda_physics']) and bool(x['rres_calls'])==bool(j['lambda_rres'])
 return h

def select(out):
 out=Path(out);p=json.loads((out/'protocol.json').read_text());rows=[]
 for j in p['jobs']:
  if not j['pilot']:continue
  d=Path(j['run_dir']);s=json.loads((d/'status.json').read_text());assert s['status'] in ['pilot_complete','trained']
  h=history_verified(j,p);prefix=[r for r in h if r['epoch']<=p['pilot_epochs']];assert len(prefix)==p['pilot_epochs']
  best=min(prefix,key=lambda r:r['val_rmse']);rows.append(dict(seed=j['seed'],weight=j['lambda_rres'],val_rmse=best['val_rmse'],best_epoch=best['epoch'],run_dir=j['run_dir']))
 baseline={r['seed']:r['val_rmse'] for r in rows if r['weight']==0};scores=[]
 for beta in p['candidate_weights']:
  group=[r for r in rows if r['weight']==beta];assert len(group)==3
  ratio=[r['val_rmse']/baseline[r['seed']] for r in group]
  scores.append(dict(weight=beta,mean_relative_rmse=statistics.mean(ratio),mean_rmse=statistics.mean(r['val_rmse'] for r in group),sd_rmse=statistics.stdev(r['val_rmse'] for r in group),improved_seeds=sum(v<1 for v in ratio)))
 best=min(scores,key=lambda s:(s['mean_relative_rmse'],s['weight']))
 result=dict(status='SELECTED',selected_weight=best['weight'],selection='minimum three-seed mean paired Validation RMSE among prespecified positive candidates; ties favor smaller weight',baseline_is_better_than_all_candidates=best['mean_relative_rmse']>=1,claim='best positive candidate for diagnostic ablation; positive auxiliary benefit not established' if best['mean_relative_rmse']>=1 else 'pilot Validation improvement only; not independent generalization evidence',pilot_epochs=p['pilot_epochs'],rows=rows,scores=scores,test_used=False)
 sf=out/'selection.json'
 if sf.exists():assert json.loads(sf.read_text())==result,'Selection drift'
 else:w.save(sf,result)
 lines=['# Auxiliary weight selection: Validation only','',str(result['claim']),'',f"Selected positive weight: {best['weight']}; common {p['pilot_epochs']}x478 update pilot.",'','| Weight | Mean RMSE | Relative RMSE | Improved seeds |','|---|---|---|---|']
 for s in scores:lines.append(f"| {s['weight']} | {s['mean_rmse']:.6f} | {s['mean_relative_rmse']:.6f} | {s['improved_seeds']}/3 |")
 (out/'selection.md').write_text('\n'.join(lines)+'\n');return result

def final_jobs(p,selection):
 beta=selection['selected_weight'];return [j for j in p['jobs'] if j['arm']=='baseline' or j['lambda_rres']==beta]

def summarize_final(out,require_complete=False):
 out=Path(out);p=json.loads((out/'protocol.json').read_text());sel=json.loads((out/'selection.json').read_text());rows=[];pending=[]
 for j in p['references']+final_jobs(p,sel):
  d=Path(j['run_dir']);sf=d/'status.json'
  if not sf.exists() or json.loads(sf.read_text())['status']!='trained':pending.append(j['name']);continue
  h=history_verified(j,None if j in p['references'] else p);best=min(h,key=lambda r:r['val_rmse'])
  rows.append(dict(seed=j['seed'],arm=j['arm'],weight=j['lambda_rres'],val_rmse=best['val_rmse'],best_epoch=best['epoch'],epochs=len(h),data_loss=best['val_data_loss'],physics_loss=best['val_physics_loss'] if j['lambda_physics'] else None,auxiliary_loss=best['val_rres_l2'] if j['lambda_rres'] else None,gradient_norm_mean=best['gradient_norm_mean'],run_dir=j['run_dir']))
 if require_complete:assert not pending
 groups=[];comparisons=[]
 if not pending:
  for arm in ['baseline','physics_only','auxiliary_only','full_pinn']:
   rs=[r for r in rows if r['arm']==arm];assert len(rs)==3;groups.append(dict(arm=arm,mean=statistics.mean(r['val_rmse'] for r in rs),sd=statistics.stdev(r['val_rmse'] for r in rs)))
  for arm,ref in [('auxiliary_only','baseline'),('full_pinn','physics_only'),('full_pinn','baseline'),('physics_only','baseline')]:
   changes=[100*(1-next(r['val_rmse'] for r in rows if r['seed']==s and r['arm']==arm)/next(r['val_rmse'] for r in rows if r['seed']==s and r['arm']==ref)) for s in [42,7,123]]
   comparisons.append(dict(arm=arm,reference=ref,mean_paired_improvement_percent=statistics.mean(changes),improved_seeds=sum(v>0 for v in changes),by_seed=dict(zip([42,7,123],changes))))
 result=dict(status='COMPLETE' if not pending else 'PENDING',selected_auxiliary_weight=sel['selected_weight'],completed=len(rows),total=12,pending=pending,rows=rows,groups=groups,comparisons=comparisons,selection_caveat=sel['claim'],test_used=False)
 w.save(out/'final_results.json',result)
 lines=['# Matched four-arm final Validation comparison','',f"{result['status']}: {len(rows)}/12. Auxiliary={sel['selected_weight']}, physics=.01. Identical all-head architecture and 478-update evaluation; no Test inference.",'','Pilot Baseline and selected Auxiliary resume exact checkpoints/optimizer/RNG to the original80-block/patience12 protocol. Full starts from the same seed; existing Physics-only follows that same full protocol. Pilot budget cannot alter LR/warmup or final patience. Validation reused for selection: exploratory, not independent Test evidence.','','| Seed | Arm | Best epoch | Val RMSE |','|---|---|---|---|']
 for r in rows:lines.append(f"| {r['seed']} | {r['arm']} | {r['best_epoch']} | {r['val_rmse']:.6f} |")
 lines+=['','| Comparison | Paired improvement | Improved seeds |','|---|---|---|']
 for c in comparisons:lines.append(f"| {c['arm']} vs {c['reference']} | {c['mean_paired_improvement_percent']:+.3f}% | {c['improved_seeds']}/3 |")
 (out/'final_report.md').write_text('\n'.join(lines)+'\n');return result
