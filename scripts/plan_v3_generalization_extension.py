"""Read-only completed-run reuse audit and unexecuted follow-up plan."""
import json,hashlib,datetime,zoneinfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v3_generalization_20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,default=str).encode()).hexdigest()
ref=json.loads((ROOT/'outputs/aux_weight_selection_20261009/runs/lead7_seed42/baseline_aux0/config.json').read_text())
protocols=[ROOT/'outputs/aux_weight_selection_20261009/protocol.json',ROOT/'outputs/p0_lambda_v3_20261009/protocol.json',ROOT/'outputs/data_efficiency/v3_baseline_p0_20261008_r1/protocol.json']
audit=[];pool={}
allowed={'subset_manifest','output_root','lead','seed','lambda_physics','physics_rhs_scale'}
for path in protocols:
 prot=json.loads(path.read_text());source_ok=all(sha(ROOT/n)==v for n,v in prot['code_sha256'].items())
 for j in prot['jobs']:
  d=Path(j['run_dir'])
  if not (d/'config.json').exists():continue
  c=json.loads((d/'config.json').read_text());settings=c['settings']
  if settings['lambda_rres']!=0:continue
  status=json.loads((d/'status.json').read_text())
  structural=c['parameters']==ref['parameters']
  diffs={k:[v,settings.get(k)] for k,v in ref['settings'].items() if k not in allowed and settings.get(k)!=v}
  m=json.loads(Path(c['job']['manifest']).read_text())
  checks=dict(completed=status['status']=='trained',same_structure=structural,settings=not diffs,source=source_ok,protocol=c['protocol_sha256']==digest(prot),manifest=sha(c['job']['manifest'])==c['manifest_sha256'],statistics=c['stats']==m['statistics'],rhs=c['applied_rhs_scale']==m['physics_rhs_scale'],best_checkpoint_exists=(d/'best.pt').exists())
  eligible=all(checks.values());weight=settings['lambda_physics'];key=(j['ratio'],j['lead'],j['seed'],weight)
  row=dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],weight=weight,run_dir=str(d),eligible=eligible,checks=checks,setting_differences=diffs)
  if eligible:
   h=json.loads((d/'history.json').read_text());assert min(h,key=lambda z:z['val_rmse'])['epoch']==status['best_epoch']
   row['checkpoint_sha256']=sha(d/'best.pt');pool[key]=row
  audit.append(row)
fixed=[(r,l,s,w) for r in [10,25,50,100] for l in [7,14] for s in [42,7,123] for w in [0,.01]]
selected=[(r,l,s,w) for r in [10,25,50,100] for l in [7,14] for s in [42,7,123] for w in [0,.001,.01,.03,.1]]
result=dict(status='PLAN_ONLY_NO_LAUNCH',fixed_total=len(fixed),fixed_reusable=sum(k in pool for k in fixed),fixed_additional=sum(k not in pool for k in fixed),selection_total=len(selected),selection_reusable=sum(k in pool for k in selected),selection_additional=sum(k not in pool for k in selected),selection_additional_after_fixed=sum(k not in pool and k not in fixed for k in selected),audit=audit)
(OUT/'reuse_audit.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Follow-up data-volume study — plan only','', 'No new training is launched. Same native predictor + we_head + rres_head in BOTH Baseline and Physics-only; Baseline disables both penalties. v3 manifests and nested indices fixed; all other settings, batch16, AdamW,478-update Validation,80-block max and patience12 retained. Best checkpoint selected only on Validation.','', '## Fixed λ=.01 question','',f"4ratios ×2leads ×3seeds ×2arms = {len(fixed)} conditions. Config/source/manifest/status audit finds {result['fixed_reusable']} reusable runs and {result['fixed_additional']} additional runs: existing matched all-head Baseline Lead7 3, existing .01 Physics-only Lead7/14 6. Train21 remaining Baselines and18 remaining Physics-only models. Before a future launch, strictly load each reused checkpoint and recheck recorded hashes; this step does not imply new training authorization.",'', '## Separate Validation-selected-weight question','', 'Retain previously investigated positive candidate set {.001,.01,.03,.1} and λ=0 control; do not expand this set based on the new Test. Per ratio/lead choose one common weight across seeds by lowest mean paired Validation RMSE; ties smaller weight. Include λ=0 as a selectable null result; no guaranteed positive selection. Selected candidates reused, no retraining solely to apply selection. All training budgets and selection intervals identical.',f"24ratio/lead/seed cells ×5weights = {len(selected)} candidate runs. Existing eligible runs: {result['selection_reusable']}; additional: {result['selection_additional']}. If the fixed-.01 plan is executed first, additional positive candidate runs remaining: {result['selection_additional_after_fixed']}.",'', '## Exclusions and limitations','', 'Old 24 Baselines have no auxiliary heads, so they are not counted as reusable in the strict identical-structure follow-up design. Existing P0 .1 models have matching heads and native settings and are reusable only in the selected-weight candidate plan, not as .01 checkpoints. Dense24-update-validation checkpoints change selection/stopping and are excluded. Guarded-gradient repair checkpoints also excluded. See reuse_audit.json for actual run paths and every check.','', 'Fixed-.01 and Validation-selected comparisons answer different questions and must be reported separately. Previously inspected Test periods remain exploratory. No Test-driven candidate adjustment, structural change or checkpoint reselection. Any future independent confirmation requires separately specified evaluation data; no new split is created here.']
(OUT/'followup_plan.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='audit'}))
