"""Report all controlled physics arms; select settings on validation only."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from summarize_lambda_pilot import write_csv, BASE, SEEDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    out=p.parse_args().output_root.resolve();assert out.is_relative_to(BASE)
    state=json.loads((out/'_queue/batch_status.json').read_text())
    dest=out/'_analysis';dest.mkdir(exist_ok=False)
    specs=[(j['arm'],j['lambda_physics'],j['seed'],Path(j['run_dir']),True) for j in state['jobs']]
    for seed in SEEDS:
        specs.extend([
            ('legacy_baseline',0.,seed,BASE/f'allocation_batch16_5plus5/ratio_10/jeju_hilam/lead7_seed{seed}',False),
            ('legacy_aux_only',0.,seed,BASE/f'lambda_pilot_lead7_20260925/lambda_0/ratio_10/pi_jeju_hilam/lead7_seed{seed}',False),
            ('legacy_p0_aux',.1,seed,BASE/f'allocation_batch16_5plus5/ratio_10/pi_jeju_hilam/lead7_seed{seed}',False)])
    rows=[];lookup={};daily={};checks={};rng={}
    for arm,lam,seed,d,new in specs:
        c=json.loads((d/'jeju_hilam_config.json').read_text())
        mp=next(d.glob('*_metrics.json'));m=json.loads(mp.read_text())
        assert json.loads((d/'progress.json').read_text())['status']=='completed'
        assert not c['smoke_only'] and c['batch_size']==16 and c['lead']==7 and c['train_samples']==376
        invariant={k:c[k] for k in ['split_sha256','full_statistics','source_sha256','data_sha256','S','max_steps','warmup_steps']}
        if 'invariant' not in checks:checks['invariant']=invariant
        assert invariant==checks['invariant']
        if new:
            assert c['lambda_rres']==0 and c['lambda_we_tie']==0 and c['zero_rres_physics']
            reproducible=[c['initial_state_sha256'],c['initial_cpu_rng_sha256'],m['first_128_train_batches_sha256']]
            if seed not in rng:rng[seed]=reproducible
            assert rng[seed]==reproducible, (arm,lam,seed,'initialization or batch-order mismatch')
            if 'physics_policy' not in checks:checks['physics_policy']=[c['physics_stencil'],c['physics_rhs_scale']]
            assert checks['physics_policy']==[c['physics_stencil'],c['physics_rhs_scale']]
        root=d.parent.parent
        mhw=json.loads((root/'mhw_skill.json').read_text())
        z=next(r for r in mhw['rows'] if r['model']==c['model'] and r['seed']==seed and r['lead_days']==7)
        assert z['status']=='ok'
        label=f'{arm}:{lam:g}'
        row=dict(arm=arm,lambda_physics=lam,label=label,seed=seed,new_controlled_run=new,
                 best_val_rmse=m['best_selection_value'],test_rmse=m['test_rmse'],test_acc=m['test_acc'],
                 mhw_csi=z['mhw_grid_csi'],mhw_pod=z['mhw_grid_pod'],mhw_far=z['mhw_grid_far'],
                 actual_steps=m['actual_steps'],metrics_path=str(mp),fields_path=m['fields_path'])
        rows.append(row);lookup[label,seed]=row
        with np.load(m['fields_path']) as f:
            obs=f['target_raw'];mask=f['mask'];idx=f['target_index'];pred=f['pred_raw']
            if 'obs' not in checks:checks.update(obs=obs.copy(),mask=mask.copy(),idx=idx.copy())
            for k,v in [('obs',obs),('mask',mask),('idx',idx)]:np.testing.assert_array_equal(v,checks[k])
            sea=mask[0]>0;err=pred[:,sea].astype(float)-obs[:,sea]
            mse=np.mean(err**2,axis=1)
            assert np.isclose(np.sqrt(mse.mean()),m['test_rmse'],atol=2e-6,rtol=0)
            daily[label,seed]=mse
    labels=list(dict.fromkeys(r['label'] for r in rows));summaries=[]
    for label in labels:
        rs=[lookup[label,s] for s in SEEDS]
        v=dict(label=label,arm=rs[0]['arm'],lambda_physics=rs[0]['lambda_physics'])
        for key in ['best_val_rmse','test_rmse','test_acc','mhw_csi','mhw_pod','mhw_far']:
            a=np.array([r[key] for r in rs]);v[key+'_mean']=float(a.mean());v[key+'_sd']=float(a.std(ddof=1))
        summaries.append(v)
    selected={}
    for arm in ['p0_daily','p0_interval','p1_daily']:
        candidates=[s for s in summaries if s['arm'] in [arm,'matched_baseline']]
        chosen=min(candidates,key=lambda r:(r['best_val_rmse_mean'],r['lambda_physics']))
        selected[arm]=dict(label=chosen['label'],validation_rmse=chosen['best_val_rmse_mean'],
                          criterion='mean best validation RMSE over all 3 seeds; common matched baseline is lambda=0; ties choose smaller lambda')
    boot={}
    for block in [30,90,180]:
        n=len(checks['idx']);generator=np.random.default_rng(20260925+7+block)
        start=generator.integers(0,n,size=(4000,int(np.ceil(n/block))))
        idx=((start[:,:,None]+np.arange(block))%n).reshape(4000,-1)[:,:n]
        for label in labels:boot[label,block]=np.mean([np.sqrt(daily[label,s][idx].mean(axis=1)) for s in SEEDS],axis=0)
    pairs=[(label,ref) for label in labels if label.startswith(('p0_','p1_'))
           for ref in ['matched_baseline:0','legacy_aux_only:0']]
    pairs += [(f'{arm}:{lam:g}',f'p0_daily:{lam:g}') for arm in ['p0_interval','p1_daily'] for lam in [.01,.1,.3]]
    pairs += [(selected[arm]['label'],selected['p0_daily']['label']) for arm in ['p0_interval','p1_daily']]
    comparisons=[]
    for label,ref in dict.fromkeys(pairs):
        if label==ref:continue
        diff=np.array([lookup[label,s]['test_rmse']-lookup[ref,s]['test_rmse'] for s in SEEDS])
        permutations=[np.mean(diff*np.array(signs)) for signs in itertools.product([-1,1],repeat=3)]
        row=dict(arm=label,reference=ref,delta_rmse=float(diff.mean()),
                 delta_seed42=float(diff[0]),delta_seed7=float(diff[1]),delta_seed123=float(diff[2]),
                 improved_seeds=int((diff<0).sum()),exact_two_sided_p=float(np.mean(np.abs(permutations)>=abs(diff.mean())-1e-14)))
        for block in [30,90,180]:
            lo,hi=np.quantile(boot[label,block]-boot[ref,block],[.025,.975])
            row[f'conditional_ci_{block}d_low']=float(lo);row[f'conditional_ci_{block}d_high']=float(hi)
        comparisons.append(row)
    write_csv(dest/'runs.csv',rows);write_csv(dest/'summary.csv',summaries);write_csv(dest/'paired_comparisons.csv',comparisons)
    (dest/'validation_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    lines=['# Controlled physics pilot: all results','',
           '10% (376 training windows), lead7, seeds42/7/123. Input statistics, source models and all evaluation indices preserved.',
           'New runs have matched initialization/batch order and lambda_rres=0. Physics stencil/scale changes apply identically to all new arms. Legacy arms are contextual comparisons, not isolated stencil effects.',
           'Lambda selected using validation only, including a shared lambda0 baseline. Existing test results informed this exploratory follow-up; no independent confirmatory claim.',
           'Time-block intervals condition on fitted models. Three seed pairs cannot give exact two-sided p<0.05; multiple comparisons remain descriptive.', '',
           '| Arm | Validation RMSE | Test RMSE mean ± SD | CSI | POD | FAR |','|---|---|---|---|---|---|']
    for s in summaries:
        lines.append(f"| {s['label']} | {s['best_val_rmse_mean']:.5f} | {s['test_rmse_mean']:.5f} ± {s['test_rmse_sd']:.5f} | {s['mhw_csi_mean']:.4f} | {s['mhw_pod_mean']:.4f} | {s['mhw_far_mean']:.4f} |")
    lines+=['','Validation-selected settings: '+json.dumps(selected),'',
            'See paired_comparisons.csv for all paired seed differences and common 30/90/180-day bootstrap intervals. No automatic expansion.','']
    (dest/'report.md').write_text('\n'.join(lines))
    print(dest/'report.md',flush=True)


if __name__=='__main__':main()
