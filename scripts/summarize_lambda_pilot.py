"""Summarize every pilot arm, choosing lambda only from validation scores."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
BASE=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')
SEEDS=[42,7,123]
ARMS=['baseline','aux0','p0_0.01','p0_0.1','p0_0.3']


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True)
    out=parser.parse_args().output_root.resolve()
    assert out.is_relative_to(BASE)
    report_dir=out/'_analysis';report_dir.mkdir(exist_ok=False)
    lookup={};rows=[];daily={};checks={}
    for regime,arm,seed in itertools.product(['ratio_10','full'],ARMS,SEEDS):
        if arm in ['baseline','p0_0.1']:
            root=BASE/('full_batch16_matched/full' if regime=='full' else 'allocation_batch16_5plus5/ratio_10')
        else:
            lam={'aux0':'0','p0_0.01':'0p01','p0_0.3':'0p3'}[arm]
            root=out/f'lambda_{lam}'/regime
        model='jeju_hilam' if arm=='baseline' else 'pi_jeju_hilam'
        d=root/model/f'lead7_seed{seed}'
        metric_path=next(d.glob('*_metrics.json'))
        m=json.loads(metric_path.read_text());c=json.loads((d/'jeju_hilam_config.json').read_text())
        p=json.loads((d/'progress.json').read_text())
        assert p['status']=='completed' and not c['smoke_only'] and c['batch_size']==16
        assert c['lead']==7 and c['seed']==seed
        ref={k:c[k] for k in ['split_sha256','full_statistics','data_sha256','source_sha256','S','max_steps','warmup_steps']}
        if 'config' not in checks:checks['config']=ref
        assert ref==checks['config']
        mhw=json.loads((root/'mhw_skill.json').read_text())
        z=next(r for r in mhw['rows'] if r['model']==model and r['seed']==seed and r['lead_days']==7)
        assert z['status']=='ok'
        r=dict(regime=regime,arm=arm,seed=seed,best_val_rmse=m['best_selection_value'],
               test_rmse=m['test_rmse'],test_acc=m['test_acc'],mhw_csi=z['mhw_grid_csi'],
               mhw_pod=z['mhw_grid_pod'],mhw_far=z['mhw_grid_far'],actual_steps=m['actual_steps'],
               lambda_physics=c['lambda_physics'],lambda_rres=0 if arm=='baseline' else c['lambda_rres'],
               fields_path=m['fields_path'],metrics_path=str(metric_path))
        lookup[regime,arm,seed]=r;rows.append(r)
        with np.load(m['fields_path']) as f:
            obs=f['target_raw'];mask=f['mask'];idx=f['target_index'];pred=f['pred_raw']
            if 'obs' not in checks:checks.update(obs=obs.copy(),mask=mask.copy(),idx=idx.copy())
            for k,x in [('obs',obs),('mask',mask),('idx',idx)]:np.testing.assert_array_equal(x,checks[k])
            sea=mask[0]>0
            mse=((pred[:,sea].astype(float)-obs[:,sea])**2).mean(axis=1)
            assert np.isclose(np.sqrt(mse.mean()),r['test_rmse'],atol=2e-6,rtol=0)
            daily[regime,arm,seed]=mse
    summaries=[];selection={}
    for regime in ['ratio_10','full']:
        for arm in ARMS:
            subset=[lookup[regime,arm,s] for s in SEEDS]
            r=dict(regime=regime,arm=arm)
            for k in ['best_val_rmse','test_rmse','test_acc','mhw_csi','mhw_pod','mhw_far']:
                v=np.array([s[k] for s in subset]);r[k+'_mean']=float(v.mean());r[k+'_sd']=float(v.std(ddof=1))
            summaries.append(r)
        candidates=[r for r in summaries if r['regime']==regime and r['arm']!='baseline']
        chosen=min(candidates,key=lambda r:(r['best_val_rmse_mean'],ARMS.index(r['arm'])))
        selection[regime]=dict(arm=chosen['arm'],criterion='mean best validation RMSE across three seeds',
                               best_val_rmse_mean=chosen['best_val_rmse_mean'],
                               note='aux0 is physics weight zero with auxiliary loss retained; it is not baseline')
    comparisons=[]
    for regime in ['ratio_10','full']:
        for arm in ARMS[1:]:
            for reference in ['baseline','aux0']:
                if arm==reference:continue
                diff=np.array([lookup[regime,arm,s]['test_rmse']-lookup[regime,reference,s]['test_rmse'] for s in SEEDS])
                perm=[np.mean(diff*np.array(signs)) for signs in itertools.product([-1,1],repeat=3)]
                row=dict(regime=regime,arm=arm,reference=reference,delta_rmse=float(diff.mean()),
                    delta_seed42=float(diff[0]),delta_seed7=float(diff[1]),delta_seed123=float(diff[2]),
                    improved_seed_count=int(np.sum(diff<0)),
                    exact_two_sided_p=float(np.mean(np.abs(perm)>=abs(diff.mean())-1e-14)))
                n=len(daily[regime,arm,42])
                for block in [30,90,180]:
                    rng=np.random.default_rng(20260925+7+block)
                    start=rng.integers(0,n,size=(4000,int(np.ceil(n/block))))
                    idx=((start[:,:,None]+np.arange(block))%n).reshape(4000,-1)[:,:n]
                    av=np.mean([np.sqrt(daily[regime,arm,s][idx].mean(axis=1)) for s in SEEDS],axis=0)
                    rv=np.mean([np.sqrt(daily[regime,reference,s][idx].mean(axis=1)) for s in SEEDS],axis=0)
                    lo,hi=np.quantile(av-rv,[.025,.975]);row[f'conditional_ci_{block}d_low']=float(lo);row[f'conditional_ci_{block}d_high']=float(hi)
                comparisons.append(row)
    write_csv(report_dir/'runs.csv',rows);write_csv(report_dir/'summary.csv',summaries);write_csv(report_dir/'paired_comparisons.csv',comparisons)
    (report_dir/'validation_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    md=['# Lambda pilot: complete results','',
        'Lead7; batch16; 10% block subset (376 samples) and original Full (7644 samples). Three seeds. All arms reported.',
        'Aux0 retains lambda_rres=10 and physics heads. Baseline has neither; legacy baseline/P0 DataLoader order need not be identical despite the same seed.',
        'Lambda is selected by mean validation RMSE, never by test score. Existing test results informed this follow-up design: interpretation is exploratory.',
        'Time-block intervals are conditional on the fitted models, not confidence intervals over new training seeds. Three paired seeds cannot yield exact two-sided p<0.05. Multiple comparisons are descriptive, without a family-wide significance claim.', '',
        '| Regime | Arm | Mean validation RMSE | Test RMSE mean ± SD | CSI | POD | FAR |', '|---|---|---|---|---|---|---|']
    for s in summaries:
        md.append(f"| {s['regime']} | {s['arm']} | {s['best_val_rmse_mean']:.5f} | {s['test_rmse_mean']:.5f} ± {s['test_rmse_sd']:.5f} | {s['mhw_csi_mean']:.4f} | {s['mhw_pod_mean']:.4f} | {s['mhw_far_mean']:.4f} |")
    md+=['','## Validation-selected setting','']
    for regime,s in selection.items():md.append(f"- {regime}: {s['arm']} (mean validation RMSE={s['best_val_rmse_mean']:.5f}).")
    md+=['','See paired_comparisons.csv for every seed difference, exact sign-flip p, and 30/90/180-day conditional time-block intervals. No automatic expansion is triggered by favorable test scores.','']
    (report_dir/'report.md').write_text('\n'.join(md))
    print(report_dir/'report.md',flush=True)


if __name__=='__main__':main()
