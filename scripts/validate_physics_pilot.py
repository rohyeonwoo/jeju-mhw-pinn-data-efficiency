"""GPU smoke validation for the authorized 30-run ocean-neighbor pilot."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import setproctitle
setproctitle.setproctitle('yeonwoo')
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')
MODULES=['train_physics_pilot.py','physics_interval.py','physics_stencil.py','physics_pilot_checks.py']


def hashes():
    return {n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in MODULES}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    out=a.output_root.resolve();assert out.is_relative_to(BASE)
    for name in ['physics_interval_cpu_check','ocean_neighbor_gradient_check','ocean_neighbors_rhs_lead7']:
        assert json.loads((ROOT/f'docs/server_preflight/{name}.json').read_text())['status']=='passed'
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED='42',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    captured=hashes();results=[];reference=None;test_reference=None
    import numpy as np
    for arm,lam in [('matched_baseline',0.),('p0_daily',.3),('p0_interval',.3),('p1_daily',.3)]:
        cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),str(ROOT/'scripts/train_physics_pilot.py'),
             '--lead','7','--seed','42','--ratio','10','--arm',arm,'--lambda-physics',str(lam),
             '--physics-stencil','ocean_neighbors','--batch-size','16','--smoke','--output-root',str(out/'runs')]
        with (out/f'{arm}.log').open('x') as f:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        tag=format(lam,'.6g').replace('.','p')
        d=out/'runs/_smoke'/arm/f'lambda_{tag}/ratio_10/pi_jeju_hilam/lead7_seed42'
        progress=json.loads((d/'progress.json').read_text())
        c=json.loads((d/'jeju_hilam_config.json').read_text())
        m=json.loads(next(d.glob('*_metrics.json')).read_text())
        assert progress['status']=='completed' and progress['step']==4
        assert c['lambda_rres']==0 and c['physics_stencil']=='ocean_neighbors'
        assert c['rhs_scale_report']['valid_stencil_cells']==660
        assert c['S']==478 and c['max_steps']==38240 and c['test_samples']==1639
        rng=[c['initial_state_sha256'],c['initial_cpu_rng_sha256'],m['first_128_train_batches_sha256']]
        if reference is None:reference=rng
        assert rng==reference,(arm,'initialization or training batch order differs')
        with (d/'jeju_hilam_epoch_history.csv').open() as f:
            for row in csv.DictReader(f):
                expected=float(row['train_data_loss'])+float(row['train_lambda_physics_effective'])*float(row['train_physics_loss'])
                assert np.isclose(float(row['train_total_loss']),expected,rtol=1e-5,atol=1e-5)
        with np.load(m['fields_path']) as z:
            values=[z[k].copy() for k in ['target_index','target_raw','mask']]
            if test_reference is None:test_reference=values
            for got,want in zip(values,test_reference):np.testing.assert_array_equal(got,want)
            assert int((z['mask'][0]>0).sum())==661
            assert np.isfinite(z['pred_raw']).all()
        results.append(dict(arm=arm,lambda_physics=lam,status='passed',run_dir=str(d),
                            initial_gradient=c['initial_gradient_diagnostic']))
        print('SMOKE PASSED',arm,flush=True)
    assert captured==hashes()
    result=dict(status='passed',scripts_sha256=captured,smoke_checks=results,
                initialization_and_batch_order_matched=True,evaluation_cells=661,physics_cells=660,
                rhs_scale=results and c['physics_rhs_scale'],process_name='yeonwoo',batch_size=16)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print('VALIDATION PASSED',a.report,flush=True)


if __name__=='__main__':main()
