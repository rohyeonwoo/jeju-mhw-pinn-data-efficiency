"""Validate all pilot GPU arms and reproduce the unchanged lambda=.1 runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import setproctitle
setproctitle.setproctitle('yeonwoo')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    cpu=json.loads((ROOT/'docs/server_preflight/lambda_pilot_cpu_check.json').read_text())
    assert cpu['status']=='passed'
    out=a.output_root.resolve()
    assert out.is_relative_to(BASE)
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED='42',MHW_PROCESS_NAME='yeonwoo',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    checks=[]
    for regime in ['ratio_10','full']:
        for lam in [0.,.01,.3]:
            tag=format(lam,'.6g').replace('.','p')
            log=out/f'{regime}_lambda_{tag}.log'
            cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),str(ROOT/'scripts/train_lambda_pilot.py'),
                 '--lead','7','--seed','42','--lambda-physics',str(lam),'--batch-size','16','--smoke','--output-root',str(out/'new')]
            cmd+=['--full'] if regime=='full' else ['--ratio','10']
            with log.open('x') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            d=out/'new/_smoke'/f'lambda_{tag}'/regime/'pi_jeju_hilam/lead7_seed42'
            progress=json.loads((d/'progress.json').read_text())
            config=json.loads((d/'jeju_hilam_config.json').read_text())
            assert progress['status']=='completed' and progress['step']==4
            assert config['lambda_physics']==lam and config['lambda_rres']==10
            assert config['S']==478 and config['max_steps']==38240
            checks.append(dict(regime=regime,lambda_physics=lam,run_dir=str(d),status='passed'))
            print('SMOKE PASSED',regime,lam,flush=True)
    dirs=[]
    for new in [False,True]:
        d=out/('compat_new' if new else 'compat_legacy')
        cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),
             str(ROOT/'scripts'/('train_lambda_pilot.py' if new else 'train_data_efficiency.py')),
             '--lead','7','--seed','42','--ratio','10','--physics-mode','p0_original',
             '--batch-size','16','--smoke','--output-root',str(d)]
        if new:cmd+=['--lambda-physics','0.1']
        with (out/f'compat_{new}.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        dirs.append(d/'_smoke'/('lambda_0p1/ratio_10' if new else 'ratio_10')/'pi_jeju_hilam/lead7_seed42')
    old,new=[torch.load(d/'jeju_hilam_best.pt',map_location='cpu',weights_only=False) for d in dirs]
    assert old['best_step']==new['best_step']
    max_diff=0.
    for k,v in old['model_state'].items():
        torch.testing.assert_close(v,new['model_state'][k],rtol=1e-5,atol=1e-6)
        if v.is_floating_point():max_diff=max(max_diff,float((v-new['model_state'][k]).abs().max()))
    arrays=[np.load(next(d.glob('*test_fields.npz'))) for d in dirs]
    for k in ['target_index','target_raw','mask']:np.testing.assert_array_equal(arrays[0][k],arrays[1][k])
    np.testing.assert_allclose(arrays[0]['pred_raw'],arrays[1]['pred_raw'],rtol=0,atol=2e-5)
    report=dict(status='passed',runner_sha256=hashlib.sha256((ROOT/'scripts/train_lambda_pilot.py').read_bytes()).hexdigest(),
                cpu_report=str(ROOT/'docs/server_preflight/lambda_pilot_cpu_check.json'),smoke_checks=checks,
                compatibility=dict(lambda_physics=.1,legacy_run_dir=str(dirs[0]),new_run_dir=str(dirs[1]),
                                   max_parameter_difference=max_diff,max_prediction_difference=float(np.max(np.abs(arrays[0]['pred_raw']-arrays[1]['pred_raw'])))),
                gpu='0',batch_size=16,process_name='yeonwoo')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print('VALIDATION PASSED',a.report,flush=True)


if __name__=='__main__':main()
