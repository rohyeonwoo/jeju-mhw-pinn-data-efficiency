"""Launch the authorized 30 controlled runs with four slots per GPU, then evaluate all outcomes."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import setproctitle
setproctitle.setproctitle('yeonwoo')
from validate_physics_pilot import ROOT, BASE, hashes
from launch_lambda_pilot import save, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--validation-report',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve();assert out.is_relative_to(BASE) and out!=BASE
    v=json.loads(a.validation_report.read_text());protocol=json.loads(a.protocol.read_text())
    assert v['status']=='passed' and v['scripts_sha256']==hashes()
    assert protocol['status']=='authorized' and protocol['total_runs']==30
    assert protocol['scripts_sha256']=={name:sha(ROOT/'scripts'/name) for name in protocol['scripts_sha256']}
    lease=(BASE/'physics_pilot_dispatch.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={r.split(',')[0].strip():int(r.split(',')[1]) for r in rows.splitlines()}
    assert all(free[g]>44500 for g in ['0','1']),free
    out.mkdir(parents=True,exist_ok=False);logs=out/'_queue';logs.mkdir()
    jobs=[]
    arms=[('matched_baseline',0.)]+[(arm,lam) for lam in [.01,.1,.3] for arm in ['p0_daily','p0_interval','p1_daily']]
    for arm,lam in arms:
        tag=format(lam,'.6g').replace('.','p')
        for seed in [42,7,123]:
            jobs.append(dict(id=len(jobs),arm=arm,lambda_physics=lam,lambda_rres=0.,seed=seed,lead=7,
                status='pending',run_dir=str(out/arm/f'lambda_{tag}/ratio_10/pi_jeju_hilam/lead7_seed{seed}')))
    assert len(jobs)==30
    state=dict(status='running',pid=os.getpid(),output_root=str(out),total_runs=30,jobs=jobs,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),slots_per_gpu={'0':4,'1':4},
        scripts_sha256=hashes(),protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),
        validation_report=str(a.validation_report.resolve()),physics_stencil='ocean_neighbors',
        scope='ratio10 lead7 3 seeds; matched baseline plus P0 daily/P0 period-mean/P1 daily x lambda .01/.1/.3',
        extension_policy='No automatic expansion; all outcomes reported')
    manifest=logs/'batch_status.json';save(manifest,state);save(logs/'protocol.json',protocol)
    save(BASE/'physics_pilot_current.json',dict(output_root=str(out),manifest=str(manifest),pid=os.getpid()))
    save(BASE/'current_attempt.json',dict(output_root=str(out),manifest=str(manifest),pid=os.getpid(),phase='physics_form_pilot'))
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    lock=threading.Lock();stop=threading.Event()
    def worker(slot):
        gpu=str(slot//4)
        while True:
            with lock:
                if stop.is_set():return
                try:j=pending.get_nowait()
                except queue.Empty:return
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(j['seed']),
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
                cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),str(ROOT/'scripts/train_physics_pilot.py'),
                     '--lead','7','--seed',str(j['seed']),'--ratio','10','--arm',j['arm'],
                     '--lambda-physics',str(j['lambda_physics']),'--physics-stencil','ocean_neighbors',
                     '--batch-size','16','--output-root',str(out)]
                log=logs/f"job_{j['id']:02d}_gpu{gpu}.log"
                try:
                    assert hashes()==state['scripts_sha256']
                    assert not Path(j['run_dir']).exists()
                    with log.open('x') as f:child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                    j.update(status='running',gpu=gpu,slot=slot,pid=child.pid,log=str(log),started_unix=time.time())
                    save(manifest,state);print(json.dumps(j),flush=True)
                except Exception as error:
                    j.update(status='failed',error=repr(error));stop.set();save(manifest,state);return
            code=child.wait();p=Path(j['run_dir'])/'progress.json'
            progress=json.loads(p.read_text()) if p.exists() else {}
            done=code==0 and progress.get('status')=='completed'
            with lock:
                j.update(status='completed' if done else 'failed',returncode=code,finished_unix=time.time())
                if not done:stop.set()
                save(manifest,state);print(json.dumps(j),flush=True)
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(8)]
    for t in threads:t.start()
    for t in threads:t.join()
    if not all(j['status']=='completed' for j in jobs):
        state['status']='stopped_after_failure';save(manifest,state);raise SystemExit(1)
    state['status']='evaluating_mhw';save(manifest,state)
    data=next((BASE.parents[1]/'data').glob('*.nc'))
    for arm,lam in arms:
        tag=format(lam,'.6g').replace('.','p');root=out/arm/f'lambda_{tag}/ratio_10'
        cmd=[sys.executable,'-B',str(ROOT/'scripts/run_as_yeonwoo.py'),str(ROOT/'src/summarize_jeju_hilam_mhw_skill.py'),
             '--data-path',str(data),'--roots',str(root/'pi_jeju_hilam'),'--leads','7','--seeds','42,7,123',
             '--output-prefix',str(root/'mhw_skill')]
        with (logs/f'mhw_{arm}_{tag}.log').open('x') as f:
            result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='mhw_evaluation_failed',failed_arm=arm,failed_lambda=lam)
            save(manifest,state);raise SystemExit(result.returncode)
    state['status']='analyzing';save(manifest,state)
    with (logs/'analysis.log').open('x') as f:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/summarize_physics_pilot.py'),
                               '--output-root',str(out)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    state.update(status='completed' if result.returncode==0 else 'analysis_failed',
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(manifest,state)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
