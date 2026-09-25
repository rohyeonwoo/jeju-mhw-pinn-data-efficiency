"""Run only the authorized 18-run P0 lambda pilot on GPU 0/1, five slots each."""
import argparse
import datetime
import fcntl
import hashlib
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
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')


def save(path,data):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--validation-report',type=Path,required=True)
    args=p.parse_args()
    out=args.output_root.resolve()
    assert out.is_relative_to(BASE) and out!=BASE
    validation=json.loads(args.validation_report.read_text())
    assert validation['status']=='passed'
    runner=ROOT/'scripts/train_lambda_pilot.py'
    assert validation['runner_sha256']==sha(runner)
    lease=(BASE/'lambda_pilot_dispatch.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Check availability once before the ten workers create any CUDA context.
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={r.split(',')[0].strip():int(r.split(',')[1]) for r in memory.splitlines()}
    assert all(free[g]>44500 for g in ['0','1']),free
    out.mkdir(parents=True,exist_ok=False)
    logs=out/'_queue';logs.mkdir()
    jobs=[]
    # Long Full runs first, to avoid a large Full-only tail after short subsets.
    for regime in ['full','ratio_10']:
        for lam in [0.,.01,.3]:
            for seed in [42,7,123]:
                tag=format(lam,'.6g').replace('.','p')
                jobs.append(dict(id=len(jobs),regime=regime,lead=7,seed=seed,lambda_physics=lam,
                    lambda_rres=10.,status='pending',
                    run_dir=str(out/f'lambda_{tag}'/regime/'pi_jeju_hilam'/f'lead7_seed{seed}')))
    assert len(jobs)==18
    state=dict(status='running',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        output_root=str(out),total_runs=18,batch_size=16,slots_per_gpu={'0':5,'1':5},jobs=jobs,
        runner_sha256=sha(runner),scheduler_sha256=sha(Path(__file__)),
        validation_report=str(args.validation_report.resolve()),
        scope='lead7, ratio10 and original Full, 3 seeds; lambda_phys 0/.01/.3, lambda_rres=10',
        extension_policy='Stop after pilot and report all outcomes; no automatic expansion selected by positive test results',
        baseline_reference=str(BASE/'allocation_batch16_5plus5'),full_reference=str(BASE/'full_batch16_matched'),
        selection='Per regime choose one lambda among 0/.01/.1/.3 minimizing mean best validation RMSE across 3 seeds; exact ties prefer smaller lambda; lambda0 retains auxiliary loss')
    manifest=logs/'batch_status.json';save(manifest,state)
    save(BASE/'lambda_pilot_current.json',dict(output_root=str(out),manifest=str(manifest),pid=os.getpid()))
    save(BASE/'current_attempt.json',dict(output_root=str(out),manifest=str(manifest),pid=os.getpid(),phase='lambda_pilot'))
    work=queue.Queue()
    for job in jobs:work.put(job)
    lock=threading.Lock();stop=threading.Event()
    def worker(slot):
        gpu=str(slot//5)
        while True:
            with lock:
                if stop.is_set():return
                try:j=work.get_nowait()
                except queue.Empty:return
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(j['seed']),MHW_PROCESS_NAME='yeonwoo',
                           OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
                cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),str(runner),
                     '--lead','7','--seed',str(j['seed']),'--lambda-physics',str(j['lambda_physics']),
                     '--output-root',str(out),'--batch-size','16']
                cmd+=['--full'] if j['regime']=='full' else ['--ratio','10']
                log=logs/f"job_{j['id']:02d}_gpu{gpu}_slot{slot}.log"
                try:
                    assert sha(runner)==state['runner_sha256']
                    assert not Path(j['run_dir']).exists()
                    with log.open('x') as f:
                        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                    j.update(status='running',gpu=gpu,slot=slot,pid=child.pid,log=str(log),started_unix=time.time())
                    save(manifest,state);print(json.dumps(j),flush=True)
                except Exception as error:
                    j.update(status='failed',error=repr(error));stop.set();save(manifest,state);return
            result=child.wait()
            progress_path=Path(j['run_dir'])/'progress.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            done=result==0 and progress.get('status')=='completed'
            with lock:
                j.update(status='completed' if done else 'failed',returncode=result,finished_unix=time.time())
                if not done:stop.set()
                save(manifest,state);print(json.dumps(j),flush=True)
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(10)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    if not all(j['status']=='completed' for j in jobs):
        state['status']='stopped_after_failure';save(manifest,state);raise SystemExit(1)
    state['status']='evaluating_mhw';save(manifest,state)
    data=next((BASE.parents[1]/'data').glob('*.nc'))
    for lam in ['0','0p01','0p3']:
        for regime in ['ratio_10','full']:
            d=out/f'lambda_{lam}'/regime
            cmd=[sys.executable,'-B',str(ROOT/'scripts/run_as_yeonwoo.py'),str(ROOT/'src/summarize_jeju_hilam_mhw_skill.py'),
                 '--data-path',str(data),'--roots',str(d/'pi_jeju_hilam'),'--leads','7','--seeds','42,7,123',
                 '--output-prefix',str(d/'mhw_skill')]
            with (logs/f'mhw_{lam}_{regime}.log').open('x') as f:
                result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(status='mhw_evaluation_failed',failed_lambda=lam,failed_regime=regime)
                save(manifest,state);raise SystemExit(result.returncode)
    state['status']='analyzing';save(manifest,state)
    with (logs/'analysis.log').open('x') as f:
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/summarize_lambda_pilot.py'),'--output-root',str(out)],
                              cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    state.update(status='completed' if result.returncode==0 else 'analysis_failed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(manifest,state)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
