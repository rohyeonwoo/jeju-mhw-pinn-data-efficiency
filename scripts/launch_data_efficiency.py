"""Three-GPU queue for the 48 fixed SeaCast/P0 data-efficiency runs."""
from __future__ import annotations
import argparse
import datetime
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
OUTPUT=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')


def save(path,payload):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0,1,2')
    args=parser.parse_args()
    gpus=args.gpus.split(',')
    assert len(gpus)==len(set(gpus))==3
    jobs=[]
    for ratio in (10,25,50,100):
        for lead in (7,14):
            for mode in ('none','p0_original'):
                for seed in (42,7,123):
                    model='jeju_hilam' if mode=='none' else 'pi_jeju_hilam'
                    run=OUTPUT/f'ratio_{ratio}'/model/f'lead{lead}_seed{seed}'
                    if run.exists():
                        raise RuntimeError(f'Refusing to overwrite existing run: {run}')
                    jobs.append(dict(id=len(jobs),ratio=ratio,lead=lead,mode=mode,seed=seed,
                                     run_dir=str(run),status='pending'))
    logs=OUTPUT/'_queue'
    logs.mkdir(parents=True,exist_ok=True)
    state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'pid':os.getpid(),'gpus':gpus,'total_runs':len(jobs),'jobs':jobs,
           'python':sys.executable,'workspace':str(ROOT),
           'runner_sha256':hashlib.sha256((ROOT/'scripts/train_data_efficiency.py').read_bytes()).hexdigest()}
    manifest=logs/'batch_status.json'
    if manifest.exists():
        raise RuntimeError(f'Queue already exists: {manifest}')
    save(manifest,state)
    lock=threading.Lock()
    stop=threading.Event()
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    def worker(gpu):
        while not stop.is_set():
            try:job=pending.get_nowait()
            except queue.Empty:return
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(job['seed']),
                       MHW_PROCESS_NAME='yeonwoo',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),
                     str(ROOT/'scripts/train_data_efficiency.py'),'--lead',str(job['lead']),
                     '--seed',str(job['seed']),'--ratio',str(job['ratio']),
                     '--physics-mode',job['mode']]
            log=logs/f"job_{job['id']:02d}_gpu{gpu}.log"
            try:
                with log.open('x') as f:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                    with lock:
                        job.update(status='running',gpu=gpu,pid=child.pid,log=str(log),started_unix=time.time())
                        save(manifest,state)
                        print(json.dumps(job),flush=True)
                    result=child.wait()
                progress=Path(job['run_dir'])/'progress.json'
                done=(result==0 and progress.is_file() and json.loads(progress.read_text())['status']=='completed')
                with lock:
                    job.update(status='completed' if done else 'failed',returncode=result,finished_unix=time.time())
                    if not done:stop.set()
                    save(manifest,state)
                    print(json.dumps(job),flush=True)
            except Exception as error:
                with lock:
                    job.update(status='failed',error=repr(error))
                    stop.set()
                    save(manifest,state)
            finally:pending.task_done()
    threads=[threading.Thread(target=worker,args=(gpu,)) for gpu in gpus]
    for t in threads:t.start()
    for t in threads:t.join()
    if any(j['status']!='completed' for j in jobs):
        state['status']='stopped_after_failure'
        save(manifest,state)
        raise SystemExit('A run failed; no new jobs were started after failure. See batch_status.json.')
    state['status']='evaluating_mhw'
    save(manifest,state)
    data=next((Path('/ceph_data/yeonwoo/MHW_subset')/'data').glob('*.nc'))
    for ratio in (10,25,50,100):
        base=OUTPUT/f'ratio_{ratio}'
        command=[sys.executable,'-B',str(ROOT/'scripts/run_as_yeonwoo.py'),
                 str(ROOT/'src/summarize_jeju_hilam_mhw_skill.py'),'--data-path',str(data),
                 '--roots',str(base/'jeju_hilam'),str(base/'pi_jeju_hilam'),
                 '--leads','7,14','--seeds','42,7,123','--output-prefix',str(base/'mhw_skill')]
        with (logs/f'mhw_ratio_{ratio}.log').open('x') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='mhw_evaluation_failed',failed_ratio=ratio)
            save(manifest,state)
            raise SystemExit(result.returncode)
    state.update(status='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(manifest,state)


if __name__=='__main__':
    main()
