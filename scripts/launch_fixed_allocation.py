"""User's 5+4 GPU allocation, followed by 100% runs on freed slots."""
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
BASE=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')
LAYOUT=[('0',10,42),('0',10,7),('0',10,123),('0',25,42),('0',25,7),
        ('1',25,123),('1',50,42),('1',50,7),('1',50,123)]


def save(path,data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def plan(output):
    jobs=[]
    groups=[]
    def add(ratio,seed,lead,mode,slot=None,gpu=None):
        family='jeju_hilam' if mode=='none' else 'pi_jeju_hilam'
        j=dict(id=len(jobs),ratio=ratio,seed=seed,lead=lead,mode=mode,status='pending',
               run_dir=str(output/f'ratio_{ratio}'/family/f'lead{lead}_seed{seed}'))
        if slot is not None:j.update(slot=slot,gpu=gpu)
        jobs.append(j)
        return j
    for slot,(gpu,ratio,seed) in enumerate(LAYOUT):
        local=[]
        for lead in (7,14):
            for mode in ('none','p0_original'):
                local.append(add(ratio,seed,lead,mode,slot,gpu))
        groups.append(local)
    remaining=[]
    for lead in (7,14):
        for mode in ('none','p0_original'):
            for seed in (42,7,123):remaining.append(add(100,seed,lead,mode))
    assert len(jobs)==48 and len({(j['ratio'],j['seed'],j['lead'],j['mode']) for j in jobs})==48
    return jobs,groups,remaining


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_root.resolve()
    assert output.is_relative_to(BASE) and output!=BASE
    output.mkdir(parents=True,exist_ok=False)
    logs=output/'_queue';logs.mkdir()
    jobs,groups,remaining=plan(output)
    work=queue.Queue()
    for j in remaining:work.put(j)
    state=dict(status='running',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               output_root=str(output),workspace=str(ROOT),python=sys.executable,total_runs=48,
               slots=[dict(slot=i,gpu=g,ratio=r,seed=s) for i,(g,r,s) in enumerate(LAYOUT)],
               slots_per_gpu={'0':5,'1':4},jobs=jobs,
               sequence='Each reserved ratio/seed: lead7 baseline, lead7 P0, lead14 baseline, lead14 P0; then shared 100% queue',
               prior_attempt=str(BASE),restart_reason='User requested stop and restart with GPU0 five / GPU1 four',
               runner_sha256=hashlib.sha256((ROOT/'scripts/train_data_efficiency.py').read_bytes()).hexdigest(),
               scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest=logs/'batch_status.json'
    save(manifest,state)
    save(BASE/'current_attempt.json',{'output_root':str(output),'manifest':str(manifest),'pid':os.getpid()})
    lock=threading.Lock();stop=threading.Event()
    def worker(slot):
        gpu=LAYOUT[slot][0]
        local=iter(groups[slot])
        while True:
            with lock:
                if stop.is_set():return
                job=next(local,None)
                if job is None:
                    try:job=work.get_nowait()
                    except queue.Empty:return
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(job['seed']),
                           MHW_PROCESS_NAME='yeonwoo',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),
                         str(ROOT/'scripts/train_data_efficiency.py'),'--lead',str(job['lead']),
                         '--seed',str(job['seed']),'--ratio',str(job['ratio']),'--physics-mode',job['mode'],
                         '--output-root',str(output)]
                log=logs/f"job_{job['id']:02d}_gpu{gpu}_slot{slot}.log"
                try:
                    assert not Path(job['run_dir']).exists()
                    with log.open('x') as f:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                    job.update(status='running',gpu=gpu,slot=slot,pid=child.pid,log=str(log),started_unix=time.time())
                    save(manifest,state)
                    print(json.dumps(job),flush=True)
                except Exception as error:
                    job.update(status='failed',error=repr(error));stop.set();save(manifest,state)
                    return
            result=child.wait()
            path=Path(job['run_dir'])/'progress.json'
            progress=json.loads(path.read_text()) if path.exists() else {}
            done=result==0 and progress.get('status')=='completed'
            with lock:
                job.update(status='completed' if done else 'failed',returncode=result,finished_unix=time.time())
                if not done:stop.set()
                save(manifest,state)
                print(json.dumps(job),flush=True)
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(9)]
    for t in threads:t.start()
    for t in threads:t.join()
    if any(j['status']!='completed' for j in jobs):
        state['status']='stopped_after_failure';save(manifest,state)
        raise SystemExit('A training run failed. New dispatch stopped; see job logs.')
    state['status']='evaluating_mhw';save(manifest,state)
    data=next((BASE.parents[1]/'data').glob('*.nc'))
    for ratio in (10,25,50,100):
        root=output/f'ratio_{ratio}'
        command=[sys.executable,'-B',str(ROOT/'scripts/run_as_yeonwoo.py'),
                 str(ROOT/'src/summarize_jeju_hilam_mhw_skill.py'),'--data-path',str(data),
                 '--roots',str(root/'jeju_hilam'),str(root/'pi_jeju_hilam'),
                 '--leads','7,14','--seeds','42,7,123','--output-prefix',str(root/'mhw_skill')]
        with (logs/f'mhw_ratio_{ratio}.log').open('x') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='mhw_evaluation_failed',failed_ratio=ratio);save(manifest,state)
            raise SystemExit(result.returncode)
    state.update(status='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(manifest,state)


if __name__=='__main__':main()
