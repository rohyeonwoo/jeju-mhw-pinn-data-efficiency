"""Adopt running training processes; dynamically schedule up to 3 jobs per GPU.

The old coordinator must be SIGSTOP'ed before adoption. Training children are
never stopped, signalled or restarted. A slot reduction only limits new jobs.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import setproctitle
setproctitle.setproctitle('yeonwoo')
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')
QUEUE=OUTPUT/'_queue'


def save(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def proc(pid):
    try:
        data=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':data[0],'ppid':int(data[1]),'start_ticks':int(data[19])}
    except FileNotFoundError:
        return None


def alive(pid,start_ticks):
    current=proc(pid)
    return current is not None and current['start_ticks']==start_ticks and current['state']!='Z'


def read_slots(path,gpus):
    value=json.loads(path.read_text())
    result=value['slots_per_gpu']
    if set(result)!=set(gpus) or any(type(v) is not int or not 1<=v<=3 for v in result.values()):
        raise ValueError('slots_per_gpu must contain the same three GPUs with integer values 1..3')
    return result


def next_assignments(jobs,slots):
    pending=iter(j for j in jobs if j['status']=='pending')
    result=[]
    for gpu,limit in slots.items():
        count=sum(j['status']=='running' and j.get('gpu')==gpu for j in jobs)
        for _ in range(max(0,limit-count)):
            job=next(pending,None)
            if job is None:return result
            result.append((gpu,job))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-pid',type=int,required=True)
    args=parser.parse_args()
    with (QUEUE/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=QUEUE/'batch_status.json'
        state=json.loads(manifest.read_text())
        old=proc(args.adopt_pid)
        assert state['pid']==args.adopt_pid and old and old['state']=='T', 'Stop only the old coordinator first'
        assert state['status']=='running'
        backup=QUEUE/f'batch_status_before_adoption_{args.adopt_pid}.json'
        with backup.open('x') as f:json.dump(state,f,indent=2)
        jobs=state['jobs']
        assert len(jobs)==len({j['id'] for j in jobs})==48
        old_ids=[]
        for j in jobs:
            if j['status']=='running':
                p=proc(j['pid'])
                assert p and p['ppid']==args.adopt_pid, (j,p)
                j.update(adopted=True,process_start_ticks=p['start_ticks'])
                old_ids.append(j['id'])
        control=QUEUE/'concurrency.json'
        slots=read_slots(control,state['gpus'])
        state.update(pid=os.getpid(),previous_coordinator_pid=args.adopt_pid,
                     previous_coordinator_start_ticks=old['start_ticks'],
                     adopted_job_ids=old_ids,slots_per_gpu=slots,
                     scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     scheduler='manage_data_efficiency.py',adopted_unix=time.time())
        save(manifest,state)
        children={}
        dispatch=True
        retired=False
        while True:
            changed=False
            for j in jobs:
                if j['status']!='running':continue
                child=children.get(j['id'])
                running=child.poll() is None if child is not None else alive(j['pid'],j['process_start_ticks'])
                if running:continue
                path=Path(j['run_dir'])/'progress.json'
                progress=json.loads(path.read_text()) if path.exists() else {}
                code=child.returncode if child is not None else None
                completed=progress.get('status')=='completed' and (child is None or code==0)
                j.update(status='completed' if completed else 'failed',returncode=code,
                         finished_unix=time.time())
                if not completed:dispatch=False
                changed=True
                print(json.dumps(j),flush=True)
            # Once all adopted children have exited, their stopped parent is no
            # longer needed. SIGTERM is queued before SIGCONT, so it cannot dispatch.
            if not retired and all(jobs[i]['status']!='running' for i in old_ids):
                p=proc(args.adopt_pid)
                if p and p['start_ticks']==old['start_ticks']:
                    assert p['state']=='T'
                    os.kill(args.adopt_pid,signal.SIGTERM)
                    os.kill(args.adopt_pid,signal.SIGCONT)
                retired=True
                state['previous_coordinator_retired']=True
                changed=True
            try:
                requested=read_slots(control,state['gpus'])
            except (ValueError,KeyError,json.JSONDecodeError) as error:
                requested=slots
                print('Invalid concurrency control; keeping previous slots:',str(error),flush=True)
            if requested!=slots:
                slots=requested
                state['slots_per_gpu']=slots
                changed=True
            if dispatch:
                for gpu,j in next_assignments(jobs,slots):
                    run=Path(j['run_dir'])
                    log=QUEUE/f"job_{j['id']:02d}_gpu{gpu}.log"
                    try:
                        assert not run.exists(),f'Refusing existing run {run}'
                        env=os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(j['seed']),
                                   MHW_PROCESS_NAME='yeonwoo',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                                   OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
                        command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),
                                 str(ROOT/'scripts/train_data_efficiency.py'),'--lead',str(j['lead']),
                                 '--seed',str(j['seed']),'--ratio',str(j['ratio']),'--physics-mode',j['mode']]
                        with log.open('x') as f:
                            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                        children[j['id']]=child
                        j.update(status='running',gpu=gpu,pid=child.pid,log=str(log),started_unix=time.time())
                        print(json.dumps(j),flush=True)
                    except Exception as error:
                        j.update(status='failed',error=repr(error))
                        dispatch=False
                    changed=True
                    save(manifest,state)
                    if not dispatch:break
            if changed:
                state['dispatch_enabled']=dispatch
                state['updated_unix']=time.time()
                save(manifest,state)
            active=any(j['status']=='running' for j in jobs)
            pending=any(j['status']=='pending' for j in jobs)
            if not active and (not pending or not dispatch):break
            time.sleep(2)
        if any(j['status']!='completed' for j in jobs):
            state['status']='stopped_after_failure'
            save(manifest,state)
            raise SystemExit('Run failure; pending jobs were not started.')
        state['status']='evaluating_mhw'
        save(manifest,state)
        data=next((OUTPUT.parents[1]/'data').glob('*.nc'))
        for ratio in (10,25,50,100):
            base=OUTPUT/f'ratio_{ratio}'
            command=[sys.executable,'-B',str(ROOT/'scripts/run_as_yeonwoo.py'),
                     str(ROOT/'src/summarize_jeju_hilam_mhw_skill.py'),'--data-path',str(data),
                     '--roots',str(base/'jeju_hilam'),str(base/'pi_jeju_hilam'),
                     '--leads','7,14','--seeds','42,7,123','--output-prefix',str(base/'mhw_skill')]
            with (QUEUE/f'mhw_ratio_{ratio}.log').open('x') as f:
                result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(status='mhw_evaluation_failed',failed_ratio=ratio)
                save(manifest,state)
                raise SystemExit(result.returncode)
        state.update(status='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(manifest,state)


if __name__=='__main__':main()
