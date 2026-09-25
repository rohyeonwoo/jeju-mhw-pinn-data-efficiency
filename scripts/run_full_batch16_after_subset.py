"""Wait for the batch16 subset study, then run the matched original-Full reference."""
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


def save(path,data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def plan(output):
    jobs=[]
    for lead in (7,14):
        for mode in ('none','p0_original'):
            family='jeju_hilam' if mode=='none' else 'pi_jeju_hilam'
            for seed in (42,7,123):
                jobs.append(dict(id=len(jobs),lead=lead,seed=seed,mode=mode,data_regime='full',
                                 train_samples=7644 if lead==7 else 7639,status='pending',
                                 run_dir=str(output/'full'/family/f'lead{lead}_seed{seed}')))
    return jobs


def environment(gpu,seed):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED=str(seed),MHW_PROCESS_NAME='yeonwoo',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    return env


def command(output,lead,seed,mode,smoke=False):
    result=[sys.executable,'-B','-u',str(ROOT/'scripts/run_as_yeonwoo.py'),
            str(ROOT/'scripts/train_data_efficiency.py'),'--full','--lead',str(lead),
            '--seed',str(seed),'--physics-mode',mode,'--batch-size','16','--output-root',str(output)]
    return result+['--smoke'] if smoke else result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    dependency=args.wait_for.resolve();output=args.output_root.resolve()
    assert dependency.is_relative_to(BASE) and output.is_relative_to(BASE) and output!=BASE
    upstream=json.loads(dependency.read_text())
    assert upstream['batch_size']==16 and upstream['total_runs']==48
    output.mkdir(parents=True,exist_ok=False)
    logs=output/'_queue';logs.mkdir()
    manifest=logs/'batch_status.json';jobs=plan(output)
    runner=ROOT/'scripts/train_data_efficiency.py'
    runner_hash=hashlib.sha256(runner.read_bytes()).hexdigest()
    state=dict(status='waiting_for_subset',pid=os.getpid(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               dependency_manifest=str(dependency),output_root=str(output),jobs=jobs,total_runs=12,
               batch_size=16,gpus=['0','1'],slots_per_gpu={'0':5,'1':5},S={'7':478,'14':478},
               max_steps=38240,warmup_steps=2390,runner_sha256=runner_hash,
               reason='Original Full reference matched to all batch16 subset conditions; 100% block subset remains separate')
    save(manifest,state)
    save(BASE/'full_followup.json',{'manifest':str(manifest),'dependency':str(dependency),'pid':os.getpid()})
    print('Waiting for all 48 subset runs and their evaluation to complete:',dependency,flush=True)
    last=None
    while True:
        upstream=json.loads(dependency.read_text())
        summary=(upstream['status'],sum(j['status']=='completed' for j in upstream['jobs']))
        if summary!=last:
            state.update(dependency_status=summary[0],dependency_completed=summary[1],updated_unix=time.time())
            save(manifest,state);last=summary
        if upstream['status']=='completed':
            assert all(j['status']=='completed' for j in upstream['jobs'])
            break
        if upstream['status'] not in ('running','evaluating_mhw') or any(j['status']=='failed' for j in upstream['jobs']):
            state['status']='blocked_by_subset_status';save(manifest,state)
            raise SystemExit('Subset study was stopped or failed; no Full training started.')
        time.sleep(30)
    if hashlib.sha256(runner.read_bytes()).hexdigest()!=runner_hash:
        state['status']='blocked_runner_changed';save(manifest,state)
        raise SystemExit('Training runner changed after scheduling; review before Full starts.')
    # Run GPU checks only after the upstream GPU jobs have finished.
    state['status']='validating_full';save(manifest,state)
    for lead in (7,14):
        for mode in ('none','p0_original'):
            with (logs/f'smoke_lead{lead}_{mode}.log').open('x') as f:
                result=subprocess.run(command(output/'_validation',lead,42,mode,smoke=True),cwd=ROOT,
                                      env=environment('0',42),stdout=f,stderr=subprocess.STDOUT)
            family='jeju_hilam' if mode=='none' else 'pi_jeju_hilam'
            smoke_dir=output/'_validation/_smoke/full'/family/f'lead{lead}_seed42'
            if result.returncode:
                state.update(status='full_smoke_failed',failed_lead=lead,failed_mode=mode);save(manifest,state)
                raise SystemExit(result.returncode)
            config=json.loads((smoke_dir/'jeju_hilam_config.json').read_text())
            assert config['train_samples']==(7644 if lead==7 else 7639)
            assert config['data_regime']=='full' and config['subset_ratio'] is None
            assert config['batch_size']==16 and config['S']==478
    state.update(status='running',started_unix=time.time());save(manifest,state)
    save(BASE/'current_attempt.json',{'output_root':str(output),'manifest':str(manifest),'pid':os.getpid(),'phase':'matched_full_batch16'})
    pending=queue.Queue()
    for j in jobs:pending.put(j)
    lock=threading.Lock();stop=threading.Event()
    def worker(gpu,slot):
        while True:
            with lock:
                if stop.is_set():return
                try:j=pending.get_nowait()
                except queue.Empty:return
                log=logs/f"job_{j['id']:02d}_gpu{gpu}_slot{slot}.log"
                try:
                    assert not Path(j['run_dir']).exists()
                    with log.open('x') as f:
                        child=subprocess.Popen(command(output,j['lead'],j['seed'],j['mode']),cwd=ROOT,
                                               env=environment(gpu,j['seed']),stdout=f,stderr=subprocess.STDOUT)
                    j.update(status='running',gpu=gpu,slot=slot,pid=child.pid,log=str(log),started_unix=time.time())
                    save(manifest,state)
                except Exception as error:
                    j.update(status='failed',error=repr(error));stop.set();save(manifest,state);return
            result=child.wait()
            p=Path(j['run_dir'])/'progress.json'
            completed=result==0 and p.exists() and json.loads(p.read_text()).get('status')=='completed'
            with lock:
                j.update(status='completed' if completed else 'failed',returncode=result,finished_unix=time.time())
                if not completed:stop.set()
                save(manifest,state)
    threads=[threading.Thread(target=worker,args=(str(slot//5),slot)) for slot in range(10)]
    for t in threads:t.start()
    for t in threads:t.join()
    if any(j['status']!='completed' for j in jobs):
        state['status']='stopped_after_failure';save(manifest,state);raise SystemExit('Full run failed.')
    state['status']='evaluating_mhw';save(manifest,state)
    data=next((BASE.parents[1]/'data').glob('*.nc'))
    cmd=[sys.executable,'-B',str(ROOT/'scripts/run_as_yeonwoo.py'),str(ROOT/'src/summarize_jeju_hilam_mhw_skill.py'),
         '--data-path',str(data),'--roots',str(output/'full/jeju_hilam'),str(output/'full/pi_jeju_hilam'),
         '--leads','7,14','--seeds','42,7,123','--output-prefix',str(output/'full/mhw_skill')]
    with (logs/'mhw_full.log').open('x') as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    state.update(status='completed' if result.returncode==0 else 'mhw_evaluation_failed',finished_unix=time.time())
    save(manifest,state)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
