"""Inference only. Loads exactly the pre-frozen overall Validation-best checkpoint."""
from eval_common import *
import argparse,traceback,signal
import numpy as np
import torch
import worker as W
from metrics import score

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);a0=ap.parse_args();p=evaluation_protocol();verify_evaluation(p)
 job=p['jobs'][a0.job];d=EVAL/'runs'/job['name'];d.mkdir(parents=True,exist_ok=True)
 gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in [7,8,9]
 gate=Path(os.environ['EVAL_START_GATE']);end=time.monotonic()+60
 while not gate.exists():
  assert time.monotonic()<end;time.sleep(.05)
 lease=json.loads((EVAL/'leases.json').read_text())['running'];assert any(x['pid']==os.getpid() and x['job']==a0.job and x['gpu']==gpu for x in lease)
 assert sum(x['gpu']==gpu for x in lease)<=2
 save(d/'status.json',dict(status='RUNNING',gpu=gpu,pid=os.getpid(),timestamp=stamp()))
 def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
 signal.signal(signal.SIGTERM,stop)
 try:
  torch.set_num_threads(1);device=torch.device('cuda:0')
  torch.cuda.set_per_process_memory_fraction(14336*2**20/torch.cuda.get_device_properties(device).total_memory,device)
  W.w.gpu_identity(d)
  gpus=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()));uuid=next(r[1].strip() for r in gpus if int(r[0])==gpu)
  assert all(x[0].strip()==uuid for x in json.loads((d/'gpu_identity.json').read_text())['nvml'])
  # setup constructs the same model/statistics; it does not perform a train step.
  original=protocol()['jobs'][a0.job];args,b,model,pa,cfg=W.setup(original,protocol(),device)
  cfg['checkpoint_dir']=str(TRAIN/'checkpoints'/job['name']);cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg)
  recorded=json.loads((Path(original['run_dir'])/'config.json').read_text());assert cfg==recorded
  cp_path=Path(job['checkpoint']);assert cp_path.name=='best.pt' and sha(cp_path)==job['checkpoint_sha256']
  cp=torch.load(cp_path,map_location=device,weights_only=False)
  assert cp['config_sha256']==cfg['config_sha256'] and cp['stats']==b.stats and cp['best_epoch']==job['best_epoch'] and cp['epoch']==job['best_epoch']
  assert cp['best_value']==job['best_val_rmse'];model.load_state_dict(cp['model_state'],strict=True);del cp
  for par in model.parameters():par.requires_grad_(False)
  model.eval();before=W.w.state_digest(model,False);begun=time.monotonic();torch.cuda.reset_peak_memory_stats()
  assert isinstance(b.test,W.old.PredictionOnly)
  first=b.test[0];assert not any(k.startswith('mixed_') or k.endswith('_seq') for k in first)
  collect=lambda dataset:W.t.collect_predictions(model,W.t.make_loader(dataset,args,False),b.graph,device,*b.stats['target'],precision='fp32',physics_args=None)
  with torch.no_grad():
   f=collect(b.test)
   oracle=W.mb.ForcingDataset(b.test.base,W.mb.arrays())
   for dataset in [oracle,W.old.PoisonForcing(oracle)]:
    other=collect(dataset)
    for k in ['pred_raw','target_raw','target_index','mask']:assert np.array_equal(f[k],other[k]),('forcing leakage',k)
    del other
  assert W.w.state_digest(model,False)==before and all(x.grad is None for x in model.parameters())
  assert np.array_equal(f['target_index'],b.test_indices+29+args.lead)
  dates=b.ds.time.values[f['target_index']]
  with np.load(TRAIN/'evaluation_reference.npz') as z:
   assert max(z['reference_indices'])<min(f['target_index']);clim=z['clim'];threshold=z['threshold']
  measured,daily=score(f,dates,clim,threshold)
  measured.update(model=job['model'],model_label=job['label'],ratio=job['ratio'],lead=job['lead'],seed=job['seed'],split='test',best_epoch=job['best_epoch'],validation_rmse=job['best_val_rmse'],generalization_gap_rmse=measured['rmse']-job['best_val_rmse'],checkpoint=str(cp_path),checkpoint_sha256=job['checkpoint_sha256'],config_sha256=cfg['config_sha256'],reference_sha256=p['reference_sha256'],forward_future_forcing=False,optimizer_steps=0,model_state_unchanged=True,forcing_invariance='PASS',peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,evaluation_seconds=time.monotonic()-begun)
  assert sha(cp_path)==job['checkpoint_sha256']
  tmp=d/'fields.tmp.npz';np.savez_compressed(tmp,**f,target_dates=dates);tmp.replace(d/'fields.npz')
  tmp=d/'daily_metrics.tmp.npz';np.savez_compressed(tmp,**daily);tmp.replace(d/'daily_metrics.npz')
  save(d/'forcing_invariance_test.json',dict(status='PASS',scope='full Test split',n_targets=len(dates),modes=['removed','oracle_present','NaN_poisoned'],prediction_bitwise_identical=True,targets_and_masks_identical=True,SST_MHW_metrics_identical=True,model_state_unchanged=True,checkpoint_sha256=job['checkpoint_sha256']))
  save(d/'metrics.json',measured);save(d/'status.json',dict(status='COMPLETED',gpu=gpu,pid=os.getpid(),checkpoint_sha256=job['checkpoint_sha256'],timestamp=stamp()))
  b.ds.close();log(f'Test evaluation complete {job["name"]}, overall Validation-best only; no training updates')
 except BaseException as e:
  save(d/'status.json',dict(status='PAUSED' if isinstance(e,KeyboardInterrupt) else 'FAILED',error=traceback.format_exc(),timestamp=stamp()));raise
if __name__=='__main__':main()
