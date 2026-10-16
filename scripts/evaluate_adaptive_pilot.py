"""Frozen adaptive checkpoint: Train/Validation only, no Test loader/inference."""
from adaptive_common import *
from torch.utils.data import DataLoader,Subset

def main(jobid):
 device=environment_device();d=jobdir(jobid,'evaluations');gpu_identity(d);j,args,bundle,model,pa,cfg=load(jobid,device);rd=Path(j['run_dir']);lock=json.loads((OUT/'checkpoint_lock.json').read_text());frozen=next(r for r in lock['rows'] if r['job']==jobid);assert lock['status']=='PASS' and sha(rd/'best.pt')==frozen['best_sha256']
 cp=torch.load(rd/'best.pt',map_location=device,weights_only=False);assert cp['config_sha256']==cfg['config_sha256'];model.load_state_dict(cp['model_state']);lam=cp['adaptive_state']['lam'];epoch=cp['best_epoch'];del cp;before=state_digest(model,False);results={};seasonal=[];spatial=[]
 for split in ['train','val']:
  model.eval()
  with torch.inference_mode():f=t.collect_predictions(model,t.make_loader(getattr(bundle,split),args,False),bundle.graph,device,*bundle.stats['target'],precision='fp32',physics_args=pa)
  mask=f['mask'].astype(bool);error=f['pred_raw'].astype(float)-f['target_raw'].astype(float);assert np.isfinite(error[mask]).all();rmse=float(np.sqrt((error[mask]**2).mean()));results[split]=dict(rmse=rmse,mae=float(abs(error[mask]).mean()),bias=float(error[mask].mean()))
  if split=='val':
   assert abs(rmse-frozen['best_val_rmse'])<2e-6
   idx=f['target_index'];months=bundle.ds.time.values[idx].astype('datetime64[M]').astype(int)%12+1;seasons=np.array(['DJF','MAM','JJA','SON'])[(months%12)//3]
   for season in ['DJF','MAM','JJA','SON']:
    keep=seasons==season;err=error[keep][mask[keep]];seasonal.append(dict(job=jobid,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model='adaptive',season=season,rmse=float(np.sqrt((err**2).mean())),count=len(err)))
   sea=bundle.train.base.ocean_mask>0
   for iy,ix in np.argwhere(sea):
    err=error[:,0,iy,ix];spatial.append(dict(job=jobid,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model='adaptive',iy=int(iy),ix=int(ix),rmse=float(np.sqrt((err**2).mean())),latitude=float(bundle.train.base.lat2d[iy,ix]),longitude=float(bundle.train.base.lon2d[iy,ix])))
  np.savez_compressed(d/(split+'_fields.npz'),**f)
  with torch.no_grad():native=t.run_epoch(model,t.make_loader(getattr(bundle,split),args,False),bundle.graph,device,*bundle.stats['target'],physics_args=pa,lambda_physics=lam,epoch=epoch)
  results[split].update(physics_loss=native['physics_loss'],raw_physics_residual_rmse=native['common_res_rmse_kday'],data_loss=native['data_loss'])
 indices=[int(r['subset_position']) for r in csv.DictReader((OUT/'diagnostics'/str(jobid)/'gradient_batch_indices.csv').open())];rows=[];model.train();effective=lam*h.physics_lambda_factor(epoch,5)
 for bi,b in enumerate(DataLoader(Subset(bundle.train,indices),batch_size=16,shuffle=False)):
  model.zero_grad(set_to_none=True);out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True);vals=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,effective,model);st=gradient_stats(vals[1],vals[2],shared(model));rows.append(dict(job=jobid,ratio=j['ratio'],lead=j['lead'],seed=j['seed'],state='adaptive_best',batch=bi,**st,lambda_base=lam,lambda_eff=effective,weighted_ratio=effective*st['G_phys']/(st['G_data']+1e-12),data_loss=float(vals[1].detach()),physics_loss=float(vals[2].detach())))
 assert state_digest(model,False)==before and sha(rd/'best.pt')==frozen['best_sha256']
 csvsave(d/'adaptive_best_gradients.csv',rows);csvsave(d/'validation_seasonal.csv',seasonal);csvsave(d/'validation_spatial.csv',spatial);save(d/'status.json',dict(status='PASS',job=jobid,metrics=results,lambda_base=lam,best_epoch=epoch,checkpoint_sha256=frozen['best_sha256'],no_test=True,no_optimizer_step=True,timestamp=stamp()));bundle.ds.close();print('VALIDATION EVAL PASS',jobid,flush=True)
if __name__=='__main__':
 import argparse,traceback
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);o=ap.parse_args()
 try:main(o.job)
 except Exception:save(jobdir(o.job,'evaluations')/'failure.json',dict(error=traceback.format_exc(),timestamp=stamp()));raise
