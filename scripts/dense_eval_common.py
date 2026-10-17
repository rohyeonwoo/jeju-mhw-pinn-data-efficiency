"""Shared read-only validation diagnostics and update-indexed evaluation schedule."""
from contextlib import contextmanager
import copy,math,random
import train_p0_lambda as lw
w=lw.w;t=w.t;torch=w.torch;np=w.np
BLOCK=478;INTERVAL=24;MAX_STEPS=80*BLOCK;PATIENCE=12*BLOCK

def next_event(step,best_step,max_steps=MAX_STEPS,patience=PATIENCE):
 return min((step//INTERVAL+1)*INTERVAL,(step//BLOCK+1)*BLOCK,best_step+patience,max_steps)

@contextmanager
def preserve_rng():
 cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
 numpy=np.random.get_state();python=random.getstate()
 try:yield
 finally:
  torch.set_rng_state(cpu)
  if cuda is not None:torch.cuda.set_rng_state_all(cuda)
  np.random.set_state(numpy);random.setstate(python)

def sst_validation(model,bundle,args,device,max_batches=None):
 with preserve_rng(),torch.no_grad():
  return t.run_epoch(model,t.make_loader(bundle.val,args,False),graph=bundle.graph,device=device,target_mean=bundle.stats['target'][0],target_std=bundle.stats['target'][1],physics_args=None,lambda_physics=0,precision='fp32',max_batches=max_batches)

def common_diagnostics(model,bundle,args,device,max_batches=None,observations=False):
 """Common we=0 resolved residual; learned-we P0 residual separately. Validation only."""
 pa_args=copy.copy(args);pa_args.physics_mode='p0_original'
 pa=t.build_physics_args(pa_args)
 manifest=__import__('json').loads(args.subset_manifest.read_text())
 w.p.apply_manifest_physics(pa_args,pa,manifest);assert pa.zero_rres_physics
 sums={key:dict(common_res_abs_sum=0.,common_res_sq_sum=0.,common_res_count=0.) for key in ['zero_we','learned_we','observed_zero_we']}
 raw=dict(norm_sse=0.,raw_sse=0.,raw_abs=0.,raw_bias=0.,count=0.);indices=[]
 with preserve_rng(),torch.no_grad():
  model.eval()
  for i,batch in enumerate(t.make_loader(bundle.val,args,False)):
   if max_batches is not None and i>=max_batches:break
   out=model(batch['x_nodes'].to(device),base_nodes=batch['tm_norm_nodes'].to(device),return_aux=model.physics_heads)
   pred=out['pred'] if isinstance(out,dict) else out
   stats=t.batch_raw_stats(pred,batch['y_nodes'].to(device),batch['y_raw_nodes'].to(device),*bundle.stats['target'])
   for k,v in stats.items():raw[k]+=v
   grid=t.nodes_to_grid_torch(pred*bundle.stats['target'][1]+bundle.stats['target'][0],bundle.graph)
   zero=torch.zeros_like(grid);tm=batch['Tm'].to(device);mask=batch['mask'].to(device)
   leads=batch['target_index']-batch['current_index'];assert (leads==args.lead).all()
   fields=[('zero_we',grid,zero)]
   if model.physics_heads:fields.append(('learned_we',grid,t.nodes_to_grid_torch(out['we'],bundle.graph)))
   if observations:fields.append(('observed_zero_we',batch['y_raw'].to(device),zero))
   for key,state,we in fields:
    values=t.physics_raw_residual_stats(state,tm,mask,batch,pa,args.lead,we,zero)
    for name,value in values.items():sums[key][name]+=value
   indices.extend(batch['target_index'].tolist())
 result=dict(val_rmse=math.sqrt(raw['raw_sse']/raw['count']),val_bias=raw['raw_bias']/raw['count'],target_index_sha256=w.digest(indices),forecast_count=len(indices),rhs_scale=pa.physics_rhs_scale,physics_mask_cells=660,definition='daily linear path, corrected ocean stencil, unclipped K/day, closure zero; common entrainment we=0, learned we separate')
 for key,values in sums.items():
  count=values['common_res_count']
  result[key]=None if not count else dict(rmse_kday=math.sqrt(values['common_res_sq_sum']/count),mae_kday=values['common_res_abs_sum']/count,count=count)
 assert result['zero_we']['count']==len(indices)*args.lead*660
 assert all(math.isfinite(v) for v in result.values() if isinstance(v,(int,float)))
 return result
