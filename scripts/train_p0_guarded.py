"""Experimental P0-daily update: data-priority projection and gradient-norm cap.

Physics equations, daily path, heads, v3 statistics and native AdamW are unchanged.
This is an asymmetric PCGrad-inspired optimizer update, NOT the original scalar
PINN gradient. The logged native total loss is a diagnostic scalar, not a global
objective whose derivative equals the projected update. No test selection.
"""
import os,sys,json,math
from pathlib import Path
import torch
import train_v3_study as w
ROOT=w.ROOT


def surgery(data_grads,physics_grads,shared,lambda_effective):
 """At most lambda_effective*||g_data|| on shared parameters; remove opposition."""
 sample=next(g for g in data_grads if g is not None)
 zero=sample.new_zeros(())
 norm2=sum((g.square().sum() for g,s in zip(data_grads,shared) if s and g is not None),zero)
 dot=sum(((d*p).sum() for d,p,s in zip(data_grads,physics_grads,shared) if s and d is not None and p is not None),zero)
 phys2=sum((p.square().sum() for p,s in zip(physics_grads,shared) if s and p is not None),zero)
 beta=torch.minimum(dot/norm2.clamp_min(1e-30),zero)
 projected=[]
 for d,p,s in zip(data_grads,physics_grads,shared):
  if s and d is not None:projected.append((torch.zeros_like(d) if p is None else p)-beta*d)
  else:projected.append(p)
 projected2=sum((g.square().sum() for g,s in zip(projected,shared) if s and g is not None),zero)
 limit=torch.as_tensor(lambda_effective,device=zero.device,dtype=zero.dtype)
 alpha=torch.minimum(limit,limit*torch.sqrt(norm2)/torch.sqrt(projected2).clamp_min(1e-30))
 # Zero data gradient means no physics update; no arbitrary model-only step.
 alpha=torch.where(norm2>1e-30,alpha,zero)
 combined=[None if d is None and p is None else (0 if d is None else d)+alpha*(0 if p is None else p) for d,p in zip(data_grads,projected)]
 actual_dot=sum(((d*p).sum() for d,p,s in zip(data_grads,projected,shared) if s and d is not None and p is not None),zero)
 metrics=dict(data_grad_norm=float(norm2.sqrt()),physics_grad_norm=float(phys2.sqrt()),cosine_before=float(dot/(norm2*phys2).sqrt().clamp_min(1e-30)),cosine_after=float(actual_dot/(norm2*projected2).sqrt().clamp_min(1e-30)),alpha=float(alpha),lambda_effective=float(limit),weighted_ratio_before=float(limit*phys2.sqrt()/norm2.sqrt().clamp_min(1e-30)),weighted_ratio_after=float(alpha*projected2.sqrt()/norm2.sqrt().clamp_min(1e-30)),conflict=bool(dot<0))
 assert all(math.isfinite(v) for v in metrics.values())
 assert metrics['weighted_ratio_after']<=lambda_effective+1e-5
 assert metrics['cosine_after']>=-1e-4 or metrics['weighted_ratio_after']<1e-6
 return combined,metrics

class InjectUpdate(torch.autograd.Function):
 @staticmethod
 def forward(ctx,scalar,gradients,*parameters):ctx.gradients=gradients;return scalar.clone()
 @staticmethod
 def backward(ctx,upstream):return (None,None,*(None if g is None else upstream*g for g in ctx.gradients))

class Guard:
 def __init__(self):self.enabled=False;self.rows=[];self.native=w.t.compute_batch_losses
 def loss(self,output,batch,graph,mean,std,pa,lambda_eff,model):
  vals=self.native(output,batch,graph,mean,std,pa,lambda_eff,model)
  if not self.enabled or not torch.is_grad_enabled() or pa is None or lambda_eff==0:return vals
  named=list(model.named_parameters());params=[v for n,v in named];shared=[not n.startswith(('we_head.','rres_head.')) for n,v in named]
  gd=torch.autograd.grad(vals[1],params,retain_graph=True,allow_unused=True)
  gp=torch.autograd.grad(vals[2],params,allow_unused=True)
  assert all(torch.isfinite(g).all() for g in [*gd,*gp] if g is not None)
  combined,metrics=surgery(gd,gp,shared,lambda_eff);self.rows.append(metrics)
  total=InjectUpdate.apply(vals[5].detach(),combined,*params)
  return (*vals[:5],total)
 def summary(self):
  rows=self.rows
  if not rows:return {}
  return {**{'guard_'+k+'_mean':sum(float(r[k]) for r in rows)/len(rows) for k in rows[0]},'guard_batches':len(rows),'guard_weighted_ratio_max':max(r['weighted_ratio_after'] for r in rows),'guard_min_cosine_after':min(r['cosine_after'] for r in rows),'guard_total_loss_semantics':'native diagnostic scalar; projected parameter update'}

def install():
 guard=Guard();original_monitor=w.Monitor;original_checkpoint=w.checkpoint
 class GuardedMonitor(original_monitor):
  def reset(self):super().reset();guard.rows=[]
  def __enter__(self):guard.enabled=True;return super().__enter__()
  def __exit__(self,*a):guard.enabled=False;return super().__exit__(*a)
 def checked_checkpoint(path,obj):
  if 'history' in obj and obj['history']:obj['history'][-1].update(guard.summary())
  return original_checkpoint(path,obj)
 w.t.compute_batch_losses=guard.loss;w.Monitor=GuardedMonitor;w.checkpoint=checked_checkpoint
 return guard

def main():
 import argparse,traceback
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);opt=ap.parse_args()
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert visible and all(x in ['1','2','3'] for x in visible),'GPU0 prohibited'
 protocol=json.loads((opt.study/'protocol.json').read_text());assert protocol['phase']=='training_validation_only' and len(protocol['jobs'])==6
 install()
 try:w.worker(opt.study,opt.job,'train')
 except Exception:
  job=protocol['jobs'][opt.job];d=Path(job['run_dir']);d.mkdir(parents=True,exist_ok=True)
  w.save(d/'failure.json',dict(error=traceback.format_exc(),timestamp=w.stamp()));w.log('guarded P0 실행실패 '+job['name']);raise
if __name__=='__main__':main()
