"""Four actual-model CPU objectives; no optimizer step and no GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import json
from unittest.mock import patch
import torch
from torch.utils.data import default_collate
import train_jeju_hilam as t
from mhw.subset_preprocessing import load_subset_bundle,apply_manifest_physics

def norm(gs):return sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5

def main():
 torch.set_num_threads(4);path=R/'data/subsets/preprocessing_subset_v3/ratio_10_lead7_seed42.json';b,m=load_subset_bundle(path)
 with patch.object(sys,'argv',['trainer','--lead','7','--physics-mode','p0_original','--device','cpu']):args=t.parse_args()
 pa=t.build_physics_args(args);apply_manifest_physics(args,pa,m)
 t.set_seed(42);model=t.JejuHiLAM(graph=b.graph,input_dim=len(b.train.input_feature_names),physics_heads=True)
 batch=default_collate([b.train[0]]);output=model(batch['x_nodes'],base_nodes=batch['tm_norm_nodes'],return_aux=True)
 named=list(model.named_parameters());names=[n for n,v in named];params=[v for n,v in named];rows=[]
 for arm,lp,lr in [('baseline',0.,0.),('physics_only',.1,0.),('auxiliary_only',0.,10.),('full_pinn',.1,10.)]:
  pa.lambda_rres=lr
  with patch.object(t,'daily_rollout_physics_loss',wraps=t.daily_rollout_physics_loss) as ph,patch.object(t,'rres_penalty',wraps=t.rres_penalty) as rr:
   losses=t.compute_batch_losses(output,batch,b.graph,*b.stats['target'],pa,lp,model)
  gs=torch.autograd.grad(losses[-1],params,allow_unused=True,retain_graph=True)
  assert ph.call_count==int(lp!=0) and rr.call_count==int(lr!=0)
  assert torch.isfinite(losses[-1]) and all(torch.isfinite(g).all() for g in gs if g is not None)
  if not lp:assert all(g is None for n,g in zip(names,gs) if n.startswith('we_head.'))
  if not lr:assert all(g is None for n,g in zip(names,gs) if n.startswith('rres_head.'))
  objective=losses[2] if lp else losses[3] if lr else losses[1]
  shared=[v for n,v in named if not n.startswith(('we_head.','rres_head.'))]
  active=torch.autograd.grad(objective,shared,retain_graph=True,allow_unused=True);assert norm(active)>0
  row=dict(arm=arm,physics_calls=ph.call_count,rres_calls=rr.call_count,total_loss=float(losses[-1].detach()),data_loss=float(losses[1].detach()),physics_loss=float(losses[2].detach()),rres_penalty=float(losses[3].detach()),shared_active_objective_gradient_norm=norm(active),we_head_gradient_norm=norm([g for n,g in zip(names,gs) if n.startswith('we_head.')]),rres_head_gradient_norm=norm([g for n,g in zip(names,gs) if n.startswith('rres_head.')]),parameters_with_gradient=sum(v.numel() for (n,v),g in zip(named,gs) if g is not None),status='PASS')
  rows.append(row)
 result=dict(status='PASS',model='actual JejuHiLAM hidden256 processor3',scope='one CPU batch; all-head architecture to prove inactive heads have grad=None; native baseline has no auxiliary heads',manifest=str(path),optimizer_steps=0,arms=rows,parameter_counts={p:sum(v.numel() for n,v in named if (n.startswith(p+'.') if p!='shared' else not n.startswith(('we_head.','rres_head.')))) for p in ('shared','we_head','rres_head')})
 (R/'outputs/v3_preflight_20261008/loss_arms.json').write_text(json.dumps(result,indent=2)+'\n');b.ds.close();print(json.dumps(result),flush=True)

if __name__=='__main__':main()
