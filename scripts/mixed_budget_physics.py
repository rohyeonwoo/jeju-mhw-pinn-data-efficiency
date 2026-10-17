"""NOAA SST proxy constraint, oracle forcing; no future target in the RHS."""
from mixed_budget_common import *
import numpy as np
import torch
from torch.utils.data import Dataset
from mhw.ocean_stencil import ocean_neighbor_gradients,ocean_neighbor_support
DAY=86400.;RHO=1025.;CP=3990.;KD=.06

class ForcingDataset(Dataset):
 def __init__(self,base,arrays):self.base=base;self.arrays=arrays;self.input_feature_names=base.input_feature_names
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  b=self.base[i];t=int(b['current_index']);L=int(b['target_index'])-t
  for k in ['h','qnet','ssr','u','v','tb','we','valid']:
   # Copied oracle forcing is used only by the loss, never x_nodes/base_nodes.
   b['mixed_'+k]=torch.from_numpy(np.array(self.arrays[k][t:t+L,None],copy=True)).float()
  return b

def arrays():
 info=json.loads((FEATURES/'manifest.json').read_text());assert info['status']=='complete'
 return {p.stem:np.load(p,mmap_mode='r') for p in FEATURES.glob('*.npy')}

def common_mask(mask,batch):return mask*(batch['mixed_valid'].to(mask.device).amin(dim=1)>0).to(mask.dtype)

def terms(field,h,qnet,ssr,u,v,tb,we,mask,dx,dy,kd=KD):
 h=h.clamp_min(5.);safe=lambda x:torch.where(mask>0,x,torch.zeros_like(x))
 field=safe(field);tb=safe(tb);gx,gy=ocean_neighbor_gradients(field,dx,dy,mask)
 return dict(flux=safe(DAY*qnet/(RHO*CP*h)),penetration=safe(DAY*ssr*torch.exp(-kd*h)/(RHO*CP*h)),entrainment=safe(-DAY*we*(field-tb)/h),advection=safe(-DAY*(u*gx+v*gy)))

def path(pred,tm,mask,batch,lead,advection):
 mask=common_mask(mask,batch);support=ocean_neighbor_support(mask);device=pred.device
 f={k:batch['mixed_'+k].to(device) for k in ['h','qnet','ssr','u','v','tb','we']};dx=batch['dx'].to(device);dy=batch['dy'].to(device)
 tm=torch.where(mask>0,tm,torch.zeros_like(tm));pred=torch.where(mask>0,pred,torch.zeros_like(pred))
 rows=[];rhs=[];res=[];increment=(pred-tm)/lead
 for j in range(lead):
  field=tm+(j/lead)*(pred-tm)
  ts=terms(field,**{k:v[:,j] for k,v in f.items()},mask=mask,dx=dx,dy=dy)
  rr=ts['flux']-ts['penetration']+ts['entrainment']+(ts['advection'] if advection else 0)
  rows.append(ts);rhs.append(rr);res.append(torch.where(support>0,increment-rr,torch.zeros_like(rr)))
 return torch.cat(res),torch.cat(rhs),support.repeat(lead,1,1,1),{k:torch.cat([r[k] for r in rows]) for k in rows[0]}

def loss(pred_raw,Tm,target_raw,mask,batch,args,lead_days,we_pred,rres_pred):
 # target_raw deliberately unused; forcing has no future NOAA SST.
 r,rhs,s,_=path(pred_raw,Tm,mask,batch,lead_days,args.mixed_variant=='B')
 from train_hybrid_jeju_forecast import compute_physics_loss
 return compute_physics_loss(r,s,args,rhs_per_day=rhs)

def raw_stats(pred_raw,tm,mask,batch,physics_args,lead_days,we_pred,rres_pred):
 r,_,s,_=path(pred_raw,tm,mask,batch,lead_days,physics_args.mixed_variant=='B')
 return dict(common_res_abs_sum=float((r.abs()*s).sum()),common_res_sq_sum=float((r.square()*s).sum()),common_res_count=float(s.sum()))

def budget_stats(pred_raw,Tm,mask,batch,args,lead_days,we_pred,rres_pred):
 _,_,s,ts=path(pred_raw,Tm,mask,batch,lead_days,args.mixed_variant=='B')
 out={'mixed_count':float(s.sum())}
 for k,x in ts.items():out['mixed_'+k+'_sum']=float((x*s).sum());out['mixed_'+k+'_sq']=float((x.square()*s).sum())
 return out

def finalize(s):
 n=max(s.get('mixed_count',0),1)
 return {f'mixed_{k}_rms_kday':(s.get('mixed_'+k+'_sq',0)/n)**.5 for k in ['flux','penetration','entrainment','advection']}
