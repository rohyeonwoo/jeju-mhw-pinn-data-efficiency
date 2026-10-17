"""Isolated adaptive pilot helpers; existing fixed trainer and src stay immutable."""
import copy,csv,json,math,time,os,sys
from pathlib import Path
import train_v3_ablation as a
from train_v3_study import *
import train_hybrid_jeju_forecast as h
OUT=ROOT/'outputs/v3_adaptive_physics_pilot_20261011'
OLD=ROOT/'outputs/v3_physics_ratio_lead_lambda001_20261009'
SPEC=dict(initial_lambda=.01,rho=.25,lambda_min=.0001,lambda_max=.03,beta=.9,interval=100,epsilon=1e-12)
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  c=csv.DictWriter(f,keys);c.writeheader();c.writerows(rows)
def protocol():return json.loads((OUT/'protocol.json').read_text())
def load(jobid,device):
 pr=protocol();j=pr['jobs'][jobid];args,bundle,model,pa,cfg=a.setup(j,pr,device)
 assert args.precision=='fp32' and pa.zero_rres_physics and pa.lambda_rres==0 and pa.lambda_we_tie==0
 return j,args,bundle,model,pa,cfg
def shared(model):return [v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.')) and v.requires_grad]
def gradient_stats(data,phys,params):
 gd=torch.autograd.grad(data,params,retain_graph=True,allow_unused=True)
 gp=torch.autograd.grad(phys,params,retain_graph=True,allow_unused=True)
 nd=sum((v.detach().double().square().sum() for v in gd if v is not None),data.new_zeros((),dtype=torch.float64))
 np_=sum((v.detach().double().square().sum() for v in gp if v is not None),data.new_zeros((),dtype=torch.float64))
 dot=sum(((x.detach().double()*y.detach().double()).sum() for x,y in zip(gd,gp) if x is not None and y is not None),data.new_zeros((),dtype=torch.float64))
 nd=float(nd.sqrt());np_=float(np_.sqrt());dot=float(dot)
 if not all(math.isfinite(x) for x in [nd,np_,dot]):raise FloatingPointError('nonfinite preclip objective gradient')
 return dict(G_data=nd,G_phys=np_,cosine=dot/max(nd*np_,SPEC['epsilon']) if nd>0 and np_>0 else None,valid_direction=nd>0 and np_>0)
class Controller:
 def __init__(self,model):
  self.params=shared(model);self.steps=0;self.lam=SPEC['initial_lambda'];self.ema_data=None;self.ema_phys=None;self.rows=[];self.effective=[];self.measure_seconds=0.;self.epoch=1
 def state(self):return dict(steps=self.steps,lam=self.lam,ema_data=self.ema_data,ema_phys=self.ema_phys,rows=self.rows,measure_seconds=self.measure_seconds)
 def restore(self,s):
  for k,v in s.items():setattr(self,k,copy.deepcopy(v))
 def measure(self,data,phys,warmup):
  start=time.perf_counter();g=gradient_stats(data,phys,self.params);self.measure_seconds+=time.perf_counter()-start
  b=SPEC['beta'];self.ema_data=g['G_data'] if self.ema_data is None else b*self.ema_data+(1-b)*g['G_data'];self.ema_phys=g['G_phys'] if self.ema_phys is None else b*self.ema_phys+(1-b)*g['G_phys']
  target=SPEC['rho']*self.ema_data/(self.ema_phys+SPEC['epsilon']);self.lam=max(SPEC['lambda_min'],min(SPEC['lambda_max'],target))
  row=dict(update=self.steps,epoch=self.epoch,**g,ema_G_data=self.ema_data,ema_G_phys=self.ema_phys,lambda_target=target,lambda_adaptive=self.lam,lambda_eff=self.lam*warmup,warmup=warmup,clamped_low=target<SPEC['lambda_min'],clamped_high=target>SPEC['lambda_max'],ratio_before_warmup=self.lam*g['G_phys']/(g['G_data']+SPEC['epsilon']),effective_ratio=self.lam*warmup*g['G_phys']/(g['G_data']+SPEC['epsilon']),data_loss=float(data.detach()),physics_loss=float(phys.detach()))
  self.rows.append(row);return row
 def __enter__(self):
  self.original=t.compute_batch_losses;t.compute_batch_losses=self.losses;return self
 def __exit__(self,*exc):t.compute_batch_losses=self.original
 def losses(self,output,batch,graph,target_mean,target_std,physics_args,lambda_eff,model):
  # Native objectives remain byte-identical. Only detached scalar coefficient changes.
  vals=self.original(output,batch,graph,target_mean,target_std,physics_args,lambda_eff,model)
  warmup=h.physics_lambda_factor(self.epoch,physics_args.physics_warmup_epochs)
  assert abs(lambda_eff-.01*warmup)<1e-12
  if torch.is_grad_enabled() and model.training:
   self.steps+=1
   if self.steps%SPEC['interval']==0:self.measure(vals[1],vals[2],warmup)
   self.effective.append(self.lam*warmup)
  coeff=float(self.lam*warmup);total=vals[1]+coeff*vals[2]
  assert torch.isfinite(total) and float(vals[3])==float(vals[4])==0
  return (*vals[:5],total)
def environment_device():
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3']
 torch.set_num_threads(2);return torch.device('cuda:0')
def jobdir(jobid,phase):
 d=OUT/phase/str(jobid);d.mkdir(parents=True,exist_ok=True);return d
def preserve_rng():return (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],np.random.get_state(),random.getstate())
def restore_rng(s):
 torch.set_rng_state(s[0]);
 if s[1]:torch.cuda.set_rng_state_all(s[1])
 np.random.set_state(s[2]);random.setstate(s[3])
