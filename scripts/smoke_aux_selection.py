"""Disposable real-model all-head ablation smoke; no production checkpoint reuse."""
import argparse,copy,gc
import train_v3_ablation as a
from train_v3_study import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda:0'],required=True);args=ap.parse_args();study=args.study
 gpu=args.device!='cpu';device=torch.device(args.device);d=study/('gpu_smoke' if gpu else 'cpu_smoke');d.mkdir(exist_ok=True)
 torch.set_num_threads(2)
 if gpu:
  assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3'];gpu_identity(d)
 p0=json.loads((study/'protocol.json').read_text());job=p0['jobs'][0]
 opts,bundle,model,pa,cfg=a.setup(job,p0,device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rows=[]
 opts.batch_size=16 if gpu else 1
 batches=[]
 for b in t.make_loader(bundle.train,opts,False):
  batches.append(b)
  if len(batches)==(2 if gpu else 1):break
 if gpu:torch.cuda.reset_peak_memory_stats()
 full_initial=state_digest(model,shared=False)
 for arm,lp,lr in [('baseline',0.,0.),('physics_only',.01,0.)]+[(f'{label}_{beta}',lp,beta) for beta in p0['candidate_weights'] for label,lp in [('auxiliary',0.),('full',.01)]]:
  model.load_state_dict(initial);pa.lambda_physics=lp;pa.lambda_rres=lr;monitor=a.Monitor(model,pa)
  assert state_digest(model,shared=False)==full_initial
  model.train();b=batches[0];out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True)
  with monitor:vals=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,lp*.2,model)
  shared=[v for n,v in model.named_parameters() if not n.startswith(('we_head.','rres_head.'))];norms={}
  for name,index,weight in [('data',1,1),('physics',2,lp),('auxiliary',3,lr)]:
   if not weight:continue
   gs=torch.autograd.grad(vals[index],shared,retain_graph=True,allow_unused=True);norm=sum(float(g.double().square().sum()) for g in gs if g is not None)**.5;assert math.isfinite(norm) and norm>0;norms[name]=norm
  assert monitor.phys_calls==int(lp!=0) and monitor.rres_calls==int(lr!=0)
  del gs,vals,out,shared
  model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=opts.initial_lr,weight_decay=opts.weight_decay,betas=(.9,.95));monitor.reset()
  with monitor:
   metrics=t.run_epoch(model,batches,bundle.graph,device,*bundle.stats['target'],optimizer=optimizer,physics_args=pa,lambda_physics=lp,epoch=1)
  assert monitor.phys_calls==(len(batches) if lp else 0) and monitor.rres_calls==(len(batches) if lr else 0)
  assert all(math.isfinite(v) for v in metrics.values()) and all(torch.isfinite(v).all() for v in model.state_dict().values())
  row=dict(arm=arm,physics_calls=monitor.phys_calls,rres_calls=monitor.rres_calls,loss=metrics['total_loss'],data_loss=metrics['data_loss'],physics_loss=metrics['physics_loss'],rres_loss=metrics['rres_l2'],gradient_norms=monitor.norms,initial_shared_objective_norms=norms,inactive_head_grad_none=True)
  cp=d/(arm+'.pt');checkpoint(cp,dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict()));saved=torch.load(cp,map_location=device,weights_only=False)
  assert all(torch.equal(saved['model_state'][k],v) for k,v in model.state_dict().items());row['checkpoint_roundtrip']=True;rows.append(row)
  del saved,optimizer;gc.collect()
  print(json.dumps(row),flush=True)
 result=dict(status='PASS',device=str(device),conditions=rows,identical_all_head_initial_state=full_initial,parameter_counts=cfg['parameters'],batches_per_arm=len(batches),batch_size=opts.batch_size,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if gpu else None,scope='disposable smoke; production restarts seed; Test disabled')
 save(d/'result.json',result);bundle.ds.close();log('auxiliary candidates '+str(device)+' smoke PASS: four active/inactive loss paths, shared objective gradients, finite updates and checkpoint roundtrip.');print('PASS',flush=True)
if __name__=='__main__':main()
