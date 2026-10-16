from adaptive_common import *
from mhw.ocean_stencil import ocean_neighbor_gradients,ocean_neighbor_support
from torch.utils.data import DataLoader,Subset

def tree_equal(x,y):
 if isinstance(x,torch.Tensor):return torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(tree_equal(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(tree_equal(a,b) for a,b in zip(x,y))
 return x==y

def cpu_tests():
 model=torch.nn.Linear(2,1,bias=False);model.weight.data.copy_(torch.tensor([[1.,2.]]));c=Controller(model);o=torch.optim.AdamW(model.parameters(),lr=1e-3)
 analytic=[]
 for i in range(3):
  o.zero_grad(set_to_none=True);data=model.weight.square().sum();phys=(3*model.weight).square().sum();st=gradient_stats(data,phys,c.params)
  assert abs(st['G_phys']/st['G_data']-9)<1e-6 and abs(st['cosine']-1)<1e-10
  before=copy.deepcopy(o.state_dict());params=copy.deepcopy(model.state_dict());c.steps=100*(i+1);c.measure(data,phys,1.)
  assert tree_equal(before,o.state_dict()) and tree_equal(params,model.state_dict()) and all(v.grad is None for v in model.parameters())
  assert abs(c.lam-.25/9)<1e-6 and isinstance(c.lam,float)
  (data+c.lam*phys).backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);o.step();analytic.append(float(norm))
 saved=c.state();restored=Controller(model);restored.restore(saved);assert restored.state()==saved
 # Explicit lower/upper clamp with native autograd on proportional objectives.
 for scale,expected in [(1e6,SPEC['lambda_min']),(1e-6,SPEC['lambda_max'])]:
  cc=Controller(model);data=model.weight.square().sum();phys=scale*data;cc.measure(data,phys,.2);assert cc.lam==expected
 return dict(status='PASS',analytic_norm_ratio=9,updates=3,optimizer_and_parameters_unchanged_by_measurement=True,detach=True,clamp_low_high=True,state_roundtrip=True,finite_gradnorms=analytic)

def main(jobid):
 cpu=cpu_tests();device=environment_device();d=jobdir(jobid,'preflight');gpu_identity(d);j,args,bundle,model,pa,cfg=load(jobid,device)
 ref=json.loads((Path(j['reference_fixed_run'])/'config.json').read_text())
 for k in ['shared_initial_sha256','parameters','stats','tensor_fingerprints','rhs_scale','split_sha256']:assert cfg[k]==ref[k],k
 assert cfg['parameters']==json.loads((Path(j['reference_baseline_run'])/'config.json').read_text())['parameters']
 b=next(iter(t.make_loader(bundle.train,args,False)));mask=b['mask'].to(device);field=torch.full_like(mask,20.);g=ocean_neighbor_gradients(field,b['dx'].to(device),b['dy'].to(device),mask);assert all((x[mask>0]==0).all() for x in g)
 field=torch.arange(mask.numel(),device=device).reshape(mask.shape).float();g1=ocean_neighbor_gradients(field,b['dx'].to(device),b['dy'].to(device),mask);field[mask==0]=float('nan');g2=ocean_neighbor_gradients(field,b['dx'].to(device),b['dy'].to(device),mask);assert all(torch.equal(x,y) for x,y in zip(g1,g2));assert int(ocean_neighbor_support(mask)[0].sum())==660
 initial=copy.deepcopy(model.state_dict());rng=preserve_rng();model.train();out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True)
 native=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,.002,model)
 controller=Controller(model);controller.epoch=1
 with controller:adapt=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,.002,model)
 assert all(torch.equal(x,y) for x,y in zip(native,adapt));assert controller.steps==1 and controller.rows==[]
 shared_parameters=shared(model);g_native=torch.autograd.grad(native[-1],shared_parameters,retain_graph=True,allow_unused=True);g_adapt=torch.autograd.grad(adapt[-1],shared_parameters,allow_unused=True)
 assert all((x is None and y is None) or torch.equal(x,y) for x,y in zip(g_native,g_adapt));del out,native,adapt,g_native,g_adapt
 model.load_state_dict(initial);restore_rng(rng);opt=torch.optim.AdamW(model.parameters(),lr=args.initial_lr,weight_decay=args.weight_decay,betas=(.9,.95));controller=Controller(model);controller.steps=98;controller.epoch=1
 torch.cuda.reset_peak_memory_stats();norms=[]
 for i in range(3):
  opt.zero_grad(set_to_none=True);out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True);oldstate=copy.deepcopy(opt.state_dict());params=state_digest(model,False)
  with controller:vals=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,.002,model)
  assert tree_equal(oldstate,opt.state_dict()) and params==state_digest(model,False);assert all(v.grad is None for v in model.parameters())
  vals[-1].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert all(v.grad is None for n,v in model.named_parameters() if n.startswith('rres_head.'));opt.step();norms.append(float(norm))
  assert all(torch.isfinite(v).all() for v in model.state_dict().values())
  if i==1:
   saved=dict(model_state=copy.deepcopy(model.state_dict()),optimizer_state=copy.deepcopy(opt.state_dict()),adaptive_state=controller.state(),rng=preserve_rng());checkpoint(d/'roundtrip.pt',saved)
 continuous=copy.deepcopy(model.state_dict());saved=torch.load(d/'roundtrip.pt',map_location=device,weights_only=False);model.load_state_dict(saved['model_state']);opt.load_state_dict(saved['optimizer_state']);controller.restore(saved['adaptive_state']);restore_rng((saved['rng'][0].cpu(),[x.cpu() for x in saved['rng'][1]],saved['rng'][2],saved['rng'][3]));opt.zero_grad(set_to_none=True)
 out=model(b['x_nodes'].to(device),base_nodes=b['tm_norm_nodes'].to(device),return_aux=True)
 with controller:vals=t.compute_batch_losses(out,b,bundle.graph,*bundle.stats['target'],pa,.002,model)
 vals[-1].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();assert all(torch.equal(v,continuous[k]) for k,v in model.state_dict().items())
 assert controller.steps==101 and len(controller.rows)==1 and controller.rows[0]['update']==100
 peak=torch.cuda.max_memory_reserved()/2**20;del saved,opt,controller,out,vals,continuous
 # Matched-stream and RNG replay before the measured benchmark.
 model.load_state_dict(initial);t.set_seed(args.seed);stream=a.Stream(t.make_loader(bundle.train,args,True));batches=list(stream.take(3));ss=stream.state();next_b=next(stream.take(1));stream2=a.Stream(t.make_loader(bundle.train,args,True));stream2.restore(ss);next_b2=next(stream2.take(1));assert torch.equal(next_b['target_index'],next_b2['target_index'])
 timings=[]
 if args.seed==42:
  for mode in ['fixed','adaptive']:
   model.load_state_dict(initial);restore_rng(rng);opt=torch.optim.AdamW(model.parameters(),lr=args.initial_lr,weight_decay=args.weight_decay,betas=(.9,.95));controller=Controller(model);monitor=a.Monitor(model,pa);loader=DataLoader(Subset(bundle.train,list(range(min(len(bundle.train),1600)))),batch_size=16,shuffle=False)
   batch_list=list(loader);repeat=(batch_list*((100+len(batch_list)-1)//len(batch_list)))[:100];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   with monitor:
    if mode=='adaptive':
     with controller:met=t.run_epoch(model,repeat,bundle.graph,device,*bundle.stats['target'],optimizer=opt,physics_args=pa,lambda_physics=.01,epoch=1)
    else:met=t.run_epoch(model,repeat,bundle.graph,device,*bundle.stats['target'],optimizer=opt,physics_args=pa,lambda_physics=.01,epoch=1)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert monitor.phys_calls==100 and monitor.rres_calls==0
   timings.append(dict(mode=mode,updates=100,seconds=elapsed,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,scope='disposable native100-update train benchmark; same initial model/RNG/batches, no Validation/Test'))
   peak=max(peak,torch.cuda.max_memory_reserved()/2**20);del opt,controller,monitor,repeat,batch_list
 save(d/'status.json',dict(status='PASS',job=jobid,cpu=cpu,native_fixed_loss_and_gradient_exact=True,coastal_constant_and_land_invariance=True,shared_parameter_identity=True,detached_scalar=True,no_optimizer_mutation_from_measurement=True,inactive_auxiliary=True,real_gpu_updates=3,preclip_norms=norms,synthetic_controller_boundary='steps98 to101 to exercise100-update schedule',resume_exact_next_model=True,batch_stream_resume_exact=True,peak_reserved_mib=peak,benchmark=timings,timestamp=stamp()));bundle.ds.close();print('PREFLIGHT PASS',jobid,flush=True)
if __name__=='__main__':
 import argparse,traceback
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);o=ap.parse_args()
 try:main(o.job)
 except Exception:save(jobdir(o.job,'preflight')/'failure.json',dict(error=traceback.format_exc(),timestamp=stamp()));raise
