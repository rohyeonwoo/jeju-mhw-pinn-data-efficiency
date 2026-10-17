"""CPU-only diagnosis of existing v2 path; no optimizer or persistent code patch."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import ast,json,copy,subprocess
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
from mhw import subset_preprocessing as p
import train_jeju_hilam as t
import train_hybrid_jeju_forecast as h
OUT=ROOT/'outputs/v2_final_preflight_20261008'
BACK=Path('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts')

def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def norm(gs):return sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5

def main():
    torch.set_num_threads(4)
    gpu=subprocess.run(['nvidia-smi'],text=True,capture_output=True)
    save('gpu_smoke.json',dict(status='NOT_RUN',reason='STEP2 coastal stencil blocker; GPU smoke prohibited until corrected path and scale are validated',gpu_inventory_accessible=(gpu.returncode==0),nvidia_smi_returncode=gpu.returncode,nvidia_smi_stdout=gpu.stdout,nvidia_smi_stderr=gpu.stderr,models={x:dict(forward='NOT_RUN',backward='NOT_RUN',loss_finite='NOT_VERIFIED',gradient_finite='NOT_VERIFIED') for x in ['Baseline','P0 Physics only']},optimizer_steps=0))
    manifest=ROOT/'data/subsets/preprocessing_subset_v2/ratio_10_lead7_seed42.json'
    bundle,m=p.load_subset_bundle(manifest);base=bundle.train.base
    mask=torch.from_numpy(base.ocean_mask)[None,None].float();dx=torch.from_numpy(base.dx)[None,None].float();dy=torch.from_numpy(base.dy)[None,None].float()
    # Extract only the pure documented stencil functions; never run the backup runner.
    tree=ast.parse((BACK/'physics_stencil.py').read_text())
    names={'shifted_neighbors','ocean_neighbor_support','ocean_neighbor_gradients'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env={'torch':torch,'F':F};exec(compile(ast.Module(body=body,type_ignores=[]),'backup_stencil_pure_functions','exec'),env)
    corrected=lambda field,dx,dy:env['ocean_neighbor_gradients'](field,dx,dy,mask)
    support=env['ocean_neighbor_support'](mask)
    constant=20*mask;other=constant+100*(1-mask)
    native=h.spatial_gradients_torch(constant,dx,dy);changed=h.spatial_gradients_torch(other,dx,dy)
    cg=corrected(constant,dx,dy);cg2=corrected(other,dx,dy)
    assert all(torch.equal(a,b) for a,b in zip(cg,cg2))
    assert all(torch.count_nonzero(g)==0 for g in cg)
    affected=(((native[0]!=0)|(native[1]!=0))&(mask>0))
    assert affected.sum()>0
    coast=dict(status='FAIL_CURRENT_ENTRYPOINT',source_current='src/train_hybrid_jeju_forecast.py:585-601',source_corrected=str(BACK/'physics_stencil.py'),ocean_cells=int(mask.sum()),corrected_physics_cells=int(support.sum()),constant_ocean_nonzero_gradient_cells=int(affected.sum()),native_constant_max_abs_gradient_K_per_m=max(float((g*mask).abs().max()) for g in native),corrected_constant_max_abs_gradient=0.,native_land_perturbation_max_abs_delta_K_per_m=max(float(((a-b)*mask).abs().max()) for a,b in zip(native,changed)),corrected_land_perturbation_max_abs_delta=0.,manifest_rhs_scale=m['physics_rhs_scale'])
    original_linear=h._linear_rhs_per_day
    def masked_linear(pred_raw,Tm,mask,batch,args,lead_days,we_pred):
        return original_linear(pred_raw,Tm,env['ocean_neighbor_support'](mask),batch,args,lead_days,we_pred)
    with patch.object(h,'spatial_gradients_torch',corrected),patch.object(h,'_linear_rhs_per_day',masked_linear):
        scale=p.compute_subset_rhs_scale(base,m['train_start_indices'],7)
    coast['diagnostic_corrected_subset_rhs_scale']=scale
    coast['legacy_to_corrected_scale_ratio']=m['physics_rhs_scale']/scale
    coast['data_modified']=False
    save('coastal_stencil.json',coast);print('Coastal diagnostic',json.dumps(coast),flush=True)
    # Same actual architecture and seed; compare all predictor weights and outputs.
    with patch.object(sys,'argv',['audit','--lead','7','--seed','42','--physics-mode','p0_original','--lambda-rres','0','--device','cpu']):args=t.parse_args()
    pa=t.build_physics_args(args);p.apply_manifest_physics(args,pa,m)
    kwargs=dict(graph=bundle.graph,input_dim=len(bundle.train.input_feature_names),hidden_dim=256,temporal_hidden_dim=256,processor_layers=3,mlp_hidden_layers=1,prediction_mode='direct',we_nonneg=True,we_scale=args.we_scale,rres_scale=args.rres_scale,we_softplus_shift=args.we_softplus_shift)
    t.set_seed(42);baseline=t.JejuHiLAM(**kwargs,physics_heads=False)
    t.set_seed(42);model=t.JejuHiLAM(**kwargs,physics_heads=True)
    shared=dict(baseline.named_parameters());named=dict(model.named_parameters())
    assert all(torch.equal(v,named[k]) for k,v in shared.items())
    batch=default_collate([bundle.train[0]])
    with torch.no_grad():bpred=baseline(batch['x_nodes'],base_nodes=batch['tm_norm_nodes'])
    del baseline
    output=model(batch['x_nodes'],base_nodes=batch['tm_norm_nodes'],return_aux=True)
    torch.testing.assert_close(output['pred'],bpred,rtol=0,atol=0)
    grid=lambda x:t.nodes_to_grid_torch(x,bundle.graph)
    pred=grid(output['pred']*bundle.stats['target'][1]+bundle.stats['target'][0])
    loss_data=F.mse_loss(output['pred'],batch['y_nodes'])
    loss_phys=h.daily_rollout_physics_loss(pred,batch['Tm'],batch['y_raw'],batch['mask'],batch,pa,7,grid(output['we']),grid(output['rres']))
    loss_rres,_,_=h.identifiability_terms(grid(output['we']),grid(output['rres']),pred,batch,batch['mask'],7,pa,model.we_scale,model.rres_scale)
    params=list(named.values());pnames=list(named)
    components={}
    for label,loss in [('data',loss_data),('physics',loss_phys),('rres',loss_rres)]:
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gs if g is not None)
        components[label]={'loss':float(loss.detach()),'shared_gradient_norm':norm([g for n,g in zip(pnames,gs) if n in shared]),'we_head_gradient_norm':norm([g for n,g in zip(pnames,gs) if n.startswith('we_head.')]),'rres_head_gradient_norm':norm([g for n,g in zip(pnames,gs) if n.startswith('rres_head.')])}
    assert components['physics']['shared_gradient_norm']>0 and components['physics']['rres_head_gradient_norm']==0
    assert components['rres']['we_head_gradient_norm']==0
    arms={}
    for label,lp,lr in [('baseline',0.,0.),('physics_only',.1,0.),('auxiliary_only',0.,10.),('full_pinn',.1,10.)]:
        total=loss_data+lp*loss_phys+lr*loss_rres
        gs=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
        arm=dict(total_loss=float(total.detach()),all_gradients_finite=all(bool(torch.isfinite(g).all()) for g in gs if g is not None),we_head_gradient_norm=norm([g for n,g in zip(pnames,gs) if n.startswith('we_head.')]),rres_head_gradient_norm=norm([g for n,g in zip(pnames,gs) if n.startswith('rres_head.')]))
        if lp==0:assert arm['we_head_gradient_norm']==0
        if lr==0:assert arm['rres_head_gradient_norm']==0
        arms[label]=arm
    calls={'physics':0,'rres':0}
    def count_phys(*a,**kw):calls['physics']+=1;return h.daily_rollout_physics_loss(*a,**kw)
    def count_rres(*a,**kw):calls['rres']+=1;return h.identifiability_terms(*a,**kw)
    auxpa=copy.copy(pa);auxpa.lambda_rres=10.
    with torch.no_grad(),patch.object(t,'daily_rollout_physics_loss',count_phys),patch.object(t,'identifiability_terms',count_rres):
        t.run_epoch(model,[batch],bundle.graph,torch.device('cpu'),*bundle.stats['target'],optimizer=None,physics_args=auxpa,lambda_physics=0.,epoch=5)
    save('cpu_gradient_path.json',dict(status='PASS_CPU_DIAGNOSTIC_NOT_GPU_SMOKE',batch_size=1,hidden_dim=256,predictor_initial_weights_identical=True,predictor_output_identical=True,physics_rhs_scale=pa.physics_rhs_scale,components=components,arms=arms,aux_only_native_function_calls=calls,zero_weight_semantics='Finite inactive terms have zero gradient contribution but native run_epoch still evaluates both losses. 0*NaN is not protected.',optimizer_steps=0))
    print('CPU gradient path checked; no optimizer step',flush=True)
    bundle.ds.close()

if __name__=='__main__':main()
