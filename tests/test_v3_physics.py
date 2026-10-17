import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
import train_hybrid_jeju_forecast as h
import train_jeju_hilam as t
from mhw.ocean_stencil import ocean_neighbor_gradients, ocean_neighbor_support
from mhw.subset_preprocessing import p0_scale_args, apply_manifest_physics

BACK=Path('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset/scripts/physics_stencil.py')

def fixture():
    mask=torch.ones(1,1,4,5);mask[...,1,2]=0;mask[...,3,4]=0
    shape=mask.shape;tm=20*mask;z=torch.zeros(shape);lead=2
    b=dict(Tm=tm,y_raw=tm+.2*mask,mask=mask,dx=torch.full(shape,20000.),dy=torch.full(shape,20000.),
        qnet_seq=torch.ones(1,lead,1,4,5)*80,hm_seq=torch.ones(1,lead,1,4,5)*20,
        u_seq=torch.ones(1,lead,1,4,5)*.1,v_seq=torch.ones(1,lead,1,4,5)*.05,hm=torch.ones(shape)*20,
        vadv_seq=torch.ones(1,lead,1,4,5)*.01)
    args=p0_scale_args();args.__dict__.update(physics_clip=10.,physics_normalize='rhs',physics_rhs_scale=.1,
        physics_loss='mse',adaptive_sampling=False,adaptive_weighting=False,zero_rres_physics=True,
        lambda_rres=0.,lambda_we_tie=0.,rres_penalty_normalize='rhs')
    return b,args,z

class StencilTests(unittest.TestCase):
    def test_constant_land_nan_inf_and_backward(self):
        b,a,z=fixture();mask=b['mask']
        for fill in (0.,100.,float('nan'),float('inf')):
            field=torch.where(mask>0,20.,fill).requires_grad_()
            gx,gy=h.spatial_gradients_torch(field,b['dx'],b['dy'],mask)
            self.assertTrue(torch.isfinite(gx).all() and torch.isfinite(gy).all())
            self.assertEqual(float(gx.abs().sum()+gy.abs().sum()),0.)
            (gx.sum()+gy.sum()).backward();self.assertTrue(torch.isfinite(field.grad).all())
    def test_matches_original_pilot(self):
        tree=ast.parse(BACK.read_text());names={'shifted_neighbors','ocean_neighbor_support','ocean_neighbor_gradients'}
        env={'torch':torch,'F':F};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'pilot','exec'),env)
        gen=torch.Generator().manual_seed(8)
        for _ in range(8):
            field=torch.randn(2,1,8,9,generator=gen);mask=(torch.rand(2,1,8,9,generator=gen)>.35).float();spacing=torch.ones_like(field)*2
            for x,y in zip(ocean_neighbor_gradients(field,spacing,spacing,mask),env['ocean_neighbor_gradients'](field,spacing,spacing,mask)):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(ocean_neighbor_support(mask),env['ocean_neighbor_support'](mask),rtol=0,atol=0)
    def test_linear_field_and_axis_support(self):
        b,a,z=fixture();m=b['mask'];yy,xx=torch.meshgrid(torch.arange(4),torch.arange(5),indexing='ij');field=(2*xx+3*yy).float()[None,None]
        gx,gy=ocean_neighbor_gradients(field,torch.ones_like(m),torch.ones_like(m),m);valid=ocean_neighbor_support(m)>0
        self.assertTrue((gx[valid]==2).all() and (gy[valid]==3).all())
        isolated=torch.zeros_like(m);isolated[...,1,1]=1
        self.assertEqual(int(ocean_neighbor_support(isolated).sum()),0)
        for g in ocean_neighbor_gradients(field,m+1,m+1,isolated):self.assertEqual(float(g.abs().sum()),0)
    def test_degenerate_single_cell(self):
        f=torch.ones(1,1,1,1)*20;m=torch.ones_like(f)
        self.assertEqual(int(ocean_neighbor_support(m).sum()),0)
        for g in ocean_neighbor_gradients(f,m,m,m):self.assertEqual(float(g.sum()),0.)
    def test_p0_p1_loss_rhs_residual_same_stencil(self):
        b,a,z=fixture();pred=(b['Tm']+.3*b['mask']).requires_grad_()
        for components in (['original'],['original','vertical']):
            a.physics_component_list=components
            rhs,mask=h._linear_rhs_per_day(pred,b['Tm'],b['mask'],b,a,2,z)
            expected=((pred-b['Tm'])/2).repeat(2,1,1,1)-rhs
            expected=torch.where(mask>0,expected,torch.zeros_like(expected))
            loss=h.daily_rollout_physics_loss(pred,b['Tm'],b['y_raw'],b['mask'],b,a,2,z,z)
            manual=h.compute_physics_loss(expected,mask,a,rhs_per_day=rhs)
            torch.testing.assert_close(loss,manual,rtol=1e-5,atol=1e-6)
            stats=t.physics_raw_residual_stats(pred,b['Tm'],b['mask'],b,a,2,z,z)
            self.assertAlmostEqual(stats['common_res_sq_sum'],float((expected.square()*mask).sum()),places=5)
            diag=h.physics_budget_diagnostics(pred,b['Tm'],b['mask'],b,a,2,z,z)
            self.assertEqual(diag['budget_cells'],float(mask.sum()))
            self.assertTrue(torch.isfinite(torch.autograd.grad(loss,pred,retain_graph=True)[0]).all())
    def test_land_fill_invariance_entire_loss_and_scale(self):
        b,a,z=fixture();mask=b['mask'];pred=b['Tm']+.3*mask
        ref=h.daily_rollout_physics_loss(pred,b['Tm'],b['y_raw'],mask,b,a,2,z,z)
        scale=h.compute_fixed_rhs_scale([b],a,2,torch.device('cpu'))
        for fill in (999.,float('nan'),float('inf')):
            other=dict(b,Tm=torch.where(mask>0,b['Tm'],fill),y_raw=torch.where(mask>0,b['y_raw'],fill))
            out=h.daily_rollout_physics_loss(torch.where(mask>0,pred,fill),other['Tm'],other['y_raw'],mask,other,a,2,z,z)
            torch.testing.assert_close(out,ref,rtol=0,atol=0)
            self.assertEqual(h.compute_fixed_rhs_scale([other],a,2,torch.device('cpu')),scale)

class BranchTests(unittest.TestCase):
    def run_arm(self,lp,lr):
        b,args,_=fixture();idx=np.flatnonzero(b['mask'].numpy().ravel());graph=SimpleNamespace(height=4,width=5,ocean_flat_indices=idx)
        n=len(idx);pred=torch.ones(1,n,requires_grad=True);we=torch.full((1,n),1e-5,requires_grad=True);rr=torch.full((1,n),1e-5,requires_grad=True)
        b.update(y_nodes=torch.zeros(1,n),target_index=torch.tensor([31]),current_index=torch.tensor([29]))
        args.lambda_rres=lr;model=SimpleNamespace(we_scale=1e-5,rres_scale=1e-4)
        # Poison inactive output and function: neither may contaminate objective.
        output=dict(pred=pred,we=we if lp else we*float('nan'),rres=rr if lr else rr*float('nan'))
        with patch.object(t,'daily_rollout_physics_loss',wraps=h.daily_rollout_physics_loss) as ph,patch.object(t,'rres_penalty',wraps=h.rres_penalty) as rh,patch.object(h,'_linear_rhs_per_day',side_effect=AssertionError('Auxiliary penalty attempted RHS recompute')):
            losses=t.compute_batch_losses(output,b,graph,20.,1.,args,lp,model)
        self.assertEqual(ph.call_count,int(lp!=0));self.assertEqual(rh.call_count,int(lr!=0))
        self.assertTrue(torch.isfinite(losses[-1]))
        gs=torch.autograd.grad(losses[-1],(pred,we,rr),allow_unused=True,retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in gs if g is not None))
        if not lp:self.assertIsNone(gs[1])
        if not lr:self.assertIsNone(gs[2])
        if lp:self.assertGreater(float(torch.autograd.grad(losses[2],pred)[0].abs().sum()),0.)
    def test_baseline(self):self.run_arm(0.,0.)
    def test_physics_only(self):self.run_arm(.1,0.)
    def test_auxiliary_only(self):self.run_arm(0.,10.)
    def test_full(self):self.run_arm(.1,10.)
    def test_inactive_loss_functions_never_called(self):
        pred=torch.ones(1,2,requires_grad=True)
        with patch.object(t,'daily_rollout_physics_loss',return_value=torch.tensor(float('nan'))) as ph,patch.object(t,'rres_penalty',return_value=torch.tensor(float('nan'))) as rr:
            vals=t.compute_batch_losses(pred,{'y_nodes':torch.zeros_like(pred)},None,0.,1.,None,0.,None)
        self.assertTrue(torch.isfinite(vals[-1]));ph.assert_not_called();rr.assert_not_called()
    def test_legacy_scale_rejected_for_physics(self):
        args=SimpleNamespace(history=30,lead=7,physics_mode='p0_original',hm_min=5.,entrainment_dt=1.,physics_rhs_scale=None)
        for schema in ('subset_normalization_v1','subset_normalization_v2'):
            with self.assertRaisesRegex(ValueError,'Legacy RHS'):apply_manifest_physics(args,SimpleNamespace(),dict(schema=schema,history=30,lead=7,physics_rhs_scale=10.))

if __name__=='__main__':unittest.main()
