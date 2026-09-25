"""Analytic ocean-boundary derivatives and invariance to land fill values."""
import torch
from train_lambda_pilot import save, ROOT
from physics_stencil import ocean_neighbor_gradients, ocean_neighbor_support, valid_ocean_stencil


def main():
    torch.set_num_threads(2)
    mask=torch.ones((1,1,7,8),dtype=torch.float64)
    mask[...,2:4,3:5]=0
    y,x=torch.meshgrid(torch.arange(7,dtype=torch.float64),torch.arange(8,dtype=torch.float64),indexing='ij')
    field=(10+3*x+2*y)[None,None]
    dx=torch.ones_like(field)*2;dy=torch.ones_like(field)*4
    valid=ocean_neighbor_support(mask).bool()
    checks=[]
    for land in [0.,-100.,1000.]:
        values=torch.where(mask>0,field,field*0+land)
        gx,gy=ocean_neighbor_gradients(values,dx,dy,mask)
        torch.testing.assert_close(gx[valid],torch.full_like(gx[valid],1.5),rtol=0,atol=0)
        torch.testing.assert_close(gy[valid],torch.full_like(gy[valid],.5),rtol=0,atol=0)
        checks.append((gx,gy))
    for gx,gy in checks[1:]:
        torch.testing.assert_close(gx,checks[0][0],rtol=0,atol=0)
        torch.testing.assert_close(gy,checks[0][1],rtol=0,atol=0)
    value=torch.where(mask>0,field,field*0).requires_grad_()
    assert torch.autograd.gradcheck(lambda z:ocean_neighbor_gradients(z,dx,dy,mask),(value,),eps=1e-6,atol=1e-6)
    gx,gy=ocean_neighbor_gradients(value,dx,dy,mask)
    grad=torch.autograd.grad((gx[valid].square()+gy[valid].square()).sum(),value)[0]
    assert (grad[mask==0]==0).all()
    isolated=torch.zeros_like(mask);isolated[...,3,3]=1
    assert ocean_neighbor_support(isolated).sum()==0
    report=dict(status='passed',linear_field_derivatives_exact=True,land_fill_invariance=True,
                land_gradient_exactly_zero=True,unsupported_cell_excluded=True,autograd_finite_difference_passed=True,
                accuracy='second-order centered when both neighbors are ocean; first-order one-sided with one ocean neighbor')
    save(ROOT/'docs/server_preflight/ocean_neighbor_gradient_check.json',report)
    print(report)


if __name__=='__main__':main()
