"""Analytic and autograd checks for the new period-mean physical loss."""
import copy
import json
import torch
from train_lambda_pilot import original, original_args, save, ROOT
from physics_interval import interval_physics_loss
import train_hybrid_jeju_forecast as hybrid


def main():
    torch.set_num_threads(2)
    args = original.build_physics_args(original_args(7, 42, 'p0_original'))
    args.physics_rhs_scale = 4.
    args.physics_clip = 0.
    shape = (1,1,4,5)
    t0 = torch.full(shape, 20., dtype=torch.float64)
    pred = torch.full(shape, 22., dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(t0)
    zero = torch.zeros_like(t0)
    ones = torch.ones((1,2,1,4,5), dtype=torch.float64)
    batch = dict(qnet_seq=ones*(1025*3990*10/86400), hm_seq=ones*10,
                 u_seq=ones*0, v_seq=ones*0, dx=t0*0+20000, dy=t0*0+20000,
                 vadv_seq=ones*.2)
    def loss(fn, par=pred, pa=args):
        return fn(par,t0,pred.detach(),mask,batch,pa,2,zero,zero)
    # +1 K/day constant forcing exactly matches +2 K in 2 days.
    assert abs(float(loss(interval_physics_loss).detach())) < 1e-20
    assert abs(float(loss(hybrid.daily_rollout_physics_loss).detach())) < 1e-20
    # Alternating 0/2 K/day has identical period budget, daily MSE is 1/scale.
    batch['qnet_seq'][:,0] *= 0
    batch['qnet_seq'][:,1] *= 2
    li, ld = loss(interval_physics_loss), loss(hybrid.daily_rollout_physics_loss)
    torch.testing.assert_close(li, torch.zeros_like(li), atol=1e-20, rtol=0)
    torch.testing.assert_close(ld, ld.new_tensor(.25), atol=1e-12, rtol=0)
    # Off the optimum, additive daily forcing variation alone does not change gradients.
    other = torch.full_like(pred,23.,requires_grad=True)
    gi = torch.autograd.grad(loss(interval_physics_loss,other),other)[0]
    gd = torch.autograd.grad(loss(hybrid.daily_rollout_physics_loss,other),other)[0]
    torch.testing.assert_close(gi,gd,atol=1e-12,rtol=0)
    # P1's added +.2 K/day gives -.2 residual, with the correct sign and units.
    p1=copy.deepcopy(args);p1.physics_component_list=['original','vertical']
    torch.testing.assert_close(loss(interval_physics_loss,pa=p1),li.new_tensor(.01),atol=1e-12,rtol=0)
    # End-state-dependent spatial advection yields genuinely different gradients.
    spatial=(pred.detach()+torch.arange(20,dtype=pred.dtype).reshape(shape)*.02).requires_grad_()
    batch['u_seq'][:,0]=.15;batch['u_seq'][:,1]=-.1
    gi=torch.autograd.grad(loss(interval_physics_loss,spatial),spatial)[0]
    gd=torch.autograd.grad(loss(hybrid.daily_rollout_physics_loss,spatial),spatial)[0]
    gap=float((gi-gd).abs().max());assert gap>1e-7
    assert torch.autograd.gradcheck(lambda x:loss(interval_physics_loss,x),(spatial,),eps=1e-6,atol=1e-5)
    report=dict(status='passed', constant_budget_exact=True, alternating_forcing_exact=True,
                independent_forcing_gradient_identity=True, vertical_sign_and_units=True,
                state_dependent_gradient_max_difference=gap, finite_difference_gradcheck=True,
                notes='All synthetic checks use no clipping; production retains legacy clipping and fixed Full P0 scale')
    save(ROOT/'docs/server_preflight/physics_interval_cpu_check.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
