"""Restrict physical derivatives to stencils containing only ocean cells."""
import torch
import torch.nn.functional as F
import train_hybrid_jeju_forecast as hybrid
from physics_interval import interval_physics_loss


def valid_ocean_stencil(mask):
    valid = mask > 0
    valid = valid.clone()
    ocean = mask > 0
    valid[..., 1:-1] &= ocean[..., :-2] & ocean[..., 2:]
    valid[..., 0] &= ocean[..., 1]
    valid[..., -1] &= ocean[..., -2]
    valid[..., 1:-1, :] &= ocean[..., :-2, :] & ocean[..., 2:, :]
    valid[..., 0, :] &= ocean[..., 1, :]
    valid[..., -1, :] &= ocean[..., -2, :]
    return valid.to(mask.dtype)


def loss_function(arm, stencil):
    base = interval_physics_loss if arm == 'p0_interval' else hybrid.daily_rollout_physics_loss
    if stencil == 'legacy':
        return base
    if stencil not in ('ocean_valid','ocean_neighbors'):
        raise ValueError(stencil)
    support = ocean_neighbor_support if stencil == 'ocean_neighbors' else valid_ocean_stencil
    def masked(pred_raw, Tm, target_raw, mask, batch, args, lead_days, we_pred, rres_pred):
        return base(pred_raw,Tm,target_raw,support(mask),batch,args,
                    lead_days,we_pred,rres_pred)
    return masked


def shifted_neighbors(field):
    return (F.pad(field[..., :-1], (1,0,0,0)),
            F.pad(field[..., 1:], (0,1,0,0)),
            F.pad(field[..., :-1, :], (0,0,1,0)),
            F.pad(field[..., 1:, :], (0,0,0,1)))


def ocean_neighbor_support(mask):
    left,right,up,down=shifted_neighbors(mask > 0)
    return ((mask>0) & (left|right) & (up|down)).to(mask.dtype)


def ocean_neighbor_gradients(field, dx, dy, mask):
    ocean=mask.to(field.device)>0
    ml,mr,mu,md=shifted_neighbors(ocean)
    left,right,up,down=shifted_neighbors(field)
    def derivative(lo,hi,has_lo,has_hi,spacing):
        numerator=torch.where(has_lo & has_hi,(hi-lo)*.5,
                              torch.where(has_hi,hi-field,
                                          torch.where(has_lo,field-lo,torch.zeros_like(field))))
        safe=torch.where(spacing.abs()>1e-12,spacing,torch.ones_like(spacing))
        return torch.where(ocean,numerator/safe,torch.zeros_like(field))
    return derivative(left,right,ml,mr,dx), derivative(up,down,mu,md,dy)


def configure_spatial_gradients(stencil, mask):
    if stencil == 'ocean_neighbors':
        hybrid.spatial_gradients_torch = lambda field,dx,dy: ocean_neighbor_gradients(field,dx,dy,mask)
