"""RNG-neutral initial gradient measurements for the controlled physics pilot."""
import hashlib
import numpy as np
import torch
from torch.utils.data import default_collate
import train_jeju_hilam as original


def state_digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gradient_diagnostic(model, selected, graph, physics, device, mean, std, lead, lam):
    was_training = model.training
    # A fixed, train-only batch, no DataLoader iterator and no RNG consumption.
    indices = np.unique(np.linspace(0, len(selected)-1, 16).astype(int))
    batch = default_collate([selected[int(i)] for i in indices])
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model.train()  # cuDNN recurrent backward requires training mode; RNG is restored below
        output = model(batch['x_nodes'].to(device), base_nodes=batch['tm_norm_nodes'].to(device), return_aux=True)
        data_loss = torch.nn.functional.mse_loss(output['pred'], batch['y_nodes'].to(device))
        grid = lambda x: original.nodes_to_grid_torch(x, graph)
        physics_loss = original.daily_rollout_physics_loss(
            grid(output['pred']*std+mean), batch['Tm'].to(device), batch['y_raw'].to(device),
            batch['mask'].to(device), batch, physics, lead, grid(output['we']), grid(output['rres']))
        params = [p for name, p in model.named_parameters() if not name.startswith(('we_head.', 'rres_head.'))]
        gd = torch.autograd.grad(data_loss, params, retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(physics_loss, params, allow_unused=True)
        nd = sum(float(g.detach().double().square().sum()) for g in gd if g is not None)**.5
        np_ = sum(float(g.detach().double().square().sum()) for g in gp if g is not None)**.5
        dot = sum(float((d.detach().double()*p.detach().double()).sum())
                  for d,p in zip(gd,gp) if d is not None and p is not None)
        result = dict(data_loss=float(data_loss.detach()), physics_loss=float(physics_loss.detach()),
                      data_gradient_norm=nd, physics_gradient_norm=np_,
                      gradient_cosine=dot/max(nd*np_, 1e-30),
                      weighted_physics_to_data_gradient=lam*np_/max(nd, 1e-30),
                      lambda_before_existing_warmup=lam,
                      train_target_indices=batch['target_index'].tolist(),
                      scope='Initial checkpoint, 16 fixed train samples, shared parameters; diagnostic only, no policy changes')
        assert all(np.isfinite(v) for v in result.values() if isinstance(v, float))
    model.train(was_training)
    return result
