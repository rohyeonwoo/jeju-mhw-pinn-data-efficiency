"""CPU checks on real JejuHiLAM: zero-loss parity and auxiliary-loss separation."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_data_efficiency import original, original_args, verify_inputs, select_training_data, save


class CaptureAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        self.grads = [None if p.grad is None else p.grad.detach().clone()
                      for group in self.param_groups for p in group['params']]
        return super().step(closure)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args_out = p.parse_args()
    torch.set_num_threads(4)
    data, _, payload, full, _ = verify_inputs(42, 7)
    bundle = original.make_jeju_hilam_bundle(data, None, 30, 7, include_residual_feature=False)
    selected, _ = select_training_data(bundle, payload, 10)
    batch = next(iter(DataLoader(selected, batch_size=2, shuffle=False)))
    common = dict(graph=bundle.graph, device=torch.device('cpu'),
                  target_mean=bundle.stats['target'][0], target_std=bundle.stats['target'][1], precision='fp32')
    results = []
    for seed in (42, 7, 123):
        args = original_args(7, seed, 'p0_original')
        args.physics_rhs_scale = full['full_rhs_scale']
        kwargs = dict(graph=bundle.graph, input_dim=len(bundle.train.input_feature_names),
                      hidden_dim=args.hidden_dim, temporal_hidden_dim=args.temporal_hidden_dim,
                      processor_layers=args.processor_layers, mlp_hidden_layers=args.mlp_hidden_layers,
                      prediction_mode=args.prediction_mode, we_nonneg=True, we_scale=args.we_scale,
                      rres_scale=args.rres_scale, we_softplus_shift=args.we_softplus_shift)
        original.set_seed(seed)
        baseline = original.JejuHiLAM(physics_heads=False, **kwargs)
        baseline_rng = torch.get_rng_state().clone()
        original.set_seed(seed)
        pinn = original.JejuHiLAM(physics_heads=True, **kwargs)
        pinn_rng = torch.get_rng_state().clone()
        shared_names = list(dict(baseline.named_parameters()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, pinn.state_dict()[name], rtol=0, atol=0)
        physics = original.build_physics_args(args)
        off = copy.deepcopy(physics)
        off.lambda_rres = 0.0
        off.lambda_we_tie = 0.0
        oa = CaptureAdamW(baseline.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay, betas=(.9,.95))
        ob = CaptureAdamW(pinn.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay, betas=(.9,.95))
        max_param, max_grad = 0.0, 0.0
        for step in range(2):
            original.set_epoch_lr(oa, args, 1)
            original.set_epoch_lr(ob, args, 1)
            sa = original.run_epoch(baseline, [batch], optimizer=oa, epoch=1,
                                    physics_args=None, lambda_physics=0.0, **common)
            sb = original.run_epoch(pinn, [batch], optimizer=ob, epoch=1,
                                    physics_args=off, lambda_physics=0.0, **common)
            assert abs(sa['data_loss']-sb['data_loss']) < 1e-7
            assert abs(sb['total_loss']-sb['data_loss']) < 1e-7
            for index, name in enumerate(shared_names):
                a = dict(baseline.named_parameters())[name]
                b = dict(pinn.named_parameters())[name]
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(oa.grads[index], ob.grads[index], rtol=1e-5, atol=1e-6)
                max_param = max(max_param, float((a-b).detach().abs().max()))
                if oa.grads[index] is not None:
                    max_grad = max(max_grad, float((oa.grads[index]-ob.grads[index]).abs().max()))
        with torch.no_grad():
            a = baseline(batch['x_nodes'], base_nodes=batch['tm_norm_nodes'])
            b = pinn(batch['x_nodes'], base_nodes=batch['tm_norm_nodes'], return_aux=True)['pred']
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
            losses = {}
            for lam in (0., .01, .1, .3):
                s = original.run_epoch(pinn, [batch], optimizer=None, epoch=5,
                                       physics_args=physics, lambda_physics=lam, **common)
                expected = s['data_loss'] + lam*s['physics_loss'] + 10*s['rres_l2']
                assert abs(s['total_loss']-expected) < 1e-5
                losses[str(lam)] = {k:s[k] for k in ['data_loss','physics_loss','rres_l2','total_loss','lambda_physics_effective']}
            assert losses['0.0']['rres_l2'] > 0
            assert losses['0.0']['total_loss'] > losses['0.0']['data_loss']
        # Separately confirm that the auxiliary loss has a path to shared features.
        pinn.zero_grad(set_to_none=True)
        outputs = pinn(batch['x_nodes'], base_nodes=batch['tm_norm_nodes'], return_aux=True)
        grid = lambda x: original.nodes_to_grid_torch(x, bundle.graph)
        rres, _, _ = original.identifiability_terms(
            we_pred=grid(outputs['we']), rres_pred=grid(outputs['rres']),
            pred_raw=grid(outputs['pred']*common['target_std']+common['target_mean']),
            batch=batch, mask=batch['mask'], lead_days=7, args=physics,
            we_scale=args.we_scale, rres_scale=args.rres_scale)
        params = [dict(pinn.named_parameters())[k] for k in shared_names]
        grads = torch.autograd.grad(10*rres, params, allow_unused=True)
        aux_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)))
        assert aux_norm > 0
        results.append(dict(seed=seed, shared_initialization_exact=True,
                            max_shared_parameter_difference=max_param, max_clipped_gradient_difference=max_grad,
                            final_prediction_max_difference=float((a-b).abs().max()),
                            auxiliary_shared_gradient_norm=aux_norm,
                            post_init_rng_equal=bool(torch.equal(baseline_rng,pinn_rng)), losses=losses))
        print('CPU parity/auxiliary-loss checks passed:',seed,flush=True)
        del baseline, pinn, oa, ob
    report = dict(status='passed', device='cpu', lead=7, batch_size=2, updates_per_seed=2,
                  scope='real graph and full-size original model; fixed identical batch, all extra loss weights zero for parity',
                  caveat='Legacy baseline/P0 share seed and shared initialization but head creation consumes RNG; historical shuffled batch orders need not coincide. New P0 arms preserve legacy P0 RNG.',
                  results=results)
    args_out.output.parent.mkdir(parents=True,exist_ok=True)
    save(args_out.output, report)
    bundle.ds.close()
    print('REPORT',args_out.output,flush=True)


if __name__ == '__main__':
    main()
