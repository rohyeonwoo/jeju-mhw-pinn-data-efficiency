"""Train-only observed budget audit. CPU only; no model fitting or test access."""
import argparse
import json
import numpy as np
import torch
from train_lambda_pilot import original, original_args, verify_inputs, save
from train_hybrid_jeju_forecast import spatial_gradients_torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    data, _, _, full, _ = verify_inputs(42, 7)
    bundle = original.make_jeju_hilam_bundle(data, None, 30, 7, include_residual_feature=False)
    base = bundle.train.base
    starts = bundle.train_indices[np.unique(np.linspace(0, len(bundle.train)-1, 512).astype(int))]
    assert np.isin(starts, bundle.train_indices).all()
    assert (starts + 36).max() < (bundle.val_indices + 36).min()
    mask = base.ocean_mask.astype(bool)
    interior = mask.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    interior[1:-1, 1:-1] &= (mask[:-2, 1:-1] & mask[2:, 1:-1]
                            & mask[1:-1, :-2] & mask[1:-1, 2:])
    regions = {'ocean': mask, 'interior': interior, 'coastal_or_edge': mask & ~interior}
    pools = {}
    def collect(path, name, value):
        for region, m in regions.items():
            v = value[..., m].reshape(-1).astype(np.float64)
            assert np.isfinite(v).all(), (path, name)
            pools.setdefault((path, name, region), []).append(v)
    conv = lambda x: np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    sst = conv(base.target_array)
    dx, dy = torch.from_numpy(base.dx), torch.from_numpy(base.dy)
    phys = {k: conv(v) for k, v in base.physics_arrays.items()}
    for offset in range(0, len(starts), 32):
        current = starts[offset:offset+32] + 29
        t0, tn = sst[current], sst[current+7]
        for path in ['observed_daily', 'observed_linear', 'node_zero_land_linear']:
            endpoint = tn if path != 'node_zero_land_linear' else tn * mask
            daily = {k: [] for k in ['tendency', 'flux', 'advection', 'vertical', 'p0_residual', 'p1_residual']}
            for step in range(7):
                prev = sst[current+step] if path == 'observed_daily' else t0 + (endpoint-t0)*(step/7)
                nxt = sst[current+step+1] if path == 'observed_daily' else t0 + (endpoint-t0)*((step+1)/7)
                gx, gy = spatial_gradients_torch(torch.from_numpy(prev), dx, dy)
                h = np.maximum(phys['mlotst'][current+step], 5.)
                flux = 86400 * phys['qnet'][current+step] / (1025*3990*h)
                adv = -86400*(phys['u_ml'][current+step]*gx.numpy()+phys['v_ml'][current+step]*gy.numpy())
                vert = phys['vadv_kday'][current+step]
                tendency = nxt-prev
                values = dict(tendency=tendency, flux=flux, advection=adv, vertical=vert,
                              p0_residual=tendency-flux-adv, p1_residual=tendency-flux-adv-vert)
                for k, v in values.items():
                    daily[k].append(v)
                    collect(path, k, v)
            for k in ['p0_residual', 'p1_residual']:
                collect(path, k+'_period_mean', np.mean(daily[k], axis=0))
    results = []
    for (path, term, region), pieces in pools.items():
        v = np.concatenate(pieces)
        results.append(dict(path=path, term=term, region=region, n=int(v.size),
                            mean=float(v.mean()), rms=float(np.sqrt(np.mean(v*v))),
                            abs_p95=float(np.quantile(np.abs(v), .95)),
                            fraction_abs_gt10=float(np.mean(np.abs(v)>10))))
    args = original_args(7, 42, 'p0_original')
    report = dict(status='completed', units='K/day', sample_count=len(starts),
                  start_indices=starts.tolist(), region_cells={k:int(v.sum()) for k,v in regions.items()},
                  source='512 evenly spaced original train windows; train-only diagnostic, not subset-only information',
                  paths='observed daily; linear raw observed endpoint; linear ocean-node endpoint with zero land fill matching actual predictor',
                  entrainment='we=0 isolates resolved forcing; learned we not diagnosed here',
                  forcing='legacy current day through current+6 for T(current+7)-T(current); no dates changed',
                  current_physics_clip=args.physics_clip, fixed_common_p0_rhs_scale=full['full_rhs_scale'],
                  source_attrs={k:{n:str(v) for n,v in bundle.ds[k].attrs.items()} for k in ['sst_mean','qnet','mlotst','u_ml','v_ml','vadv_kday']},
                  results=results)
    save(a.output, report)
    for row in results:
        if row['term'] in ['tendency','p0_residual','p1_residual','p0_residual_period_mean'] and row['region']=='ocean':
            print(json.dumps(row), flush=True)
    bundle.ds.close()


if __name__ == '__main__':
    main()
