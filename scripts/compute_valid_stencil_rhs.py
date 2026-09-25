"""Full-train P0 RHS statistics on physically valid ocean derivative stencils."""
import argparse
import numpy as np
import torch
from train_lambda_pilot import original, original_args, verify_inputs, save, ROOT, sha
import train_hybrid_jeju_forecast as hybrid
from physics_stencil import valid_ocean_stencil, ocean_neighbor_support, ocean_neighbor_gradients


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stencil',choices=['ocean_valid','ocean_neighbors'],required=True)
    opt=parser.parse_args()
    torch.set_num_threads(4)
    data, _, _, audit, _ = verify_inputs(42,7)
    bundle=original.make_jeju_hilam_bundle(data,None,30,7,include_residual_feature=False)
    base=bundle.train.base
    args=original.build_physics_args(original_args(7,42,'p0_original'))
    tensor=lambda v:torch.from_numpy(np.nan_to_num(v,nan=0.,posinf=0.,neginf=0.))
    ocean=tensor(base.ocean_mask)[None,None]
    valid=(ocean_neighbor_support if opt.stencil=='ocean_neighbors' else valid_ocean_stencil)(ocean)
    sq=0.;count=0.;legacy_sq=0.;legacy_count=0
    with torch.no_grad():
        for offset in range(0,len(bundle.train_indices),64):
            current=bundle.train_indices[offset:offset+64]+29
            idx=current[:,None]+np.arange(7)[None,:]
            b={k:tensor(base.physics_arrays[v][idx])[:,:,None] for k,v in
               [('qnet_seq','qnet'),('hm_seq','mlotst'),('u_seq','u_ml'),('v_seq','v_ml')]}
            b.update(dx=tensor(base.dx)[None,None],dy=tensor(base.dy)[None,None])
            t0=tensor(base.target_array[current])[:,None]
            tn=tensor(base.target_array[current+7])[:,None]
            m=valid.expand_as(t0)
            rhs,rm=hybrid._linear_rhs_per_day(tn,t0,m,b,args,7,torch.zeros_like(t0))

            om=ocean.expand_as(t0).repeat(7,1,1,1)
            legacy_sq+=float((rhs.double().square()*om).sum());legacy_count+=float(om.sum())
            if opt.stencil=='ocean_neighbors':
                old=hybrid.spatial_gradients_torch
                try:
                    hybrid.spatial_gradients_torch=lambda field,dx,dy:ocean_neighbor_gradients(field,dx,dy,ocean)
                    rhs,rm=hybrid._linear_rhs_per_day(tn,t0,m,b,args,7,torch.zeros_like(t0))
                finally:
                    hybrid.spatial_gradients_torch=old
            sq+=float((rhs.double().square()*rm).sum());count+=float(rm.sum())
    legacy=legacy_sq/legacy_count
    assert np.isclose(legacy,audit['full_rhs_scale'],rtol=1e-5), (legacy,audit['full_rhs_scale'])
    result=dict(status='passed',stencil=opt.stencil,lead=7,full_train_samples=len(bundle.train),
                physics_rhs_scale=sq/count,legacy_rhs_scale_recomputed=legacy,
                legacy_rhs_scale=audit['full_rhs_scale'],ocean_cells=int(ocean.sum()),
                valid_stencil_cells=int(valid.sum()),split_sha256=audit['split_sha256'],
                data_sha256=sha(data),stencil_sha256=sha(ROOT/'scripts/physics_stencil.py'),
                basis='Full train observed linear endpoints, we=0, P0 resolved RHS, K/day squared; same scale for P0/P1 and daily/period mean',
                scope='Only physical-loss RHS scale; input/target normalization unchanged')
    save(ROOT/f'docs/server_preflight/{opt.stencil}_rhs_lead7.json',result)
    bundle.ds.close();print(result,flush=True)


if __name__=='__main__':main()
