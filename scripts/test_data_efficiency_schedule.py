"""Check batch continuity against ordinary loaders and original training functions."""
import itertools
import json
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader
from train_data_efficiency import BatchStream, original, original_args


def loader(seed=42):
    return DataLoader([{'target_index': i} for i in range(19)], batch_size=8,
                      shuffle=True, generator=torch.Generator().manual_seed(seed))


def values(batches):
    return [b['target_index'].tolist() for b in batches]


# Independent reference: repeated ordinary shuffled passes, including 3-item tail.
reference_loader = loader()
reference = []
for _ in range(5):
    reference.extend(values(reference_loader))
stream = BatchStream(loader())
actual = values(stream.take(4)) + values(stream.take(5)) + values(stream.take(6))
assert actual == reference
assert [len(b) for b in actual] == [8, 8, 3] * 5
assert stream.completed_steps == 15 and stream.completed_samples == 95

# Exercise the original run_epoch, not a reimplemented optimizer loop.
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.25))
    def forward(self, x, base_nodes=None, return_aux=False):
        return x * self.w
samples = [{'x_nodes': torch.tensor([float(i+1)]), 'y_nodes': torch.tensor([0.]),
            'y_raw_nodes': torch.tensor([0.]), 'tm_norm_nodes': torch.tensor([0.]),
            'target_index': i} for i in range(5)]
def tiny_loader():
    return DataLoader(samples,batch_size=2,shuffle=False)
a,b=Toy(),Toy()
oa=torch.optim.AdamW(a.parameters(),lr=0.001)
ob=torch.optim.AdamW(b.parameters(),lr=0.001)
s=BatchStream(tiny_loader())
kwargs=dict(graph=None,device=torch.device('cpu'),target_mean=0.,target_std=1.)
original.run_epoch(a,s.take(2),optimizer=oa,**kwargs)
original.run_epoch(a,s.take(4),optimizer=oa,**kwargs)
for _ in range(2):
    original.run_epoch(b,tiny_loader(),optimizer=ob,**kwargs)
torch.testing.assert_close(a.w,b.w,rtol=0,atol=0)
assert s.completed_steps==6
assert int(oa.state[a.w]['step'])==6

for lead, S, batch_size in ((7,956,8),(14,955,8),(7,478,16),(14,478,16)):
    args=original_args(lead,42,'p0_original')
    args.batch_size=batch_size
    assert S==({7:7644,14:7639}[lead]+batch_size-1)//batch_size
    opt=torch.optim.AdamW(Toy().parameters(),lr=args.initial_lr)
    for completed,epoch in ((0,1),(S-1,1),(S,2),(5*S-1,5),(5*S,6),(80*S-1,80)):
        assert completed//S+1==epoch
        lr=original.set_epoch_lr(opt,args,epoch)
        factor=original.physics_lambda_factor(epoch,5)
        assert factor==min(epoch/5,1.)
        if epoch==1:
            assert abs(lr-5.6e-5)<1e-12
        if epoch==5:
            assert lr==args.lr
        if epoch==80:
            assert lr==0.
print(json.dumps({'status':'passed','checks':['persistent shuffled iterator','partial final batches',
      'no skipped batch at validation','original run_epoch optimizer equivalence',
      'Full-step LR/physics warmup boundaries at both leads']}))
