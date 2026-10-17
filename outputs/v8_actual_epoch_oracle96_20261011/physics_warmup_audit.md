# Physics warmup PASS

Original src/train_hybrid_jeju_forecast.py:524 factor=min(epoch/warmup,1); src/train_jeju_hilam.py:323 applies lambda once, :288 adds weighted loss once. Original Oracle96 worker:79 overrides warmup0. New config5 gives factors0.2/0.4/0.6/0.8/1; effective lambda0.02/0.04/0.06/0.08/0.1. Epoch6–80 remains0.1. Baseline0 skips physics. CPU real predictor/loss gradient checks on leads7/14 PASS; physics_entrypoint_cpu.json. NoAdv/Adv invoke same effective lambda, no other warmup. GPU confirmation PENDING until smoke gate.
