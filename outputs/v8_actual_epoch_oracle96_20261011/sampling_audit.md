# Sampling PASS

24 manifests × 4 model-order replicas × 80 epochs: exact once, no missing/duplicate, last batch retained, reproducible shuffle. Synthetic sizes1/15/16/17/32/33 PASS. Details sampling_audit.json.

10%376→24→1920;25%987→62→4960;50%1974→124→9920;100%3995→250→20000. 100% is blocked subset, not Full train.

Implementation: scripts/oracle96_actual/worker.py CheckedEpoch; native src/train_jeju_hilam.py:215 DataLoader; scripts/train_v3_study.py:53 old cycling iterator is not called by new training loop.
