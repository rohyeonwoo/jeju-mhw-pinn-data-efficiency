# CPU preflight PASS

Sampling24 conditions×80×4 PASS; nested/boundary manifests PASS; physics gradient both leads PASS; actual native run_epoch training loop80 epochs PASS; interrupted afterepoch3 / fresh-process-equivalent restore reproduces weights bitwise and histories/order exactly; best and post-warmup reload PASS; NaN rejection PASS. Protected input/old-output hashes PASS. New worker has no evaluate phase and controller never executes Test.

GPU smoke and representative actual data training are still PENDING; not claimed as PASS. See preflight.json and regression artifacts. Original science source hashes preserved; changed protocol code isolated in scripts/oracle96_actual/.

## 실제 코드 위치

- `scripts/oracle96_actual/worker.py:13–23` — `arguments`
- `scripts/oracle96_actual/worker.py:25–37` — `setup`
- `scripts/oracle96_actual/worker.py:39–50` — `CheckedEpoch`
- `scripts/oracle96_actual/worker.py:52–63` — `Monitor`
- `scripts/oracle96_actual/worker.py:65–66` — `rng_state`
- `scripts/oracle96_actual/worker.py:67–70` — `restore_rng`
- `scripts/oracle96_actual/worker.py:72–132` — `run_training`
- `scripts/oracle96_actual/worker.py:134–169` — `main`
- `scripts/oracle96_actual/controller.py:6–9` — `ident`
- `scripts/oracle96_actual/controller.py:11–12` — `job_status`
- `scripts/oracle96_actual/controller.py:14–34` — `gate`
- `scripts/oracle96_actual/controller.py:36–39` — `resource_rows`
- `scripts/oracle96_actual/controller.py:41–44` — `eligible`
- `scripts/oracle96_actual/controller.py:46–50` — `publish`
- `scripts/oracle96_actual/controller.py:52–92` — `stage`
- `scripts/oracle96_actual/controller.py:94–100` — `acquire`
- `scripts/oracle96_actual/controller.py:102–125` — `main`
- `scripts/oracle96_actual/preflight.py:10–10` — `doc`
- `scripts/oracle96_actual/preflight.py:11–38` — `test_loop_resume`
- `scripts/oracle96_actual/preflight.py:40–143` — `main`
- `scripts/oracle96_actual/summarize.py:9–11` — `csvout`
- `scripts/oracle96_actual/summarize.py:12–12` — `table`
- `scripts/oracle96_actual/summarize.py:13–70` — `main`

GPU resource policy synthetic checks PASS. Existing Validation/MHW reference copied byte-for-byte (SHA checked), without recomputation. No Test evaluation command exists in the worker CLI.
