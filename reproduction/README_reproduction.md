# Oracle120 subset experiment: reproduction bundle

This export contains the frozen scientific source, subset/protocol manifests, configuration and training histories, final metrics, statistical evidence, publication figures and analysis code. Large ocean data, checkpoints and prediction arrays remain outside GitHub. This is a curated publication source snapshot, not a complete backup of Ceph.

## Start here

- Final analysis: `outputs/paper_analysis_20261016/README.md`
- Seven research questions: `outputs/paper_analysis_20261016/reports/analysis_summary_ko.md`
- Protocol: `outputs/paper_final_verification_20261015/experiment_protocol_audit.md`
- Exact retained and proposed-cleanup files: `reproduction/retention_inventory.csv`, `cleanup_candidates.csv`
- Exported file checksums: `GITHUB_EXPORT_SHA256SUMS`

## Scope

5 models (Baseline, Auxiliary-only, P0, NoAdv, Adv), 4 nested subsets (10/25/50/100%), leads 7/14, seeds42/7/123. 120 runs,80 actual epochs, Validation-best. The 100% subset is3995 retained windows, not all original training samples. Test has been examined previously; conclusions are exploratory. Older WSDM experiments remain separate.

## Reproduction levels

1. **Review the reported results:** final CSV/LaTeX tables and figures/captions are included. No training/data download is needed to inspect them.
2. **Recompute full analysis from frozen predictions:** requires the Ceph-retained final120 predictions/daily MHW labels, original metadata and physical features, in addition to this repository. `outputs/paper_analysis_20261016/code/analyze.py`, `refine_figures.py`, `finalize_findings.py` reproduce the analysis.
3. **Repeat training/evaluation:** requires budget2d raw NetCDF, external3D GLORYS source (or the exact retained features), v3 training manifests plus referenced v2 climatologies, and the environment. Keep retained protocol/config/indices/checkpoints and the recorded source unchanged for exact provenance. GPU scheduling is server-specific. No training was run as part of preparing this export.

## Environment and paths

Use `requirements.txt` plus `reproduction/requirements-reproduction.txt`. Install an appropriate PyTorch CUDA build explicitly. `requirements-observed.txt`/`observed_environment.json` capture the environment at export time; they are not asserted to be the original execution lockfile. `setproctitle` is required by Oracle runners (not optional there).

Original scripts deliberately preserve absolute Ceph paths, physical GPU IDs and hash guards. This snapshot is **not a portable one-command clone-and-train release**. On the original server use the recorded `/ceph_data/yeonwoo/MHW_subset` layout and environment. Elsewhere provide the same dataset paths/mount layout or perform an explicit documented path/scheduler adaptation and regenerate new-run manifests without overwriting historical ones. Do not silently edit frozen source then claim its original hashes still verify.

Primary training code: `scripts/oracle96_actual/worker.py` and `scripts/oracle96_auxiliary24/aux_worker.py`. Historical `scripts/oracle96/run_all.py` uses a different virtual-epoch protocol: do not confuse it with the final actual-epoch experiment. Queue controllers have side effects; read protocols before launching.

Input provenance: `reproduction/scientific_inputs.json`, `external_dependencies.csv`, feature manifest and recorded source hashes. `reproduction/check_retained.py` is a read-only file-existence/metadata check; it does not import trainers or initialize GPUs. Use `--root` to inspect a full retained Ceph tree. A GitHub-only copy is expected to report missing large files.

## Storage policy

The files proposed for cleanup are intermediate binary artifacts outside primary final120 results and recorded protected-input dependencies, plus completed-run restart/alternate-selection checkpoints. All120 Validation-best weights remain on Ceph. Deleting them sacrifices reanalysis of those specific pilot predictions/checkpoints; small historical evidence is retained. No cleanup has occurred until explicit approval. GitHub uploading alone is not a backup of external inputs or trained models.

The export preserves source files byte-for-byte. Its top-level README/.gitignore are export-specific; the original README is `reproduction/README_original.md`. Runtime caches, credentials, machine logs, data arrays and model weights are excluded.

## Reviewed minimal retention proposal

See reproduction/cleanup_plan_ko.md for exact candidates. Completed last.pt/post_warmup_best.pt are proposed for deletion, but primary best.pt120/120 and all frozen final predictions remain on Ceph. Exact optimizer/RNG resume and old alternate/pilot checkpoint reanalysis would be sacrificed.
