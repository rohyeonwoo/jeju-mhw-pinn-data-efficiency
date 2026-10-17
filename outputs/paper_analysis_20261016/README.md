# Oracle120 final paper analysis

Start with [Korean analysis summary](reports/analysis_summary_ko.md), [claim–evidence matrix](reports/claim_evidence_matrix.md), [limitations](reports/limitations.md), and [reproducibility](reports/reproducibility_summary.md).

- [Artifact manifest](figure_table_manifest.csv): 5 main and 6 supplementary figures; Table1–6 plus separate Validation and explicit NoAdv-positive comparison tables.
- [English captions](captions/all_captions.md); figures in PNG (350dpi), PDF and SVG.
- `tables/`: CSV and LaTeX; detailed Test/Validation seed metrics and regional summaries.
- `statistics/`: six paired comparisons; original 90-day bootstrap; physical/seasonal summaries.
- `plot_data/`: grid errors, physical histograms/maps, regional scores, neutral calendar case.
- `reports/output_validation.json`: numerical and format checks.

LaTeX tables use booktabs/longtable; UTF-8 symbols are supported with XeLaTeX/LuaLaTeX. Tables with many columns may be placed in a landscape supplement or reformatted by the target journal. The CSV files retain full precision.

Full analysis: `code/analyze.py`; final S1/S5 display refinement can be repeated with `code/refine_figures.py`. Both write only within this new analysis directory, except the required append to the shared research log. No training, inference, new bootstrap or prior audit rerun.

Final numerical interpretation and explicit NoAdv-positive CSI confidence intervals: `code/finalize_findings.py` (run after the two scripts above).

Manuscript revision: Table 5 contains only the eight NoAdv–Adv conditions (six display columns). Supplementary Table S1 (`tables/tableS_paired_all.tex/.csv`) retains all 48 comparisons and 15 original columns. Table 6 uses repository-relative evidence identifiers and verification methods; see `reports/public_evidence_index.md` for pinned public links. P0 comparisons differ in active heads and physical definitions; only NoAdv–Adv isolates the implemented advection toggle. Run `python -B code/revise_manuscript_tables.py` to reproduce this revision from the saved summary statistics; it also runs automatically at the end of `analyze.py` and `finalize_findings.py`.
