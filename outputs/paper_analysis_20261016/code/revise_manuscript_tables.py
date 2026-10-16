"""Revise publication tables/narrative from frozen statistics; no training or new inference."""
from pathlib import Path
import datetime
import hashlib
import json
from zoneinfo import ZoneInfo
import pandas as pd

O = Path(__file__).resolve().parents[1]
R = O.parents[1]
PREFIX = 'outputs/paper_final_verification_20261015/'
REPO = 'https://github.com/rohyeonwoo/jeju-mhw-pinn-data-efficiency'
EVIDENCE_COMMIT = '8b4ea55f0d9a5e201dcbe745ab729103f47e8f16'
SCOPE = ('Only the matched NoAdv–Adv comparison isolates inclusion of the horizontal-advection term in this implemented training constraint. '
         'P0 versus either NoAdv or Adv also changes active heads and physical-term definitions: the we head is active only in P0; '
         'the Rres head is inactive in all three. Shortwave penetration, learned entrainment versus an MLD-deepening proxy, '
         'and arithmetic versus thickness-weighted mixed-layer velocity definitions also differ. '
         'Sharing the same network architecture does not imply identical active objectives. '
         'This ablation concerns predictive utility of the implemented constraint, not the independent causal role of advection in the ocean.')
KO = ('수평 이류항 포함 여부의 효과를 분리한 비교는 동일 구현의 NoAdv–Adv뿐이다. '
      'P0와 NoAdv·Adv의 비교는 활성 head 및 물리항 정의가 함께 달라지는 복합 비교다. '
      'P0에서만 we head가 활성화되며, 세 모델 모두 Rres head는 비활성이다. '
      '단파 투과, 학습 연행과 MLD-deepening proxy, 산술 평균과 두께 가중 혼합층 유속 정의도 다르다. '
      '동일한 네트워크 구조가 동일한 활성 목적함수를 뜻하지 않는다. '
      '따라서 P0 비교를 수평 이류항만의 독립 효과로 해석하지 않으며, NoAdv–Adv 결과도 구현된 학습 제약의 예측 효용에 한정한다.')

def write(rel, text):
    (O / rel).write_text(text, encoding='utf-8')

def add_once(rel, text):
    p = O / rel
    old = p.read_text()
    if text not in old:
        write(rel, old.rstrip() + '\n\n' + text + '\n')

def latex(frame, caption, label, **kw):
    return frame.to_latex(index=False, escape=True, float_format=lambda x: f'{x:.5f}', caption=caption.replace('%', r'\%').replace('_', r'\_'), label=label, **kw)

full = pd.read_csv(O / 'statistics/paired_summary.csv')
assert full.shape == (48, 15)
(O / 'tables/tableS_paired_all.csv').write_bytes((O / 'statistics/paired_summary.csv').read_bytes())
supp_caption = ('Supplementary Table S1. All six paired model comparisons across eight ratio/lead conditions (48 rows). '
                'Differences are candidate minus reference; positive RMSE differences favor the reference, whereas positive CSI differences favor the candidate. '
                'Three-seed sample SD and existing conditional 90-day bootstrap RMSE intervals; no multiplicity correction. '
                'P0 comparisons change active heads and physical definitions and are not an isolated advection ablation.')
supp_tex = latex(full, supp_caption, 'tab:tableS_paired_all', longtable=True)
write('tables/tableS_paired_all.tex', '% Wide supplementary table: use pdflscape landscape and a small font.\n' + supp_tex)
main = full[(full.reference == 'NoAdv') & (full.candidate == 'Adv')].sort_values(['ratio', 'lead']).copy()
assert len(main) == 8
seeds = pd.read_csv(O / 'statistics/paired_seed_results.csv')
seeds = seeds[(seeds.reference == 'NoAdv') & (seeds.candidate == 'Adv')]
rows = []
for row in main.itertuples():
    g = seeds[(seeds.ratio == row.ratio) & (seeds.lead == row.lead)]
    assert len(g) == 3
    assert abs(g.delta_rmse.mean() - row.delta_rmse_mean) < 1e-12
    rows.append(dict(ratio=row.ratio, lead=row.lead,
                     delta_rmse_adv_minus_noadv_mean=row.delta_rmse_mean,
                     delta_rmse_sd=row.delta_rmse_sd, rmse_ci_low=row.ci95_low, rmse_ci_high=row.ci95_high,
                     noadv_rmse_wins=int((g.delta_rmse > 0).sum()),
                     delta_csi_noadv_minus_adv_mean=-row.delta_csi_mean))
m = pd.DataFrame(rows)
m.to_csv(O / 'tables/table5_paired.csv', index=False)
compact = pd.DataFrame({'Ratio (%)': m.ratio, 'Lead (d)': m.lead,
    'ΔRMSE (K)': [f'{r.delta_rmse_adv_minus_noadv_mean:.4f} ± {r.delta_rmse_sd:.4f}' for r in m.itertuples()],
    'RMSE 95% CI': [f'[{r.rmse_ci_low:.4f}, {r.rmse_ci_high:.4f}]' for r in m.itertuples()],
    'NoAdv wins': m.noadv_rmse_wins.astype(str) + '/3',
    'ΔCSI': m.delta_csi_noadv_minus_adv_mean})
main_caption = ('Matched NoAdv–Adv ablation across eight ratio/lead conditions. '
                'Positive ΔRMSE=RMSE(Adv)−RMSE(NoAdv) and ΔCSI=CSI(NoAdv)−CSI(Adv) favor NoAdv. '
                'RMSE is mean ± sample SD across three seeds; CSI is the seed mean. Wins count seeds with strictly lower NoAdv RMSE. '
                'The 95% intervals reuse a 90-day temporal block bootstrap conditional on these seeds, without multiplicity correction. '
                'Full comparisons appear in Supplementary Table S1; extended NoAdv–Adv CSI intervals appear in the supplementary NoAdv–Adv table.')
write('tables/table5_paired.tex', latex(compact, main_caption, 'tab:table5_paired'))

# Preserve the original local-path audit record separately from publication-facing evidence.
local = O / 'reports/table6_evidence_local.csv'
if not local.exists():
    original = pd.read_csv(O / 'tables/table6_evidence.csv')
    original.to_csv(local, index=False)
items = [
 ('Model and protocol', 'protocol120.csv', 'Inspect all 120 run configurations, epoch counts and selected-checkpoint records.'),
 ('MHW threshold', 'mhw_reference_check.json', 'Inspect the prior bitwise reference comparison and shared evaluation-threshold check.'),
 ('Test metrics and alignment', 'analysis_status.json', 'Inspect the prior 120-run metric recomputation and target/mask/label alignment status.'),
 ('Stencil', 'stencil_tests.csv', 'Inspect the 88 recorded regular-grid numerical tests and their outcomes.'),
 ('Coordinates / support', 'dataset_metadata.json', 'Check ascending coordinates and recorded ocean/support counts of 661/660.'),
 ('Auxiliary diagnostics', 'aux_diagnostics_rechecked.csv', 'Inspect the saved diagnostic-batch outputs and gradients; no optimizer update is performed.'),
 ('Physics units', 'horizontal_advection_audit.md', 'Inspect the K/day conversion, coordinate conventions and recorded upstream limitations.')]
evidence = pd.DataFrame([dict(evidence_id=f'E{i}', item=a, evidence=PREFIX+b, verification_method=c,
    status='Existing evidence reused; audit not rerun') for i, (a,b,c) in enumerate(items, 1)])
assert all((R / p).is_file() for p in evidence.evidence)
evidence.to_csv(O / 'tables/table6_evidence.csv', index=False)
evidence_caption = ('Existing numerical and reproducibility evidence. E1–E7 identify repository files listed in '
                    'table6_evidence.csv and reports/public_evidence_index.md. All entries reuse the prior audit; '
                    'the listed checks were not rerun for this manuscript revision.')
view = evidence[['evidence_id', 'item', 'verification_method']].rename(columns={
    'evidence_id':'ID', 'item':'Evidence', 'verification_method':'Verification method'})
write('tables/table6_evidence.tex', latex(view, evidence_caption, 'tab:table6_evidence', column_format='lp{0.23\\linewidth}p{0.58\\linewidth}'))
write('reports/public_evidence_index.md', '# Public evidence identifiers\n\n'
    f'Repository: {REPO}. Evidence snapshot: `{EVIDENCE_COMMIT}` on the published analysis branch (PR #1). '
    'Paths are relative to the repository root. These links identify existing audit records, not a new verification run.\n\n'
    + '\n'.join(f'- **{r.evidence_id} — {r.item}**: [`{r.evidence}`]({REPO}/blob/{EVIDENCE_COMMIT}/{r.evidence}). {r.verification_method}' for r in evidence.itertuples())
    + '\n\nThe original server paths are retained only as a local reproducibility record in `table6_evidence_local.csv`.\n')

# Keep all interpretation surfaces consistent, without changing model/training code.
for rel in ['reports/limitations.md', 'captions/figure3_advection_effect.md']:
    add_once(rel, SCOPE)
allcap = O / 'captions/all_captions.md'
oldcap = allcap.read_text()
if SCOPE not in oldcap:
    original = (O / 'captions/figure3_advection_effect.md').read_text().split('\n\n')[0]
    assert original in oldcap
    write('captions/all_captions.md', oldcap.replace(original, original + '\n\n' + SCOPE, 1))
t2 = O / 'tables/table2_models.tex'
t2text = t2.read_text()
old = 'P0 and Adv differ in more than advection.'
new = ('P0 versus NoAdv or Adv changes active heads and physical-term definitions: only P0 activates the we head; '
       'all three deactivate Rres. Only NoAdv–Adv isolates horizontal-advection inclusion in the implemented constraint.')
write('tables/table2_models.tex', t2text.replace(old, new))
summary = O / 'reports/analysis_summary_ko.md'
s = summary.read_text()
if KO not in s:
    s = s.replace('## 3. NoAdv와 Adv 차이가 반복되는 조건\n', '## 3. NoAdv와 Adv 차이가 반복되는 조건\n\n' + KO + '\n')
s = s.replace('CSI까지 NoAdv 우위 방향을 양수로 통일한 별도 표는 `tables/tableS_noadv_adv.csv`다.',
              '본문 표 `tables/table5_paired.tex`는 NoAdv–Adv 8개 조건을 6개 열로 제시한다. 전체 48개 대응 비교는 Supplementary Table S1 (`tables/tableS_paired_all.tex`)에 있으며, CSI 신뢰구간을 포함한 확장표는 `tables/tableS_noadv_adv.csv`다.')
write('reports/analysis_summary_ko.md', s)
claims = pd.read_csv(O / 'reports/claim_evidence_matrix.csv')
claims['evidence'] = claims.evidence.str.replace('tables/table5_paired.csv; figures/main/figure2_sst_performance.png',
                                              'tables/tableS_paired_all.csv; figures/main/figure2_sst_performance.png', regex=False)
claimtext = 'Only NoAdv–Adv isolates inclusion of horizontal advection in the implemented training constraint.'
claims = claims[claims.claim != claimtext]
claims = pd.concat([claims, pd.DataFrame([dict(category='Comparison scope', claim=claimtext,
    evidence='tables/table2_models.csv; tables/table5_paired.csv; ../paper_final_verification_20261015/model_definition_audit.md',
    limitation=SCOPE)])], ignore_index=True)
claims.to_csv(O / 'reports/claim_evidence_matrix.csv', index=False)
def mdcell(x):
    return str(x).replace('|', '\\|').replace('\n', ' ')
write('reports/claim_evidence_matrix.md', '# Claim–evidence matrix\n\n| ' + ' | '.join(claims.columns) + ' |\n| '
      + ' | '.join(['---'] * len(claims.columns)) + ' |\n'
      + '\n'.join('| ' + ' | '.join(map(mdcell, row)) + ' |' for row in claims.itertuples(index=False, name=None)) + '\n')
add_once('README.md', 'Manuscript revision: Table 5 contains only the eight NoAdv–Adv conditions (six display columns). '
         'Supplementary Table S1 (`tables/tableS_paired_all.tex/.csv`) retains all 48 comparisons and 15 original columns. '
         'Table 6 uses repository-relative evidence identifiers and verification methods; see `reports/public_evidence_index.md` for pinned public links. '
         'P0 comparisons differ in active heads and physical definitions; only NoAdv–Adv isolates the implemented advection toggle. '
         'Run `python -B code/revise_manuscript_tables.py` to reproduce this revision from the saved summary statistics; '
         'it also runs automatically at the end of `analyze.py` and `finalize_findings.py`.')
add_once('reports/reproducibility_summary.md', 'Publication layout is generated by `code/revise_manuscript_tables.py` from unchanged '
         '`statistics/paired_summary.csv` and `statistics/paired_seed_results.csv`. '
         'Table 5: eight matched NoAdv–Adv conditions; Supplementary Table S1: all 48 rows and 15 original columns. '
         'Public evidence identifiers and pinned links are in `reports/public_evidence_index.md`; original local paths are retained in '
         '`reports/table6_evidence_local.csv`. ' + SCOPE)

manifest = pd.read_csv(O / 'figure_table_manifest.csv')
for name, caption, model_names in [('table5_paired', main_caption, 'NoAdv;Adv'),
                                  ('table6_evidence', evidence_caption, ''),
                                  ('tableS_paired_all', supp_caption, 'Baseline;Auxiliary;P0;NoAdv;Adv')]:
    template = manifest[manifest.id == ('table5_paired' if name == 'tableS_paired_all' else name)].iloc[0].to_dict()
    template.update(id=name, artifact=f'tables/{name}.tex', plot_data=f'tables/{name}.csv', caption=caption,
                    models=model_names, analysis_code='code/analyze.py;code/revise_manuscript_tables.py')
    manifest = manifest[manifest.id != name]
    manifest = pd.concat([manifest, pd.DataFrame([template])], ignore_index=True)
manifest.loc[manifest.id == 'table2_models', 'caption'] = manifest.loc[manifest.id == 'table2_models', 'caption'].str.replace(old, new, regex=False)
manifest.loc[manifest.id == 'figure3_advection_effect', 'caption'] = (O / 'captions/figure3_advection_effect.md').read_text().strip()
manifest.to_csv(O / 'figure_table_manifest.csv', index=False)

# Integrity checks use frozen source tables, not re-estimation of the experiments.
pd.testing.assert_frame_equal(pd.read_csv(O/'tables/tableS_paired_all.csv'), full, check_exact=False, rtol=1e-14, atol=1e-15)
assert int(m.noadv_rmse_wins.sum()) == 22
assert int(((m.rmse_ci_low > 0) | (m.rmse_ci_high < 0)).sum()) == 6
assert '/ceph_data/' not in (O/'tables/table6_evidence.csv').read_text()
assert '/home/' not in (O/'tables/table6_evidence.tex').read_text()
assert compact.shape == (8, 6)
checkpath = O/'reports/output_validation.json'
check = json.loads(checkpath.read_text())
check.update(tables=int((manifest.kind == 'table').sum()), manuscript_main_paired_rows=8,
             manuscript_main_paired_display_columns=6, supplementary_paired_shape=[48,15],
             public_evidence_entries=7, publication_evidence_absolute_paths=False,
             manuscript_revision_numerics_checked=True, latex_compiled=False)
write('reports/output_validation.json', json.dumps(check, indent=2) + '\n')
invpath = O/'reports/source_inventory.json'
inv = json.loads(invpath.read_text())
for name in ['analyze.py', 'finalize_findings.py', 'revise_manuscript_tables.py']:
    p = O/'code'/name
    inv[str(p)] = dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
write('reports/source_inventory.json', json.dumps(inv, indent=2) + '\n')
now = datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
with (R/'docs/research_log.md').open('a') as f:
    f.write(f'\n- {now} — 논문 표·비교 해석 수정: outputs/paper_analysis_20261016의 tables, reports, captions, code 및 manifest 갱신. '
            'Table5는 NoAdv–Adv 8행·6열로 축약, 전체48행·15열은 tableS_paired_all로 보존. Table6은 저장소 상대 식별자·검증 방법으로 전환하고 기존 절대경로는 로컬 감사용 CSV에 보존. '
            'P0의 활성 we head 및 물리항 정의 차이, NoAdv–Adv만의 이류항 ablation 범위를 명시. 재생성 코드에 후처리 연결. '
            '검증: 전체48행 통계 보존, NoAdv RMSE22/24 및 CI6/8 일치, 본문 근거표 절대경로 없음. LaTeX 컴파일러 부재로 조판 미검증. 학습·추론·통계 재추정·서버 정리 없음.\n')
print(json.dumps({'main_table_shape': list(compact.shape), 'supplementary_shape': list(full.shape), 'evidence_entries': len(evidence), 'NoAdv_RMSE_wins': int(m.noadv_rmse_wins.sum()), 'latex_compiled': False}))
