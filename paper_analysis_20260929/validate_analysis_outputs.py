#!/usr/bin/env python3
"""CPU-only independent numerical/output validation. No training or inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,importlib.util,hashlib
import numpy as np
import pandas as pd
from PIL import Image
base=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else base/'results'
data=out/'analysis_data';checks={}
def load(name):return pd.read_csv(data/(name+'.csv'))
spec=importlib.util.spec_from_file_location('paper_analysis',base/'analyze_data_scarcity_paper.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=np.arange(1,1+2*2*3*31,dtype=float).reshape(2,2,3,31)
for block in [1,7,90]:
 actual=m.bootstrap_daily(a,301,block,123)
 starts=np.random.default_rng(123).integers(0,31,size=(301,int(np.ceil(31/block))))
 ix=((starts[:,:,None]+np.arange(block))%31).reshape(301,-1)[:,:31]
 expected=np.array([[[np.sqrt(a[r,model,seed][ix].mean(axis=1)) for seed in range(3)] for model in range(2)] for r in range(2)]).mean(axis=2).transpose(2,0,1)
 target=np.stack([expected[:,:,1]-expected[:,:,0],(expected[:,:,0]-expected[:,:,1])/expected[:,:,0]*100],axis=-1)
 assert np.allclose(actual,target)
assert np.allclose(m.bootstrap_daily(np.ones((1,2,3,31)),301,7,123),0)
checks['bootstrap_matches_existing_reference_and_zero_effect_test']=True
long=load('experiment_metrics_long');primary=load('primary_metrics_long');pair=load('paired_model_differences');summary=load('sst_summary')
assert len(long)==108 and len(primary)==48 and len(pair)==30 and len(summary)==10
assert not primary.duplicated(['training_fraction','lead_day','model','seed']).any()
for r in pair.itertuples():
 d=long[(long.experiment_family=='batch16_original')&(long.data_regime==r.data_regime)&(long.lead_day==r.lead_day)&(long.seed==r.seed)]
 if r.data_regime=='block_subset':d=d[d.training_fraction==r.training_fraction]
 b=d[d.model=='baseline'].iloc[0];p=d[d.model=='PINN'].iloc[0]
 assert np.isclose(p.rmse-b.rmse,r.delta_rmse,atol=1e-10)
 assert np.isclose((b.rmse-p.rmse)/b.rmse*100,r.relative_rmse_improvement_pct,atol=1e-8)
 for metric in ['mhw_csi','mhw_pod','mhw_far','strong_csi']:
  assert np.isclose(p[metric]-b[metric],getattr(r,'delta_'+metric),atol=1e-10)
checks['all_pairs_and_signs_independently_verified']=True
for r in summary.itertuples():
 d=pair[(pair.data_regime==r.data_regime)&(pair.lead_day==r.lead_day)]
 if r.data_regime=='block_subset':d=d[d.training_fraction==r.training_fraction]
 assert np.isclose(d.baseline_rmse.mean(),r.baseline_rmse_mean)
 assert np.isclose(d.baseline_rmse.std(ddof=1),r.baseline_rmse_std)
 assert np.isclose(d.pinn_rmse.std(ddof=1),r.pinn_rmse_std)
 assert int((d.delta_rmse<0).sum())==r.n_pinn_better
 assert np.isclose((r.baseline_rmse_mean-r.pinn_rmse_mean)/r.baseline_rmse_mean*100,r.relative_improvement_pct,atol=1e-8)
report=(out/'analysis_report.md').read_text()
assert m.mdtable(summary) in report
assert m.mdtable(load('mhw_metrics_summary')) in report
assert m.mdtable(load('scarcity_trend')) in report
checks['report_tables_equal_csv_values']=True
trends=load('scarcity_trend').set_index('lead_day')
assert not bool(trends.loc[7,'benefit_monotonically_increases_as_fraction_decreases'])
assert bool(trends.loc[14,'benefit_monotonically_increases_as_fraction_decreases'])
assert 'lead7 is non-monotonic and lead14 is monotonic' in report
assert 'Neither lead is monotonic' not in report
checks['lead_specific_trend_text_verified']=True
meta=json.loads((out/'paper_figures/figure_metadata.json').read_text())
assert len(meta)==15
for item in meta:
 p=out/'paper_figures'/item['figure']
 with Image.open(p.with_suffix('.png')) as im:
  assert min(im.info['dpi'])>=300;assert min(im.size)>800
 assert p.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
 plot=pd.read_csv(data/item['plot_data_csv']); source=pd.read_csv(data/item['source_csv'])
 common=[c for c in plot.columns if c in source.columns and pd.api.types.is_numeric_dtype(plot[c])]
 # Each plot data record must correspond to an actual source row on all shared numeric columns.
 for _,r in plot.iterrows():
  match=np.ones(len(source),dtype=bool)
  for c in common:match &= np.isclose(pd.to_numeric(source[c],errors='coerce'),r[c],atol=1e-9,rtol=1e-9,equal_nan=True)
  assert match.any(),(item['figure'],r.to_dict())
checks['15_figures_pdf_png_300dpi_and_plot_source_numeric_consistency']=True
sst_tex=(out/'paper_tables/table_sst.tex').read_text()
for r in summary[summary.data_regime=='block_subset'].itertuples():
 assert f'{r.baseline_rmse_mean:.4f}' in sst_tex and f'{r.pinn_rmse_mean:.4f}' in sst_tex and f'{r.relative_improvement_pct:+.2f}' in sst_tex
for name in ['table_setup','table_sst','table_mhw']:
 t=(out/'paper_tables'/(name+'.tex')).read_text();assert '\\begin{table*}' in t and '\\end{table*}' in t
checks['latex_values_and_two_column_environments_verified']=True
files=load('file_inventory')
for r in files.itertuples():
 s=Path(r.path).stat()
 assert s.st_size==r.bytes
 assert abs(pd.Timestamp(r.modified_utc).value-s.st_mtime_ns)<=1000,(r.path,r.modified_utc,s.st_mtime_ns)
checks['all_discovered_original_result_files_size_mtime_unchanged']=True
checks['inventory_mtime_precision']='ISO microseconds; <=1 microsecond conversion tolerance; used-source contents additionally SHA-256 verified'
prov=load('source_provenance');assert prov.unchanged.all() and (prov.sha256_before==prov.sha256_after).all()
checks['all_used_source_hashes_unchanged']=True
root=Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')
defs=[]
for folder in [root/'allocation_batch16_5plus5'/f'ratio_{r}' for r in [10,25,50,100]]+[root/'full_batch16_matched/full']:
 d=json.loads((folder/'mhw_skill.json').read_text())['config'];defs.append({k:d[k] for k in ['threshold','mhw_counting','strong_plus_definition','event_definition']})
assert all(x==defs[0] for x in defs)
checks['mhw_definitions_identical_across_fractions_and_full']=True
assert 'torch' not in sys.modules; checks['torch_not_imported_cuda_hidden']=os.environ['CUDA_VISIBLE_DEVICES']==''
checks['status']='passed';checks['input_source_count']=len(prov);checks['original_discovered_files']=len(files)
(out/'independent_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
