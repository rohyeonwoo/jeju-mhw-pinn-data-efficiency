"""CPU-only rendering of existing curve CSVs; no analysis, training or inference."""
import os
from pathlib import Path
BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']=str(OUT/'cache/matplotlib')
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import hashlib,json,textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs={p:sha(p) for p in (OUT/'analysis_data').glob('*.csv')}
plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320,'axes.spines.top':False,'axes.spines.right':False})
sums=pd.read_csv(OUT/'analysis_data/sst_and_mhw_seed_summary.csv')
ecdfs=pd.read_csv(OUT/'analysis_data/paired_error_ecdf.csv')
SEEDS=[7,42,123];FRACTIONS=[10,25,50,100]
colors={'baseline':'#2864A0','PINN':'#C44B32'}
labels={'baseline':'Baseline','PINN':'PINN (P0 + auxiliary)'}
rendered=[]
def decorate(ax):
 ax.set_xticks(FRACTIONS);ax.set_xlabel('Training block fraction (%)');ax.grid(axis='y',alpha=.18)
def savefig(fig,name,caption,source,plotdata):
 expected=pd.read_csv(OUT/'analysis_data'/(name+'_plot_data.csv'))
 assert len(expected)==len(plotdata)
 for c in plotdata.select_dtypes(include=[np.number]).columns:
  assert np.allclose(expected[c],plotdata[c],equal_nan=True,rtol=1e-9,atol=1e-10),(name,c)
 assert len(fig.axes)==2
 top,bottom=fig.axes
 assert top.get_position().y0>bottom.get_position().y0
 assert np.isclose(top.get_position().x0,bottom.get_position().x0)
 for ext in ['pdf','png']:fig.savefig(OUT/'paper_figures'/(name+'.'+ext),dpi=320,bbox_inches='tight')
 plt.close(fig);rendered.append(name)
source=(BASE/'analyze_data_scarcity_paper.py').read_text()
blocks=[source[source.index('    for metric,label in '):source.index('    d=boot[')],source[source.index('    e=pd.DataFrame(ecdfs);'):source.index("    write_json(figures/'figure_metadata.json'")]]
for block in blocks:exec(compile(textwrap.dedent(block),'<existing plotting code>','exec'))
assert len(rendered)==4 and all(sha(p)==h for p,h in inputs.items())
p=OUT/'paper_figures/figure_metadata.json';metadata=json.loads(p.read_text())
for item in metadata:
 if item['figure'] in rendered:item['panel_layout']='2 rows x 1 column; lead7 above lead14';item['figure_size_inches']=[4.8,6.4]
p.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
p=OUT/'analysis_metadata.json';metadata=json.loads(p.read_text());metadata['script_sha256']=sha(BASE/'analyze_data_scarcity_paper.py');metadata['curve_layout_update']='2026-10-01 Asia/Seoul; existing 1x2 curve panels changed to 2x1 using stored CSV only';p.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
(OUT/'vertical_layout_verification.json').write_text(json.dumps({'figures':rendered,'layout':'2x1','csv_files_unchanged':len(inputs),'plot_values_match_saved_csv':True,'cpu_only':True,'training':False},indent=2)+'\n')
print('Updated PDF + PNG:',', '.join(rendered))
print('All analysis CSVs unchanged:',len(inputs))
