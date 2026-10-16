"""CPU-only: preserve reference figure style, change panel layout to 2x1."""
import os
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'results'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']=str(O/'cache/matplotlib')
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,hashlib,textwrap,ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=B/'analyze_data_scarcity_paper.py';s=p.read_text()
start=s.index("    d=summary[summary.data_regime=='block_subset']") if "    # Reference-style SST figures" not in s else s.index('    # Reference-style SST figures')
end=s.index("    for lead in [7,14]:\n        d=pair[",start)
block='''    # Reference-style SST figures: preserve errors, colors, labels; stack panels only.
    with plt.rc_context({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'lines.linewidth':0.9,'lines.markersize':3,'axes.linewidth':0.6,'pdf.fonttype':42,'ps.fonttype':42}):
        d=boot[(boot.block_days==90)&(boot.data_regime=='block_subset')].sort_values(['lead_day','training_fraction'])
        fig,axes=plt.subplots(2,1,figsize=(3.6,4.4),sharey=True,layout='constrained')
        for ax,lead in zip(axes,[7,14]):
            g=d[d.lead_day==lead].sort_values('training_fraction')
            y=g.relative_improvement_pct.to_numpy()
            errors=np.vstack([y-g.relative_ci_low.to_numpy(),g.relative_ci_high.to_numpy()-y])
            ax.errorbar(g.training_fraction,y,yerr=errors,fmt='o-',color='#2864A0',capsize=2.5,elinewidth=0.7,capthick=0.7)
            ax.axhline(0,color='.35',lw=0.7,ls='--')
            ax.set_xticks(FRACTIONS);ax.set_xlabel('Block-allocation label (%)');ax.set_ylabel('Relative RMSE reduction (%)')
            ax.set_title(f'{lead}-day lead');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        savefig(fig,'fig_relative_improvement','Relative reduction in seed-mean SST RMSE across controlled training-window allocations. Error bars denote 95% percentile intervals from the paired 90-day moving-block bootstrap. Positive values favor the physics-informed configuration. Panels stacked: 7-day above 14-day.','bootstrap_ci.csv',d)
        d=sums[sums.data_regime=='block_subset']
        fig,axes=plt.subplots(2,1,figsize=(3.6,4.4),sharey=False,layout='constrained')
        for ax,lead in zip(axes,[7,14]):
            for m,color,label in [('baseline','#2864A0','Baseline'),('PINN','#DD6B35','P0 PINN')]:
                g=d[(d.lead_day==lead)&(d.model==m)].sort_values('training_fraction')
                ax.errorbar(g.training_fraction,g.rmse_mean,yerr=g.rmse_std,fmt='o-',color=color,label=label,capsize=2.5,elinewidth=0.7,capthick=0.7)
            ax.set_xticks(FRACTIONS);ax.set_xlabel('Block-allocation label (%)');ax.set_ylabel('SST RMSE (K)')
            ax.set_title(f'{lead}-day lead');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        axes[0].legend(frameon=False,loc='upper right')
        savefig(fig,'fig_rmse_vs_fraction','Absolute SST RMSE across controlled training-window allocations. Seed mean ± sample SD (n=3), separate y scales per lead as in supplied reference. Baseline versus P0 PINN (original P0 includes auxiliary loss). Panels stacked: 7-day above 14-day.','sst_and_mhw_seed_summary.csv',d)
'''
s=s[:start]+block+s[end:];ast.parse(s);p.write_text(s)
boot=pd.read_csv(O/'analysis_data/bootstrap_ci.csv');sums=pd.read_csv(O/'analysis_data/sst_and_mhw_seed_summary.csv');FRACTIONS=[10,25,50,100]
inputs={q:hashlib.sha256(q.read_bytes()).hexdigest() for q in [O/'analysis_data/bootstrap_ci.csv',O/'analysis_data/sst_summary.csv',O/'analysis_data/sst_and_mhw_seed_summary.csv']}
metadata=json.loads((O/'paper_figures/figure_metadata.json').read_text());checks={}
def savefig(fig,name,caption,source,plotdata):
    assert len(fig.axes)==2 and fig.axes[0].get_position().y0>fig.axes[1].get_position().y0
    for ax,lead in zip(fig.axes,[7,14]):
        g=plotdata[plotdata.lead_day==lead].sort_values('training_fraction')
        assert len(ax.containers)==(1 if name=='fig_relative_improvement' else 2)
        if name=='fig_relative_improvement':
            segments=ax.containers[0].lines[2][0].get_segments()
            assert np.allclose([a[:,1] for a in segments],g[['relative_ci_low','relative_ci_high']])
            assert np.allclose(ax.containers[0].lines[0].get_ydata(),g.relative_improvement_pct)
        else:
            for container,model in zip(ax.containers,['baseline','PINN']):
                r=g[g.model==model];assert np.allclose(container.lines[0].get_ydata(),r.rmse_mean)
                assert np.allclose([a[:,1] for a in container.lines[2][0].get_segments()],np.column_stack([r.rmse_mean-r.rmse_std,r.rmse_mean+r.rmse_std]))
    for ext in ['pdf','png']:fig.savefig(O/'paper_figures'/(name+'.'+ext),dpi=320,bbox_inches='tight')
    plt.close(fig)
    plotdata.to_csv(O/'analysis_data'/(name+'_plot_data.csv'),index=False,na_rep='NaN',float_format='%.12g')
    metadata[:]=[m for m in metadata if m['figure']!=name]
    metadata.append({'figure':name,'caption':caption,'source_csv':source,'plot_data_csv':name+'_plot_data.csv','png_dpi':320,'panel_layout':'2x1; 7-day above 14-day','figure_size_inches':[3.6,4.4]})
    checks[name]='plotted means and all error-bar endpoints match source CSV; 2x1 layout'
exec(compile(textwrap.dedent(block),'<reference-style SST plots>','exec'))
assert all(hashlib.sha256(q.read_bytes()).hexdigest()==h for q,h in inputs.items())
(O/'paper_figures/figure_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
p=O/'analysis_metadata.json';m=json.loads(p.read_text());m['script_sha256']=hashlib.sha256(s.encode()).hexdigest();m['rmse_layout_update']='2026-10-01: reference style preserved; relative 90-day bootstrap 95% CI restored; absolute mean ± SD and independent lead axes; 2x1 panels';p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
checks['source_metrics_and_bootstrap_unchanged']=True
(O/'rmse_layout_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
