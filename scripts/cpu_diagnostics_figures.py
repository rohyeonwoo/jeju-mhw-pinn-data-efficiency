from cpu_diagnostics_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 d=OUT/'figures';d.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
 def export(fig,name):fig.tight_layout();fig.savefig(d/(name+'.png'),dpi=180);fig.savefig(d/(name+'.pdf'));plt.close(fig)
 c=readcsv(OUT/'cumulative_summary.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,lead in zip(axs,[7,14]):
  for split in ['train','val','test']:
   rr=[next(r for r in c if int(r['ratio'])==ratio and int(r['lead'])==lead and r['split']==split) for ratio in RATIOS];ax.plot(RATIOS,[float(r['rmse_skill_vs_zero']) for r in rr],marker='o',label=split)
  ax.axhline(0,color='black',ls='--');ax.set(title=f'Lead {lead}: native observed linear path',xlabel='Subset (%)',ylabel='Cumulative skill vs zero change',xticks=RATIOS);ax.legend()
 export(fig,'cumulative_physics_skill')
 c=readcsv(OUT/'coverage_summary.csv');a=readcsv(OUT/'anomaly_summary.csv');fig,axs=plt.subplots(1,2,figsize=(10,4));x=np.arange(4)
 empty=np.array([float(next(r for r in c if int(r['ratio'])==ratio)['empty_fraction']) for ratio in RATIOS]);single=np.array([float(next(r for r in c if int(r['ratio'])==ratio)['singleton_fraction']) for ratio in RATIOS]);axs[0].bar(x,empty,label='0 observations');axs[0].bar(x,single,bottom=empty,label='1 observation');axs[0].bar(x,1-empty-single,bottom=empty+single,label='2+ observations');axs[0].set(xticks=x,xticklabels=RATIOS,xlabel='Subset (%)',ylabel='Fraction of 366 calendar bins');axs[0].legend()
 for split in ['train','val','test']:
  y=[float(next(r for r in a if int(r['ratio'])==ratio and r['lead']=='7' and r['split']==split and r['variable']=='sst_anom')['history_z_std']) for ratio in RATIOS];axs[1].plot(RATIOS,y,marker='o',label=split)
 axs[1].set(xticks=RATIOS,xlabel='Subset (%)',ylabel='Normalized SST anomaly std',title='Lead7 history-weighted inputs');axs[1].legend();export(fig,'climatology_coverage_shift')
 g=readcsv(OUT/'adaptive_summary.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,ratio,lead in zip(axs,[50,10],[14,7]):
  for mode in ['instantaneous','ordered_frozen_batch_EMA']:
   rr=[next(r for r in g if int(r['ratio'])==ratio and int(r['lead'])==lead and r['state']=='best' and r['mode']==mode and float(r['rho'])==rho) for rho in [.05,.1,.25]];ax.plot([.05,.1,.25],[float(r['lambda_median']) for r in rr],marker='o',label=mode)
  ax.axhline(.01,ls='--',color='black',label='Fixed .01');ax.axhline(.03,ls=':',color='grey');ax.set(xlabel='Target ratio rho',ylabel='Mean of seed median base lambda',title=f'{ratio}% / Lead{lead}: offline only');ax.legend(fontsize=7)
 export(fig,'offline_lambda_candidates');finish('figures',figures=3)
if __name__=='__main__':main()
