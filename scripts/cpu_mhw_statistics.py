from cpu_diagnostics_common import *
from mhw.mhw_skill import day_of_year,mhw_binary_field
import itertools

def weights(n,lead,block):
 rng=np.random.default_rng(20260925+lead+block);starts=rng.integers(0,n,(4000,int(np.ceil(n/block))));ix=((starts[:,:,None]+np.arange(block))%n).reshape(4000,-1)[:,:n]
 w=np.empty((4000,n),dtype=np.float64)
 for i,row in enumerate(ix):w[i]=np.bincount(row,minlength=n)
 assert np.all(w.sum(1)==n);return w

def null_p(rep,estimate):
 # Center bootstrap distribution at its observed statistic to approximate H0.
 return float((1+np.count_nonzero(abs(rep-estimate)>=abs(estimate)))/(len(rep)+1))
def holm(rows,pkey,key):
 order=sorted(range(len(rows)),key=lambda i:rows[i][pkey]);prev=0
 for rank,i in enumerate(order):prev=max(prev,min(1.,(len(rows)-rank)*rows[i][pkey]));rows[i][key]=prev

def metrics_counts(c):
 h,m,f,cn=np.moveaxis(c,-1,0)
 return {k:np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den>0) for k,num,den in [('csi',h,h+m+f),('pod',h,h+m),('far',f,h+f)]}

def regression_checks():
 jb,j=pair_jobs()[0];b=fields(jb,'test',['daily_sse','daily_count']);p=fields(j,'test',['daily_sse','daily_count'])
 w=weights(len(b['daily_sse']),j['lead'],90);delta=np.sqrt((w@p['daily_sse'])/(w@p['daily_count']))-np.sqrt((w@b['daily_sse'])/(w@b['daily_count']))
 with np.load(OLD/'bootstrap_replicates.npz') as z:expected=z[f"r{j['ratio']}_l{j['lead']}_seed{j['seed']}_b90_delta"]
 np.testing.assert_allclose(delta,expected,atol=1e-12,rtol=1e-12)
 c=np.array([4.,2.,1.,8.]);m=metrics_counts(c);assert m['csi']==4/7 and m['pod']==4/6 and m['far']==1/5
 save(OUT/'statistics_regression.json',dict(status='PASS',replicate_count=4000,paired_weight_matrix_matches_saved_index_bootstrap=True,max_abs_delta_difference=float(abs(delta-expected).max()),known_contingency_counts=True))

def main():
 regression_checks()
 ref=json.loads((OLD/'mhw_protocol.json').read_text())
 with np.load(ref['reference']) as z:threshold=z['threshold'];clim=z['climatology'];sea=z['mask']>0
 sst_stats=[];transitions=[];distance=[];daily={};truth={};original={};seed_pairs=readcsv(OLD/'seed_improvements.csv')
 for jb,j in pair_jobs():
  fp=fields(j,'test',['pred_raw','target_raw','target_index','target_dates']);fb=fields(jb,'test',['pred_raw','target_raw','target_index']);assert np.array_equal(fp['target_index'],fb['target_index']);assert np.array_equal(fp['target_raw'],fb['target_raw'])
  idx=fp['target_index'];dates=fp['target_dates'];doy=day_of_year(dates);obs=fp['target_raw'][:,0];pred=fp['pred_raw'][:,0];base=fb['pred_raw'][:,0];th=threshold[doy];cl=clim[doy]
  ob=mhw_binary_field(obs,doy,threshold,5)[:,sea];bp=mhw_binary_field(base,doy,threshold,5)[:,sea];pp=mhw_binary_field(pred,doy,threshold,5)[:,sea]
  for jj,pb in [(jb,bp),(j,pp)]:
   with np.load(Path(jj['evaluation_dir'])/'mhw_fields.npz') as z:
    np.testing.assert_array_equal(pb,z['pred_mhw'][:,sea]);np.testing.assert_array_equal(ob,z['obs_mhw'][:,sea]);np.testing.assert_array_equal(idx,z['target_index'])
  if j['lead'] in truth:np.testing.assert_array_equal(truth[j['lead']],idx)
  else:truth[j['lead']]=idx
  oo=obs[:,sea];bb=base[:,sea];ppv=pred[:,sea];tt=th[:,sea];cc=cl[:,sea];dd=oo-tt;intensity=np.divide(oo-cc,tt-cc,out=np.full_like(oo,np.nan),where=(tt-cc)>0)
  groups={'all':np.ones_like(ob),'MHW':ob,'non_MHW':~ob,'near_threshold_abs025K':abs(dd)<=.25,'near_above_0to025K':(dd>0)&(dd<=.25),'near_below_m025to0K':(dd>=-.25)&(dd<=0),'high_intensity_obs_MHW_ratio_ge2':ob&(intensity>=2)}
  for group,sel in groups.items():
   for arm,arr in [('baseline',bb),('physics_only',ppv)]:
    error=arr[sel].astype(float)-oo[sel];delta=arr[sel]-tt[sel]
    sst_stats.append(dict(**meta(j),group=group,model=arm,n=int(sel.sum()),rmse=np.sqrt(np.mean(error**2)) if error.size else None,bias=error.mean() if error.size else None,threshold_distance_mean=delta.mean() if delta.size else None,pred_below_threshold_fraction=np.mean(delta<=0) if delta.size else None))
  category={'both_TP':ob&bp&pp,'new_FN':ob&bp&~pp,'recovered_FN':ob&~bp&pp,'both_FN':ob&~bp&~pp,'new_FP':~ob&~bp&pp,'removed_FP':~ob&bp&~pp,'both_FP':~ob&bp&pp,'both_TN':~ob&~bp&~pp}
  assert sum(int(x.sum()) for x in category.values())==ob.size
  for label,sel in category.items():
   transitions.append(dict(**meta(j),event_transition=label,n=int(sel.sum()),near_threshold_abs025K_fraction=np.mean(abs(dd[sel])<=.25) if sel.any() else None,pinn_minus_baseline_temperature_K=np.mean((ppv-bb)[sel]) if sel.any() else None,baseline_bias=np.mean((bb-oo)[sel]) if sel.any() else None,physics_bias=np.mean((ppv-oo)[sel]) if sel.any() else None,observed_threshold_distance=np.mean(dd[sel]) if sel.any() else None))
  bounds=[-np.inf,-1,-.5,-.25,0,.25,.5,1,np.inf]
  for lo,hi in zip(bounds[:-1],bounds[1:]):
   sel=(dd>lo)&(dd<=hi)
   if not sel.any():continue
   distance.append(dict(**meta(j),obs_distance_bin=f'({lo},{hi}] K',n=int(sel.sum()),baseline_exceedance_rate=np.mean(bb[sel]>tt[sel]),physics_exceedance_rate=np.mean(ppv[sel]>tt[sel]),baseline_event_rate=np.mean(bp[sel]),physics_event_rate=np.mean(pp[sel]),baseline_bias=np.mean((bb-oo)[sel]),physics_bias=np.mean((ppv-oo)[sel]),scope='single-threshold exceedance separated from native5-day event'))
  for arm,pb,arr in [('baseline',bp,bb),('physics_only',pp,ppv)]:
   counts=np.stack([(pb&ob).sum(1),(~pb&ob).sum(1),(pb&~ob).sum(1),(~pb&~ob).sum(1)],1).astype(float);daily[j['ratio'],j['lead'],j['seed'],arm]=counts
   original[j['ratio'],j['lead'],j['seed'],arm]=metrics_counts(counts.sum(0))
  print('MHW',meta(j),flush=True)
 for name,rs in [('mhw_group_errors.csv',sst_stats),('mhw_transitions.csv',transitions),('mhw_threshold_bins.csv',distance)]:csvsave(name,rs)
 np.savez_compressed(OUT/'mhw_daily_counts.npz',**{f'r{r}_l{l}_s{s}_{a}':v for (r,l,s,a),v in daily.items()})
 boots=[];sst=[];contrasts=[];reps={}
 with np.load(OLD/'bootstrap_replicates.npz') as old:
  for block in [30,90,180]:
   family=[]
   for lead in [7,14]:
    w=weights(len(truth[lead]),lead,block)
    for ratio in RATIOS:
     # SST p-values from saved raw bootstrap replicates, never from CI endpoints.
     estimate=np.mean([float(r['delta_rmse']) for r in seed_pairs if int(r['ratio'])==ratio and int(r['lead'])==lead]);rep=old[f'r{ratio}_l{lead}_seedfixed3_mean_b{block}_delta'];lo,hi=np.quantile(rep,[.025,.975]);family.append(dict(ratio=ratio,lead=lead,block_days=block,delta_rmse=estimate,ci_low=lo,ci_high=hi,p_centered_bootstrap=null_p(rep,estimate),replicates=4000))
     pairs={metric:[] for metric in ['csi','pod','far']};est={metric:[] for metric in pairs}
     for seed in SEEDS:
      bm=metrics_counts(w@daily[ratio,lead,seed,'baseline']);pm=metrics_counts(w@daily[ratio,lead,seed,'physics_only'])
      for metric in pairs:pairs[metric].append(pm[metric]-bm[metric]);est[metric].append(float(original[ratio,lead,seed,'physics_only'][metric]-original[ratio,lead,seed,'baseline'][metric]))
     for metric in pairs:
      sample=np.mean(pairs[metric],0);point=np.mean(est[metric]);lo,hi=np.quantile(sample,[.025,.975]);boots.append(dict(ratio=ratio,lead=lead,metric=metric,block_days=block,delta=point,ci_low=lo,ci_high=hi,p_centered_bootstrap=null_p(sample,point),seed_delta_sd=np.std(est[metric],ddof=1),replicates=4000));reps[f'r{ratio}_l{lead}_b{block}_{metric}']=sample
    for low,high in itertools.combinations(RATIOS,2):
     sample=old[f'r{low}_l{lead}_seedfixed3_mean_b{block}_improvement']-old[f'r{high}_l{lead}_seedfixed3_mean_b{block}_improvement'];point=np.mean([float(r['improvement_percent']) for r in seed_pairs if int(r['ratio'])==low and int(r['lead'])==lead])-np.mean([float(r['improvement_percent']) for r in seed_pairs if int(r['ratio'])==high and int(r['lead'])==lead]);lo,hi=np.quantile(sample,[.025,.975]);contrasts.append(dict(lower_ratio=low,higher_ratio=high,lead=lead,block_days=block,improvement_difference_pp=point,ci_low=lo,ci_high=hi,p_centered_bootstrap=null_p(sample,point),matching='identical target dates and seeds within lead'))
   holm(family,'p_centered_bootstrap','p_holm_8');sst+=family
   for metric in ['csi','pod','far']:holm([r for r in boots if r['block_days']==block and r['metric']==metric],'p_centered_bootstrap','p_holm_8_per_metric')
   holm([r for r in boots if r['block_days']==block],'p_centered_bootstrap','p_holm_24_metrics_conditions');holm([r for r in contrasts if r['block_days']==block],'p_centered_bootstrap','p_holm_12_contrasts')
   print('BOOTSTRAP',block,flush=True)
 # Exact paired seed-sign randomization: only8 sign assignments; explicitly low power.
 seedstat=[]
 for ratio in RATIOS:
  for lead in [7,14]:
   rr=[r for r in seed_pairs if int(r['ratio'])==ratio and int(r['lead'])==lead];d=np.array([float(r['delta_rmse']) for r in rr]);signs=np.array(list(itertools.product([-1,1],repeat=3)));null=(signs*d).mean(1);seedstat.append(dict(ratio=ratio,lead=lead,improved_seeds=int((d<0).sum()),delta_mean=d.mean(),delta_seed_sd=d.std(ddof=1),exact_seed_sign_p=np.mean(abs(null)>=abs(d.mean())-1e-14),scope='3seeds; exact8 sign flips under symmetry/exchangeability; min two-sided p=.25'))
 for name,rr in [('sst_holm.csv',sst),('mhw_bootstrap.csv',boots),('paired_ratio_contrasts.csv',contrasts),('seed_uncertainty.csv',seedstat)]:csvsave(name,rr)
 np.savez_compressed(OUT/'mhw_bootstrap_replicates.npz',**reps)
 # Sanity checks for multiplicity and null centering.
 a=[{'p':.01},{'p':.04},{'p':.03}];holm(a,'p','adj');assert np.allclose([r['adj'] for r in a],[.03,.06,.06]);assert null_p(np.zeros(4000),0)==1
 finish('mhw_statistics',test_pairs=24,bootstrap_draws=4000,block_lengths=[30,90,180],limitation='Centered block-bootstrap approximate null test, conditional on fixed3 fitted seeds; not exact finite-sample p. Event labels computed on original contiguous Test before block resampling; artificial block seams never create events. Threshold diagnostics exploratory.')
if __name__=='__main__':main()
