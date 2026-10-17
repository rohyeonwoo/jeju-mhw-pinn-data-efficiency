from cpu_diagnostics_common import *
def main():
 rows=readcsv(ADAPT/'gradient_diagnostics.csv');out=[];summary=[]
 for job in sorted({r['job'] for r in rows},key=int):
  for state in ['best','last']:
   rr=[r for r in rows if r['job']==job and r['state']==state];assert len(rr)==32
   for rho in [.05,.10,.25]:
    ed=ep=None;group=[]
    for i,r in enumerate(rr):
     gd=float(r['G_data']);gp=float(r['G_phys']);warm=float(r['checkpoint_warmup_factor']);ed=gd if ed is None else .9*ed+.1*gd;ep=gp if ep is None else .9*ep+.1*gp
     for mode,target in [('instantaneous',rho*gd/(gp+1e-12)),('ordered_frozen_batch_EMA',rho*ed/(ep+1e-12))]:
      lam=float(np.clip(target,.0001,.03));row=dict(job=int(job),ratio=int(r['ratio']),lead=int(r['lead']),seed=int(r['seed']),state=state,checkpoint_epoch=int(r['checkpoint_epoch']),rho=rho,mode=mode,batch=i,G_data=gd,G_phys=gp,cosine=float(r['cosine']),lambda_target=target,lambda_base=lam,lambda_eff=lam*warm,fixed_lambda_eff=.01*warm,base_ratio=lam*gp/(gd+1e-12),effective_ratio=lam*warm*gp/(gd+1e-12),clamp_low=target<.0001,clamp_high=target>.03,greater_than_fixed=lam>.01)
      out.append(row);group.append(row)
    for mode in ['instantaneous','ordered_frozen_batch_EMA']:
     g=[r for r in group if r['mode']==mode];v=np.array([r['lambda_base'] for r in g]);summary.append(dict(job=int(job),ratio=int(rr[0]['ratio']),lead=int(rr[0]['lead']),seed=int(rr[0]['seed']),state=state,rho=rho,mode=mode,lambda_p10=np.quantile(v,.1),lambda_median=np.median(v),lambda_p90=np.quantile(v,.9),upper_clamp_fraction=np.mean([r['clamp_high'] for r in g]),lower_clamp_fraction=np.mean([r['clamp_low'] for r in g]),greater_than_fixed_fraction=np.mean(v>.01),effective_ratio_median=np.median([r['effective_ratio'] for r in g]),conflict_fraction=np.mean([r['cosine']<0 for r in g])))
 csvsave('adaptive_offline_batches.csv',out);csvsave('adaptive_offline_summary.csv',summary)
 assert all(.0001<=r['lambda_base']<=.03 and r['lambda_eff']<=r['lambda_base'] for r in out)
 finish('adaptive_offline',frozen_gradient_batches=len(rows),rho_candidates=[.05,.10,.25],limitation='Ordered frozen-batch EMA is an offline sensitivity illustration, not reconstruction of training dynamics or 100-update temporal spacing. No rho selected from Test; paused pilot unchanged.')
if __name__=='__main__':main()
