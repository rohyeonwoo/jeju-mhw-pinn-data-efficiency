"""CPU-only native v6 midpoint-layer means; no fitted/full-data input statistics."""
from mixed_budget_common import *
import numpy as np
import xarray as xr
import experiment_mixed_layer_budget_cpu as c

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 if (FEATURES/'manifest.json').exists():
  m=json.loads((FEATURES/'manifest.json').read_text());assert m['status']=='complete'
  for p,h in m['sha256'].items():assert sha(p)==h,p
  print('Feature cache verified; reused');return
 FEATURES.mkdir(parents=True,exist_ok=True)
 if list(FEATURES.glob('*.npy')):raise RuntimeError('Incomplete existing feature files; preserve and inspect before retry')
 with xr.open_dataset(BUDGET) as b,xr.open_dataset(GLORYS) as g:
  bd=c.detect_dims(b);gd=c.detect_dims(g,True);z,t=c.check_alignment(b,g,bd,gd);edges=c.vertical_edges(z);ocean=c.read2d(b,'ocean_mask_static',bd)>.5
  assert c.units_mode(b.qnet.attrs,'auto','qnet')=='W_m2' and c.units_mode(b.ssr.attrs,'auto','ssr')=='W_m2'
  shape=(len(t),*ocean.shape);keys=['h','qnet','ssr','u','v','tb','we','tm','valid'];out={k:np.lib.format.open_memmap(FEATURES/(k+'.npy'),mode='w+',dtype='float32',shape=shape) for k in keys}
  hp=None;valid_count=0
  for start in range(0,len(t),64):
   stop=min(start+64,len(t));sl={gd['time']:slice(start,stop)}
   fields={k:g[k].isel(sl).transpose(gd['time'],gd['depth'],gd['lat'],gd['lon']).values.astype('float64') for k in ['thetao','uo','vo']}
   hchunk=b.mlotst.isel({bd['time']:slice(start,stop)}).values.astype('float64');qchunk=b.qnet.isel({bd['time']:slice(start,stop)}).values;swchunk=b.ssr.isel({bd['time']:slice(start,stop)}).values
   for j,i in enumerate(range(start,stop)):
    h=hchunk[j];hu=np.maximum(h,5);good=np.isfinite(h)&(h>0)
    tm,tb,ok=c.layer_mean_and_below(fields['thetao'][j],z,edges,hu,good)
    u,_,uok=c.layer_mean_and_below(fields['uo'][j],z,edges,hu,good);v,_,vok=c.layer_mean_and_below(fields['vo'][j],z,edges,hu,good)
    we=np.zeros_like(h) if hp is None else np.maximum(0,hu-hp)/86400
    valid=ocean&ok&uok&vok&np.isfinite(we)&np.isfinite(qchunk[j])&np.isfinite(swchunk[j])&((we==0)|np.isfinite(tb))
    vals=dict(h=hu,qnet=qchunk[j],ssr=swchunk[j],u=u,v=v,tb=np.where(we==0,np.nan_to_num(tb,nan=0),tb),we=we,tm=tm,valid=valid)
    for k,a in vals.items():out[k][i]=np.where(valid,a,0).astype('float32')
    valid_count+=int(valid.sum());hp=hu
   if start%1024==0:print('features',stop,len(t),flush=True)
  for a in out.values():a.flush()
  save(FEATURES/'manifest.json',dict(status='complete',created=stamp(),shape=shape,valid_ocean_cell_days=valid_count,source={str(BUDGET):sha(BUDGET),str(GLORYS):sha(GLORYS)},code={str(Path(__file__)):sha(__file__),str(ROOT/'scripts/experiment_mixed_layer_budget_cpu.py'):sha(ROOT/'scripts/experiment_mixed_layer_budget_cpu.py')},sha256={str(FEATURES/(k+'.npy')):sha(FEATURES/(k+'.npy')) for k in keys},we='max(0,max(H[t],5)-max(H[t-1],5))/86400; first day0 unused by training',Tb='first finite thetao level center strictly deeper than max(H,5)',Tm='v6 midpoint-interface weighted layer mean; NOAA is training proxy instead',difference_from_CPU_v6='backward MLD difference at forcing t, not forward t+1 minus t; explicitly requested GPU proxy',future_forcing='oracle; no future SST or derived fields in x_nodes',fit_statistics=False))
 log('prepare_mixed_budget_features.py: CPU 3D→2D feature cache completed; raw/v1/v2/v3 preserved; backward MLD proxy and oracle-forcing scope recorded.');print('CPU features complete')
if __name__=='__main__':main()
