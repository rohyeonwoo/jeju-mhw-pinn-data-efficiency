"""CPU-only evaluation of frozen forecast fields; no tuning or checkpoint selection."""
from mixed_budget_common import *
import argparse,math
import numpy as np
import pandas as pd
import torch
import xarray as xr
import mixed_budget_physics as mb
import experiment_mixed_layer_budget_cpu as c
from mhw.mhw_skill import day_of_year,compute_climatology_threshold,compute_mhw_skill_metrics,mhw_binary_field
from summarize_jeju_hilam_mhw_skill import anomaly_metrics

def finite(x):
 if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [finite(v) for v in x]
 if isinstance(x,np.generic):x=x.item()
 return None if isinstance(x,float) and not math.isfinite(x) else x

def common_physics(f,ar,sst,dx,dy,lead):
 idx=f['target_index']-lead;pred=np.squeeze(f['pred_raw']);mask=np.squeeze(f['mask']);tot={k:dict(sq=0.,n=0.,cum=0.,ncum=0.) for k in ['A','B']}
 with torch.no_grad():
  for start in range(0,len(idx),32):
   ii=idx[start:start+32];days=ii[:,None]+np.arange(lead)[None,:];bs={'mixed_'+k:torch.tensor(np.array(ar[k][days][:,:,None]),dtype=torch.float32) for k in ['h','qnet','ssr','u','v','tb','we','valid']}
   bs.update(dx=torch.tensor(dx,dtype=torch.float32)[None,None],dy=torch.tensor(dy,dtype=torch.float32))
   rr,rhs,s,ts=mb.path(torch.tensor(pred[start:start+32,None]),torch.tensor(sst[ii,None]),torch.tensor(mask[start:start+32,None]),bs,lead,False)
   for name,r in [('A',rr),('B',rr-ts['advection']*s)]:
    r=torch.where(s>0,r,torch.zeros_like(r));tot[name]['sq']+=float(r.double().square().sum());tot[name]['n']+=int(s.sum());rc=r.reshape(lead,len(ii),1,*r.shape[-2:]).sum(0);sc=s[:len(ii)];tot[name]['cum']+=float((rc.double().square()*sc).sum());tot[name]['ncum']+=int(sc.sum())
 return {f'common_{k}_{metric}':math.sqrt(v[numer]/max(v[denom],1)) for k,v in tot.items() for metric,numer,denom in [('residual_rms_kday','sq','n'),('cumulative_residual_rms_k','cum','ncum')]}

def prepare_threshold(ds):
 path=STUDY/'evaluation_reference.npz';sst=ds.sst_mean.values.astype('float32');dates=ds.time.values;doy=day_of_year(dates);ref=np.arange(len(dates))<round(len(dates)*.7)
 if path.exists():
  with np.load(path) as z:clim=z['clim'];th=z['threshold'];assert np.array_equal(z['reference_indices'],np.flatnonzero(ref))
 else:
  clim,th=compute_climatology_threshold(sst,doy,ref,window=5,quantile=.9);np.savez_compressed(path,clim=clim,threshold=th,reference_indices=np.flatnonzero(ref))
 save(STUDY/'evaluation_reference.json',dict(source_sha256=sha(BUDGET),reference_last_date=str(dates[np.flatnonzero(ref)[-1]]),method='unchanged first70% calendar, DOY window5, quantile.9, duration5; separate from subset input climatology',sha256=sha(path)))
 return sst,doy,clim,th

def evaluate_fields(j,split,ds,sst,doy,clim,th,ar):
 d=Path(j['run_dir']);folder=d/'evaluation'/split;dest=folder/'analysis.json'
 if dest.exists():return json.loads(dest.read_text())
 with np.load(folder/'fields.npz') as z:f={k:z[k] for k in z.files}
 idx=f['target_index'];assert np.all(np.diff(idx)==1),'MHW duration requires consecutive target dates';assert idx[0]>=round(len(sst)*.7)
 pred=np.squeeze(f['pred_raw']);obs=np.squeeze(f['target_raw']);sea=np.squeeze(f['mask'][0])>0;err=pred.astype(float)-obs
 assert np.isfinite(err[:,sea]).all();months=pd.DatetimeIndex(ds.time.values[idx]).month.to_numpy();land=~sea;coast=np.zeros_like(sea);coast[1:]|=land[:-1];coast[:-1]|=land[1:];coast[:,1:]|=land[:,:-1];coast[:,:-1]|=land[:,1:];coast&=sea
 row=dict(model=j['model'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],split=split,status='complete')
 row.update(rmse=float(np.sqrt(np.mean(err[:,sea]**2))),mae=float(np.mean(abs(err[:,sea]))),bias=float(np.mean(err[:,sea])))
 row.update(anomaly_metrics(pred,obs,clim[doy[idx]],sea));row['acc']=row.pop('test_acc')
 for tag,ms in [('coast',coast),('offshore',sea&~coast)]:row[tag+'_rmse']=float(np.sqrt(np.mean(err[:,ms]**2)))
 for season,mos in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:row[season+'_rmse']=float(np.sqrt(np.mean(err[np.isin(months,mos)][:,sea]**2)))
 row.update(compute_mhw_skill_metrics(pred,obs,doy[idx],th,sea,min_duration=5,area_frac=.15,onset_tolerance_days=3))
 pm=mhw_binary_field(pred,doy[idx],th,5)&sea;om=mhw_binary_field(obs,doy[idx],th,5)&sea
 hits=(pm&om).reshape(len(idx),-1).sum(1);miss=(~pm&om).reshape(len(idx),-1).sum(1);fa=(pm&~om).reshape(len(idx),-1).sum(1)
 row['mhw_grid_precision']=float(hits.sum()/max(hits.sum()+fa.sum(),1))
 row['mhw_day_sst_rmse']=float(np.sqrt(np.mean(err[om]**2))) if om.any() else None;row['mhw_day_sst_bias']=float(err[om].mean()) if om.any() else None
 dx,dy=c.spacing(ds.latitude.values,ds.longitude.values);row.update(common_physics(f,ar,sst,dx,dy,j['lead']))
 hist=json.loads((d/'history.json').read_text());stat=json.loads((d/'status.json').read_text());row.update(best_epoch=stat['best_epoch'],epochs=stat['epochs'],steps=stat['steps'],early_stopping=stat['early_stopping'],best_val_rmse=stat['best_val_rmse'],peak_reserved_mib=stat['peak_reserved_mib'],train_data_loss=hist[-1]['train_data_loss'],val_data_loss=hist[-1]['val_data_loss'],train_physics_loss=hist[-1]['train_physics_loss'],train_total_loss=hist[-1]['train_total_loss'],elapsed_seconds=hist[-1]['elapsed_seconds'])
 np.savez_compressed(folder/'daily_metrics.npz',target_index=idx,sse=np.sum(err[:,sea]**2,axis=1),count=np.full(len(idx),sea.sum()),signed=np.sum(err[:,sea],axis=1),hits=hits,misses=miss,false_alarms=fa,pred_mhw=pm,obs_mhw=om)
 row=finite(row);save(dest,row);return row

def comparisons(p):
 out=[];cases=[];rng=np.random.default_rng(20261011)
 for ratio,lead in [(10,7),(50,14)]:
  for split in ['val','test']:
   data={}
   for model in ['B0','B1','A','B']:
    data[model]=[]
    for seed in [42,7,123]:
     j=next(j for j in p['jobs'] if (j['ratio'],j['lead'],j['seed'],j['model'])==(ratio,lead,seed,model))
     path=Path(j['run_dir'])/'evaluation'/split/'daily_metrics.npz'
     if not path.exists():break
     with np.load(path) as z:data[model].append({k:z[k] for k in z.files})
   if any(len(v)!=3 for v in data.values()):continue
   idx=data['B0'][0]['target_index'];n=len(idx)
   assert all(np.array_equal(x['target_index'],idx) for v in data.values() for x in v)
   for left,right in [('A','B'),('A','B0'),('B','B0'),('B1','B0'),('A','B1'),('B','B1')]:
    for seed,aa,bb in zip([42,7,123],data[left],data[right]):
     om=aa['obs_mhw'];assert np.array_equal(om,bb['obs_mhw']);recovered=(om&~bb['pred_mhw']&aa['pred_mhw']).reshape(n,-1).sum(1);newmiss=(om&bb['pred_mhw']&~aa['pred_mhw']).reshape(n,-1).sum(1)
     for day,r,m in zip(idx,recovered,newmiss):cases.append(dict(ratio=ratio,lead=lead,split=split,seed=seed,left=left,right=right,target_index=int(day),recovered_grid_days=int(r),new_missed_grid_days=int(m)))
    # Fixed three fitted seeds; pool counts/errors, time blocks paired across all models/seeds.
    sums={name:{k:sum(x[k] for x in data[name]) for k in ['sse','count','hits','misses','false_alarms']} for name in [left,right]}
    def metrics(x,ix=None):
     a={k:v.sum() if ix is None else v[ix].sum(1) for k,v in x.items()};h=a['hits'];m=a['misses'];f=a['false_alarms']
     return dict(rmse=np.sqrt(a['sse']/a['count']),csi=h/np.maximum(h+m+f,1),pod=h/np.maximum(h+m,1),precision=h/np.maximum(h+f,1),f1=2*h/np.maximum(2*h+m+f,1))
    pointL=metrics(sums[left]);pointR=metrics(sums[right])
    for block in [90,30,180]:
     starts=rng.integers(n,size=(2000,math.ceil(n/block)));ix=((starts[:,:,None]+np.arange(block)[None,None,:])%n).reshape(2000,-1)[:,:n]
     a=metrics(sums[left],ix);b=metrics(sums[right],ix)
     for metric in a:
      lo,hi=np.quantile(a[metric]-b[metric],[.025,.975]);out.append(dict(ratio=ratio,lead=lead,split=split,left=left,right=right,metric=metric,difference=pointL[metric]-pointR[metric],ci_low=lo,ci_high=hi,block_days=block,replicates=2000,scope='conditional on3 fitted seeds; pooled metrics; grid MHW only; exploratory Test'))
 pd.DataFrame(out).to_csv(STUDY/'paired_block_intervals.csv',index=False);pd.DataFrame(cases).to_csv(STUDY/'mhw_missed_recovered.csv',index=False)

def write_report(p,rows):
 d=pd.DataFrame(rows);d.to_csv(STUDY/'results_by_seed.csv',index=False)
 available=d[d.status=='complete'] if len(d) else d
 for name,cols in [('sst_metrics.csv',['model','ratio','lead','seed','split','status','rmse','mae','bias','acc','coast_rmse','offshore_rmse','DJF_rmse','MAM_rmse','JJA_rmse','SON_rmse']),('mhw_metrics.csv',['model','ratio','lead','seed','split','status']+[k for k in d if k.startswith('mhw_')]),('physics_diagnostics.csv',['model','ratio','lead','seed','split','status']+[k for k in d if k.startswith(('common_','train_physics','train_total'))])]:d.reindex(columns=cols).to_csv(STUDY/name,index=False)
 summary=[]
 if len(available):
  for key,g in available.groupby(['model','ratio','lead','split']):
   r=dict(zip(['model','ratio','lead','split'],key));r['n_seeds']=len(g)
   for metric in ['rmse','mae','bias','mhw_grid_csi','mhw_grid_pod','mhw_grid_f1','common_A_residual_rms_kday','common_B_residual_rms_kday']:
    v=pd.to_numeric(g[metric]);r[metric+'_mean']=v.mean();r[metric+'_sd']=v.std(ddof=1) if len(v)>1 else None
   summary.append(r)
 pd.DataFrame(summary).to_csv(STUDY/'comparison_summary.csv',index=False)
 lines=['# v7 개선 혼합층식 파일럿','',f'상태: 평가 완료 {len(available)}/{len(p["jobs"])*2} model-splits. 미완료 수치를 추정하지 않음.','', '물리적 포괄성: B는 A에 수평 이류만 추가. 물리적 적합성: 동일 출력에 공통 A/B residual을 별도로 계산. 예측 성능: Validation-best SST·MHW 지표로 판정.','', 'NOAA SST를 Tm proxy로 사용하고, 미래 재분석은 loss-only oracle forcing이다. GPU 연행 proxy는 backward MLD 차분으로 기존 CPU v6 forward 차분과 다르다. B1은 학습 we·고정1K 대비로 A/B와 여러 항이 달라 단일 원인 비교가 아니다.','', '물리 λ=.1 고정, Rres/tie=0, 공통 v3 RHS mean-square scale. 이 파일럿의 최적 λ나 Kd를 의미하지 않는다.','', 'MHW threshold는 기존 raw SST 첫70% 달력 자료, DOY±5,90th percentile,5일 지속. subset-only 입력 climatology와 별개. target 전체 daily 연속성 검증 후 event filtering. Grid-day 점수와 domain-area event onset/duration 점수는 별도 열이며 서로 대체하지 않는다.','', '3 seed 평균/SD와 paired90일(30/180 민감도) 블록 구간을 별도 보고한다. 구간은 고정된3개 학습 seed에 조건부이고 독립 seed 모집단 유의성을 뜻하지 않는다. Test 반복 열람 이력 때문에 탐색적 결과이다.','', '|Model|Data %|Lead|Seed|Split|SST RMSE|MHW CSI|MHW POD|Bias|Common A residual K/day|Status|','|---|---|---|---|---|---|---|---|---|---|---|']
 for r in rows:
  vals=[r.get(k,'미완료') for k in ['model','ratio','lead','seed','split','rmse','mhw_grid_csi','mhw_grid_pod','bias','common_A_residual_rms_kday','status']];lines.append('| '+' | '.join(f'{x:.6g}' if isinstance(x,float) else str(x) for x in vals)+' |')
 lines+=['','CPU 사전 검증과 GPU smoke 상태는 preflight_report.json/smoke_test_report.json을 확인한다. GPU 미실행 상태는 PASS로 표시하지 않는다.','', '질문1–7(SST·MHW 개선/트레이드오프/자료 부족 이점/가설): 모든 paired 결과 전에는 미판정. 질문8(포괄성≠적합성): 별도 공통 residual로 검증. 질문9(후속 자료/연안 혼합): 예측 결과와 잔차 지도를 함께 검토해야 함. 질문10(Adaptive 재개): 이 실행기는 재개하지 않음.']
 (STUDY/'report_ko.md').write_text('\n'.join(lines)+'\n');(STUDY/'summary_ko.md').write_text(f'평가 완료 {len(available)}/48 model-splits. GPU 학습 결과를 미리 가정하지 않음. results_by_seed.csv 및 comparison_summary.csv 참조.\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pending-only',action='store_true');a=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='';torch.set_num_threads(1);p=protocol();verify_code(p);rows=[]
 if a.pending_only:
  rows=[dict(model=j['model'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],split=s,status='not_evaluated') for j in p['jobs'] for s in ['val','test']]
 else:
  with xr.open_dataset(BUDGET) as ds:
   sst,doy,clim,th=prepare_threshold(ds);ar=mb.arrays()
   for j in p['jobs']:
    for split in ['val','test']:
     if (Path(j['run_dir'])/'evaluation'/split/'metrics.json').exists():rows.append(evaluate_fields(j,split,ds,sst,doy,clim,th,ar))
     else:rows.append(dict(model=j['model'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],split=split,status='not_evaluated'))
  comparisons(p);guard_hashes('post_evaluation')
 write_report(p,rows);log('v7 CPU 요약 작성; pending_only='+str(a.pending_only))
if __name__=='__main__':main()
