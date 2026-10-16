"""Additive CPU audit; never changes the frozen v7 model/protocol or source data."""
from mixed_budget_common import *
import numpy as np
import pandas as pd
import xarray as xr
import torch
from copy import copy
from torch.utils.data import default_collate
import mixed_budget_physics as mb
import train_mixed_budget_gpu as trainer
import train_v3_study as w

OUT=STUDY/'assumption_audit'

def metrics(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 assert x.size and np.isfinite(x).all() and np.isfinite(y).all()
 e=x-y
 return dict(n=int(e.size),bias_k=float(e.mean()),rmse_k=float(np.sqrt(np.mean(e*e))),std_difference_k=float(e.std()),mae_k=float(abs(e).mean()),p05_difference_k=float(np.quantile(e,.05)),p95_difference_k=float(np.quantile(e,.95)),correlation=float(np.corrcoef(x,y)[0,1]))

def temperature_audit(p,ar):
 rows=[];increments=[];splits=[]
 with xr.open_dataset(BUDGET) as ds:
  sst=ds.sst_mean.values.astype(np.float64);dates=pd.DatetimeIndex(ds.time.values);sea=ds.ocean_mask_static.values>.5
  tm=np.asarray(ar['tm'],dtype=np.float64)
  previous=json.loads((STUDY/'subset_input_audit.json').read_text())['rows']
  for job in p['jobs']:
   if job['model']!='B0':continue
   mf=Path(job['manifest']);m=json.loads(mf.read_text());assert sha(mf)==job['manifest_sha256']
   old=next(r for r in previous if (r['ratio'],r['lead'],r['seed'])==(job['ratio'],job['lead'],job['seed']))
   assert old['stats_sha256']==digest(m['statistics'])
   h=m['history'];L=m['lead'];starts=np.array(m['train_start_indices']);val=np.array(m['val_start_indices']);test=np.array(m['test_start_indices'])
   inputs=np.unique(starts[:,None]+np.arange(h));targets=starts+h-1+L
   assert np.array_equal(inputs,m['input_time_indices']) and np.array_equal(targets,m['target_time_indices'])
   forcing=np.unique(starts[:,None]+h-1+np.arange(L))
   assert not np.intersect1d(starts,val).size and not np.intersect1d(starts,test).size
   assert max(inputs.max(),targets.max(),forcing.max())<val.min()+h-1+L
   clim=mf.parent/m['climatology_file'];assert sha(clim)==m['climatology_sha256']
   with np.load(clim) as c:assert np.array_equal(c['input_time_indices'],inputs)
   common=dict(ratio=job['ratio'],lead=L,seed=job['seed'])
   splits.append(dict(**common,status='PASS',selected_input_days=len(inputs),selected_forcing_days=len(forcing),last_train_target=str(dates[targets.max()].date()),first_val_target=str(dates[val.min()+h-1+L].date()),first_test_target=str(dates[test.min()+h-1+L].date()),stats_sha256=digest(m['statistics']),manifest_sha256=sha(mf),rhs_scale=m['physics_rhs_scale'],note='Sample/target separation; adjacent split histories can overlap. No held-out fitting or Test metrics.'))
   # Equal weight per unique date and ocean cell, no duplicated history-window weights.
   for split,ids in [('selected_train_input',inputs),('validation_target',val+h-1+L)]:
    xx=sst[ids][:,sea];yy=tm[ids][:,sea];hh=np.asarray(ar['h'][ids][:,sea],dtype=float);we=np.asarray(ar['we'][ids][:,sea],dtype=float)
    assert np.all(ar['valid'][ids][:,sea]>0)
    groups=[('all',np.ones(xx.shape,dtype=bool))]
    for name,months in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:groups.append((name,np.broadcast_to(np.isin(dates[ids].month,months)[:,None],xx.shape)))
    for name,low,high in [('H5_10',5,10),('H10_20',10,20),('H20_50',20,50),('H50plus',50,np.inf)]:groups.append((name,(hh>=low)&(hh<high)))
    for name,select in groups:
     if not select.any():continue
     d=-86400*we[select]*(xx[select]-yy[select])/hh[select]
     rows.append(dict(**common,split=split,group=name,**metrics(xx[select],yy[select]),entrainment_proxy_difference_mean_kday=float(d.mean()),entrainment_proxy_difference_rms_kday=float(np.sqrt(np.mean(d*d))),mean_transmitted_fraction=float(np.exp(-.06*hh[select]).mean())))
   for split,st in [('selected_train',starts),('validation',val)]:
    current=st+h-1;end=current+L
    x=(sst[end]-sst[current])[:,sea];y=(tm[end]-tm[current])[:,sea]
    increments.append(dict(**common,split=split,**metrics(x.ravel(),y.ravel())))
   print('temperature/split audit',common,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'temperature_proxy.csv',index=False)
 pd.DataFrame(increments).to_csv(OUT/'temperature_increment_proxy.csv',index=False)
 save(OUT/'split_statistics_checks.json',splits)
 return rows,increments,splits

def counterfactual(p,ar):
 trainer.install();torch.set_num_threads(1);results=[]
 for ratio,L in [(10,7),(50,14)]:
  job=next(j for j in p['jobs'] if (j['ratio'],j['lead'],j['seed'],j['model'])==(ratio,L,42,'A'))
  a,b,model,pa,cfg=trainer.setup(job,p,torch.device('cpu'));model.eval();initial=w.state_digest(model,False)
  for split in ['train','val']:
   dataset=getattr(b,split);indices=[0,len(dataset)//2]
   original=default_collate([dataset[i] for i in indices])
   # Change only strictly future qnet cache days; current input/history and on-disk data untouched.
   replacement=np.array(ar['qnet'],copy=True)
   future=np.unique(np.concatenate([np.arange(int(original['current_index'][i])+1,int(original['target_index'][i])) for i in range(2)]))
   replacement[future]+=100.
   altered=mb.ForcingDataset(dataset.base,{**ar,'qnet':replacement})
   changed=default_collate([altered[i] for i in indices])
   assert torch.equal(original['x_nodes'],changed['x_nodes']) and torch.equal(original['tm_norm_nodes'],changed['tm_norm_nodes'])
   assert not torch.equal(original['mixed_qnet'],changed['mixed_qnet'])
   with torch.no_grad():
    out=model(original['x_nodes'],base_nodes=original['tm_norm_nodes'],return_aux=True)
    out2=model(changed['x_nodes'],base_nodes=changed['tm_norm_nodes'],return_aux=True)
    assert torch.equal(out['pred'],out2['pred'])
    for arm in ['A','B']:
     pa.mixed_variant=arm
     loss1=w.t.compute_batch_losses(out,original,b.graph,*b.stats['target'],pa,.1,model)
     loss2=w.t.compute_batch_losses(out2,changed,b.graph,*b.stats['target'],pa,.1,model)
     assert torch.equal(loss1[1],loss2[1]) and not torch.equal(loss1[2],loss2[2])
     target_changed={**original,'y_raw':original['y_raw']+10,'y_nodes':original['y_nodes']+10/b.stats['target'][1]}
     target_loss=w.t.compute_batch_losses(out,target_changed,b.graph,*b.stats['target'],pa,.1,model)
     assert torch.equal(loss1[2],target_loss[2]) and not torch.equal(loss1[1],target_loss[1])
     # Run actual validation entry path: fixed model; oracle change cannot change SST RMSE.
     va=w.t.run_epoch(model,[original],b.graph,torch.device('cpu'),*b.stats['target'],physics_args=pa,lambda_physics=.1,epoch=1)
     vb=w.t.run_epoch(model,[changed],b.graph,torch.device('cpu'),*b.stats['target'],physics_args=pa,lambda_physics=.1,epoch=1)
     assert va['rmse']==vb['rmse'] and va['physics_loss']!=vb['physics_loss']
     results.append(dict(ratio=ratio,lead=L,seed=42,split=split,arm=arm,batch_size=2,future_qnet_offset_wm2=100,prediction_max_difference=float((out['pred']-out2['pred']).abs().max()),data_loss=float(loss1[1]),physics_before=float(loss1[2]),physics_after=float(loss2[2]),rmse_before=va['rmse'],rmse_after=vb['rmse'],future_noaa_target_not_in_physics=True,status='PASS'))
   del replacement,altered,changed
  assert initial==w.state_digest(model,False)
  b.ds.close();del b,model
 assert not torch.cuda.is_initialized()
 save(OUT/'counterfactual_checks.json',dict(status='PASS',optimizer_steps=0,cuda_initialized=False,scope='Fixed randomly initialized actual native model, actual dataset and validation entry point; train and validation batches, two leads; no performance estimate',rows=results))
 return results

def table(rows,keys):
 return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |' for r in rows)+'\n'

def report(rows,inc,splits,checks):
 src=ROOT/'outputs/v6_cpu_flux_residual_audit_20261011/penetration_comparison.csv';allpen=pd.read_csv(src)
 pen=allpen[(allpen.kh_m2s==0)&allpen.entrainment_proxy].copy()
 pen['mean_added_residual_k']=pen.bias_k_on-pen.bias_k_off
 pen.to_csv(OUT/'previous_cpu_sign_reversal.csv',index=False)
 assert (pen.bias_k_off<0).all() and (pen.bias_k_on>0).all()
 # Do not select a new Kd from these retrospective data.
 selected=[r for r in rows if r['group']=='all' and (r['split']=='selected_train_input' or r['seed']==42)]
 valinc=[r for r in inc if r['split']=='validation' and r['seed']==42]
 text=f'''# v7 단일지수·수온 대리변수·미래 forcing 감사

작성: {stamp()}

## 판정
- 모델 입력/고정 가중치 예측에 미래 forcing이 들어가는 경로: 코드 추적 및 실제 CPU 반사실 검사 PASS.
- v3 통계/기후값에 Validation/Test를 새로 맞추는 경로: 6개 manifest와 기후값 해시·선택 날짜 및 실제 loader 확인 PASS. 기존 fitting 코드를 변경하거나 통계를 재계산하지 않음.
- 미래 관측을 전혀 사용하지 않는 학습이라는 주장: 부정확. 학습 물리손실은 미래 ERA5·GLORYS 관측/재분석을 보조 감독으로 사용하는 oracle loss-only 구성이다.
- NOAA SST와 GLORYS 혼합층 평균 수온의 동일성: 성립하지 않음. 차이를 아래 수치로 기록하며 A/B를 완전한 혼합층 수온 예측모형으로 부르지 않는다.
- 전체 순단파 단일지수 근사 및 가열 과보정 가능성: 명시. Kd 최적성/광학적 정확성은 검증되지 않았으며 부호 반전으로 오류 원인을 확정하지 않는다.
- 신규 GPU/학습/optimizer step/다운로드 없음. 고정된 v7 소스·프로토콜·전처리 파일 변경 없음. Test 수온/예측 성능을 새로 평가하지 않음.

## 1. 전체 순단파에 적용한 단일지수 근사

구현은 Qpen=ssr·exp(-0.06·max(H,5)), Qeffective=qnet-Qpen이다. `ssr`은 파일의 전체 표면 순단파이며, 이 코드에는 파장대별 분리, 가시광 비율, 두 지수 항, 현장 광학계수 fitting이 없다. 따라서 전체 순단파를 하나의 감쇠길이 16.667 m로 표현하는 근사이다. 표면 순단파 `ssr`을 하향단파 `ssrd`라고 쓰지 않는다.

H=5/10/20/50 m의 통과 비율은 각각 {', '.join(f'{100*np.exp(-.06*h):.3f}%' for h in [5,10,20,50])}이다. H=5m에서는 순단파의 약 74%를 혼합층 가열에서 차감한다. 이 수치의 물리적 적합성은 현재 코드만으로 확증할 수 없다. 계수 0.06은 고정 가정이며 최적값이라는 표현을 피한다. `qnet`에 포함된 순단파 중 통과분만 차감하므로 코드상 이중 가산은 없다.

근거: `scripts/mixed_budget_physics.py:7,25-28`; `scripts/prepare_mixed_budget_features.py:17,23-31`; 기존 CPU `scripts/experiment_mixed_layer_budget_cpu.py:292-297,317`.

## 2. NOAA SST − GLORYS 혼합층 수온

단위 K(수온 차이). GLORYS Tm은 v7 cache의 midpoint-interface 두께 가중 평균이며 NOAA는 raw `sst_mean`이다. 661개 해양 격자를 동일 가중하며 각 고유 날짜는 한 번만 집계한다. 물리손실의 stencil support 660개와 모집단을 혼동하지 않는다. 상관은 절대 수온의 계절주기를 포함하며 anomaly correlation이 아니다. NaN/Inf 및 cache valid 검사를 통과한 값만 사용하고 누락값은 없었다.

{table(selected,['ratio','lead','seed','split','n','bias_k','rmse_k','correlation'])}

리드 기간 변화량 차이 ΔNOAA−ΔGLORYS(Validation 전체 샘플, K):

{table(valinc,['lead','n','bias_k','rmse_k','correlation'])}

계절·H별 차이 및 연행항 대리변수 효과는 `temperature_proxy.csv` 전체 표에 저장했다. 같은 Tb,H,we에서 NOAA 대신 GLORYS Tm을 사용할 때의 차이는 E_NOAA−E_GLORYS = -86400·we·(SST−Tm)/H이다. CSV의 이 항은 관측 시점의 수온 대체 효과이며, 학습 중 예측 보간 경로의 오차나 예측 성능으로 해석하지 않는다. Δ수온 불일치는 상수 bias 보정만으로 두 목표가 같아지지 않음을 보여준다. SST−Tm은 서로 다른 자료원과 수직 대표성의 복합 차이이며 순수 수직 층화로 원인을 단정할 수 없다.

근거: `scripts/prepare_mixed_budget_features.py:25-30`; `scripts/mixed_budget_physics.py:30-44`; `src/mhw/datasets.py:295-302`; `src/mhw/jeju_hilam_dataset.py:95-112`. 코드 변수 `Tm`은 GPU 손실에서는 현재 NOAA SST이며 cache `tm.npy`의 GLORYS 층평균과 다르다. A/B에서 Tb와 u/v는 GLORYS, 온도 공간구배 및 선형 중간 상태는 NOAA 예측 대리변수이다.

## 3. 미래 forcing: 세 종류의 정보 사용을 분리

1. 모델 forward: `x_nodes`는 start..start+29의 history, `tm_norm_nodes`는 현재 SST. 미래 필드는 forward 인자가 아니다. 현 시점 t에서 cache forcing은 t..t+L-1이며 strictly future 날짜도 포함한다. 일평균 t forcing의 실제 관측 가용 시각도 검증되지 않았으므로 운영 예보에서 알려진 자료로 취급하지 않는다.
2. 학습: 미래 qnet/ssr/H/u/v/Tb와 H의 backward 차분 we가 loss와 gradient에 영향을 준다. 미래 GLORYS 온도로 만든 Tb도 특권적 감독 정보다. 이것을 '미래 정보 없음'이라고 보고하면 안 된다. 관측 미래 NOAA 타깃은 data loss에만 사용하고 A/B physics RHS에는 직접 넣지 않는다. Prediction 경로가 원인적으로 과거 입력만 사용한다는 사실과 학습 보조 정보가 oracle이라는 사실을 동시에 보고해야 한다.
3. Validation/Test: native loop의 physics/total 진단은 oracle forcing에 의존한다. 고정 모델의 SST RMSE 및 예측값은 forcing을 바꿔도 동일했고 best/early stopping은 `va['rmse']`만 사용한다. Validation은 의도된 모델 선택 자료이며 독립 확인 자료라는 의미가 아니다. Test의 physics 진단도 oracle 진단이다. 이전 Test 확인 이력 때문에 확증적 성능 주장에는 별도의 독립 평가가 필요하다.

반사실 검사: 실제 v3 loader·native HiLAM을 CPU에서 고정하고, 각 샘플의 strictly future qnet에 +100 W/m²를 주었다. 실제 cache의 메모리 사본만 바꿨다. 두 리드×train/val×A/B 총 {len(checks)}개 비교에서 입력·예측·data loss·Validation RMSE는 동일, physics loss는 변했다. 미래 NOAA label을 +10 K 바꾸면 data loss는 변하고 A/B physics loss는 동일했다. 무작위 초기 모델의 batch2 검사이므로 성능 비교가 아니다.

{table(checks,['lead','split','arm','prediction_max_difference','physics_before','physics_after','rmse_before','rmse_after'])}

근거: `scripts/mixed_budget_physics.py:12-16,42-46`; `src/mhw/datasets.py:265-278,295-324`; `src/train_jeju_hilam.py:264-288,337-353,440-454`; `scripts/train_mixed_budget_gpu.py:98-104,116-128`. 미래 validity도 물리손실 마스크에만 쓰이며 SST RMSE 마스크를 바꾸지 않는다.

통계: cache는 모든 날짜를 담지만 per-day/직전 날짜 파생량이며 평균/표준편차/기후값을 fitting하지 않는다. v3 통계/기후값은 선택된 train history 및 target 통계를 고정 로딩한다. RHS scale은 기존 v3 선택 학습 통계이고 미래 held-out 값으로 재추정하지 않는다. split 경계의 입력 history 중복은 기존 sliding window 설계이며 train/Val/Test의 모든 관측 날짜가 완전히 분리됐다는 뜻은 아니다.

{table(splits,['ratio','lead','seed','selected_input_days','last_train_target','first_val_target','first_test_target','status'])}

근거: `src/mhw/subset_climatology.py:26-55,93-105,117-129`; `src/mhw/subset_preprocessing.py:135-167,170-175,187-206`; `scripts/prepare_mixed_budget_features.py:19-35`; `scripts/train_mixed_budget_gpu.py:22-30`. 6개 manifest 통계 SHA는 기존 CPU 사전검사와 동일, 기후값 fitting 날짜는 manifest 선택 input 날짜와 동일했다.

## 4. 가열항의 과도한 보정 가능성

기존 v6 CPU Validation 24개 window/lead, Kh=0, entrainment on의 누적 잔차 r=관측 수온 변화−누적 RHS(K). 아래는 학습 모델의 예측 오차가 아니다.

{table(pen.to_dict('records'),['lead','target','advection','rmse_k_off','rmse_k_on','bias_k_off','bias_k_on','mean_added_residual_k'])}

양의 투과열을 RHS에서 빼므로 r_on=r_off+ΣQpen/(ρCpH)·86400으로 평균 잔차가 양의 방향으로 이동한다. 음수→양수 반전은 평균 가열 과대에서 평균 가열 부족/냉각 과대로 넘어갔음을 뜻한다. 전체 순단파 단일지수 근사에서 투과열을 과하게 차감했을 가능성을 기록한다. 그러나 불완전한 연행·수평 이류·자료 불일치 등도 잔차에 남으므로, 부호 반전만으로 Kd 또는 가열항의 오류를 특정할 수 없다. RMSE 감소는 평균 bias의 크기 감소와 동의어가 아니며 둘 다 따로 보고한다.

이 기존 CPU 식은 관측 GLORYS Tm과 forward H[t+1]-H[t] 연행을 사용했다. v7 A/B는 예측 NOAA 보간 상태와 backward H[t]-H[t-1] 연행이다. 따라서 기존 CPU의 개선률/잔차 부호를 v7 학습 결과로 전이하지 않는다. 고정 λ, Kd, 공통 RHS scale은 이번 감사에서 변경하지 않았다.

원자료: `outputs/v6_cpu_flux_residual_audit_20261011/penetration_comparison.csv`; 잔차 정의 `scripts/experiment_mixed_layer_budget_cpu.py:309-324`.

## 논문/실험 문서에 사용할 문장

“Shortwave transmission below the mixed layer is approximated by a single exponential applied to the entire ERA5 net surface shortwave flux (Kd = 0.06 m⁻¹); this fixed approximation is not a spectrally resolved or locally calibrated optical model.”

“The proposed constraint uses predicted NOAA SST as a proxy for mixed-layer temperature and future ERA5/GLORYS fields as oracle, loss-only supervision; these future fields are excluded from predictor inputs, and checkpoint selection uses validation SST RMSE.”

“The penetration correction reduced cumulative budget RMSE in the CPU screening but reversed the mean residual from negative to positive, indicating a possible excessive reduction of mixed-layer heating that cannot be attributed uniquely to the optical approximation.”

## 限界と保全

해시/검사 상세: `checks.json`, `counterfactual_checks.json`, `split_statistics_checks.json`. 신규 산출물은 이 audit 디렉터리와 별도 감사 스크립트에만 저장하며 실행 기록은 `docs/research_log.md`에 누적한다. 원본 ERA5 시간 누적·GLORYS 최초 재격자화 및 예보 당시 자료 가용성은 이 검사로 입증되지 않는다. 새 GPU 실행 코드는 변경하지 않았고 중단된 smoke도 재시작하지 않았다.
'''
 text=text.replace('## 限界と保全','## 한계와 원본 보존')
 (OUT/'report_ko.md').write_text(text)


def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 OUT.mkdir(parents=True,exist_ok=True)
 p=protocol();verify_code(p);before=sha(STUDY/'experiment_protocol.json');ar=mb.arrays()
 fm=json.loads((FEATURES/'manifest.json').read_text())
 for path,h in fm['sha256'].items():assert sha(path)==h,path
 rows,inc,splits=temperature_audit(p,ar);checks=counterfactual(p,ar);report(rows,inc,splits,checks)
 verify_code(p);assert before==sha(STUDY/'experiment_protocol.json')
 for path,h in fm['sha256'].items():assert sha(path)==h,path
 save(OUT/'checks.json',dict(timestamp=stamp(),status='PASS_WITH_DOCUMENTED_ORACLE_AND_PROXY_LIMITATIONS',frozen_code_files=len(p['code_sha256']),protocol_sha256=before,source_code_unchanged=True,feature_cache_unchanged=True,manifest_count=len(splits),counterfactual_comparisons=len(checks),cuda_initialized=torch.cuda.is_initialized(),optimizer_steps=0,training_status=json.loads((STUDY/'training_status.json').read_text()),audit_script_sha256=sha(__file__)))
 log('scripts/audit_mixed_budget_assumptions_cpu.py 및 outputs/v7_gpu_mixed_budget_comparison_20261011/assumption_audit/: 전체 순단파 단일지수 근사·NOAA/GLORYS 수온 차이·미래 forcing 경로 및 CPU 잔차 부호 반전 감사. 6개 통계/기후값, 실제 두 리드 train/val A/B 반사실 8건 PASS; oracle 감독과 과보정 가능성 명시. 모델/프로토콜/전처리 변경 및 GPU/학습 없음.')
 print('AUDIT COMPLETE',str(OUT),flush=True)

if __name__=='__main__':main()
