"""Post-run CPU-only audit and Validation summary; no inference or Test metrics."""
import os,json,csv,statistics,math,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
OUT=Path(__file__).resolve().parent;S=OUT.parent;ROOT=S.parents[1]
os.environ['CUDA_VISIBLE_DEVICES']=''
(OUT/'runtime').mkdir(exist_ok=True)
for key in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[key]=str(OUT/'runtime')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=json.loads((S/'experiment_protocol.json').read_text());models=['Baseline','P0','NoAdv','Adv'];rows=[];histories={};configs={};checks=0;diffs=[]
for j in p['jobs']:
 d=Path(j['run_dir']);h=json.loads((d/'history.json').read_text());st=json.loads((d/'status.json').read_text());audit=json.loads((d/'checkpoint_audit.json').read_text());inv=json.loads((d/'forcing_invariance_val.json').read_text());cfg=json.loads((d/'config.json').read_text());best=min(h,key=lambda x:x['val_rmse']);last=h[-1]
 assert st['status']=='COMPLETED' and len(h)==80 and [x['epoch'] for x in h]==list(range(1,81))
 assert st['steps']==j['total_updates']==80*j['updates_per_epoch'] and st['validation_selection_calls']==80 and not st['early_stopping']
 assert best['epoch']==st['best_epoch'] and best['val_rmse']==st['best_val_rmse']
 for x in h:
  assert x['train_samples']==j['n_train'] and x['epoch_updates']==j['updates_per_epoch'] and x['sampling_exact_once']
  assert x['steps']==x['epoch']*j['updates_per_epoch']
  assert abs(x['lambda_phys_effective']-j['lambda_physics']*min(x['epoch']/5,1))<1e-12
  lr=2e-5+(x['epoch']/5)*(2e-4-2e-5) if x['epoch']<=5 else .5*2e-4*(1+math.cos(math.pi*(x['epoch']-5)/75))
  assert abs(x['lr']-lr)<1e-15
  assert all(math.isfinite(x[k]) for k in ['train_total_loss','train_rmse','val_rmse','gradient_norm_mean','gradient_norm_max'])
 assert inv['status']=='PASS' and inv['scope']=='full split' and inv['prediction_bitwise_identical'] and inv['model_state_unchanged']
 for name,c in audit.items():
  assert c['status']=='PASS' and (S/'checkpoints'/j['name']/(name+'.pt')).is_file()
  assert abs(c['expected_val_rmse']-c['reloaded_val_rmse'])<2e-6
  diffs.append(abs(c['expected_val_rmse']-c['reloaded_val_rmse']));checks+=1
 assert st['best_val_rmse']==st['post_warmup_best_val_rmse']
 rows.append(dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['label'],best_epoch=best['epoch'],best_val_rmse=best['val_rmse'],post_warmup_best_epoch=st['post_warmup_best_epoch'],epoch80_val_rmse=last['val_rmse'],deterioration_pct=100*(last['val_rmse']/best['val_rmse']-1),train_rmse_at_best=best['train_rmse'],train_rmse_epoch80=last['train_rmse'],updates=last['steps'],run_hours=last['elapsed_seconds']/3600,peak_reserved_mib=max(x['peak_reserved_mib'] for x in h)))
 histories[j['name']]=h;configs[j['name']]=cfg
for ratio in [10,25,50,100]:
 for lead in [7,14]:
  for seed in [42,7,123]:
   names=[f'{m}_r{ratio}_lead{lead}_seed{seed}' for m in models]
   for name in names[1:]:
    for k in ['stats','shared_initial_sha256','manifest_sha256','rhs_scale']:
     assert configs[name][k]==configs[names[0]][k]
    assert [r['sample_order_sha256'] for r in histories[name]]==[r['sample_order_sha256'] for r in histories[names[0]]]
for r in rows:
 b=next(x for x in rows if x['model']=='Baseline' and all(x[k]==r[k] for k in ['ratio','lead','seed']))
 r['improvement_pct']=100*(b['best_val_rmse']-r['best_val_rmse'])/b['best_val_rmse']
groups=[]
for ratio in [10,25,50,100]:
 for lead in [7,14]:
  for model in models:
   rs=[x for x in rows if (x['ratio'],x['lead'],x['model'])==(ratio,lead,model)]
   groups.append(dict(ratio=ratio,lead=lead,model=model,val_rmse_mean=statistics.mean(x['best_val_rmse'] for x in rs),val_rmse_sd=statistics.stdev(x['best_val_rmse'] for x in rs),improvement_pct_mean=statistics.mean(x['improvement_pct'] for x in rs),improvement_pct_sd=statistics.stdev(x['improvement_pct'] for x in rs),improved_seeds=sum(x['improvement_pct']>0 for x in rs),best_epoch_median=statistics.median(x['best_epoch'] for x in rs),epoch80_deterioration_pct_mean=statistics.mean(x['deterioration_pct'] for x in rs)))
def csvsave(name,data):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
csvsave('per_run_review.csv',rows);csvsave('comparison_summary.csv',groups)
# Compare only the four genuinely completed historical representative runs.
old=S.parent/'v8_training_only_oracle96_20261011';comparison=[]
for model in models:
 name=f'{model}_r10_lead7_seed42';st=json.loads((old/'runs'/name/'status.json').read_text());r=next(x for x in rows if (x['ratio'],x['lead'],x['seed'],x['model'])==(10,7,42,model))
 comparison.append(dict(model=model,old_val=st['best_val_rmse'],new_val=r['best_val_rmse'],vs_own_old_improvement_pct=100*(st['best_val_rmse']-r['best_val_rmse'])/st['best_val_rmse'],old_best_virtual_epoch=st['best_epoch'],new_best_actual_epoch=r['best_epoch'],old_updates=st['steps'],new_updates=r['updates']))
csvsave('virtual_vs_actual_seed42.csv',comparison)
resources=list(csv.DictReader((S/'resource_usage.csv').open()));assert {int(x['gpu']) for x in resources}=={7,8,9};assert max(int(x['our_live_processes']) for x in resources)<=2
leases=json.loads((S/'leases.json').read_text());assert not leases['running']
result=dict(status='PASS',training_runs=96,epoch_rows=96*80,checkpoint_reload_records=checks,max_reload_rmse_difference=max(diffs),best_epoch_range=[min(x['best_epoch'] for x in rows),max(x['best_epoch'] for x in rows)],best_epoch_median=statistics.median(x['best_epoch'] for x in rows),before_full_warmup=sum(x['best_epoch']<5 for x in rows),all_overall_and_post_warmup_best_equal=True,matched_initialization_statistics_order=True,sampling_and_schedules=True,full_validation_forcing_invariance=True,only_gpu789_max2=True,remaining_leases=0,test_evaluation='NOT_RUN',scope='existing logs; no checkpoint deserialization or new prediction')
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=dict(zip(models,['#315c9c','#dc8a27','#198570','#b64872']))
for col,lead in enumerate([7,14]):
 for model in models:
  rs=[x for x in groups if x['lead']==lead and x['model']==model];xx=[0,1,2,3]
  axes[0,col].errorbar(xx,[x['val_rmse_mean'] for x in rs],yerr=[x['val_rmse_sd'] for x in rs],label=model,color=colors[model],marker='o',capsize=3)
  if model!='Baseline':axes[1,col].plot(xx,[x['improvement_pct_mean'] for x in rs],label=model,color=colors[model],marker='o')
 axes[0,col].set_title(f'Lead {lead}: Validation RMSE (mean ± seed SD)');axes[0,col].set_ylabel('RMSE (K)')
 axes[1,col].set_title(f'Lead {lead}: paired improvement over Baseline');axes[1,col].set_ylabel('Mean of seed-wise improvements (%)');axes[1,col].axhline(0,color='gray',ls=':')
 for ax in axes[:,col]:ax.set_xticks([0,1,2,3],['10%','25%','50%','100%']);ax.set_xlabel('Training subset');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Actual Epoch Oracle96 | 96/96 completed | Validation only; three seeds')
for ext in ['png','pdf']:fig.savefig(OUT/f'validation_overview.{ext}',dpi=160)
plt.close(fig)
def table(cols,data):return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data])
wide=[]
for lead in [7,14]:
 for ratio in [10,25,50,100]:
  rr=[next(g for g in groups if (g['ratio'],g['lead'],g['model'])==(ratio,lead,m)) for m in models]
  wide.append([lead,f'{ratio}%']+[f'{g["val_rmse_mean"]:.4f} ± {g["val_rmse_sd"]:.4f}' for g in rr])
imp=[]
for lead in [7,14]:
 for ratio in [10,25,50,100]:
  rr=[next(g for g in groups if (g['ratio'],g['lead'],g['model'])==(ratio,lead,m)) for m in models[1:]]
  imp.append([lead,f'{ratio}%']+[f'{g["improvement_pct_mean"]:+.3f}% ({g["improved_seeds"]}/3)' for g in rr])
seedtab=[]
for seed in [42,7,123]:
 rs=[next(x for x in rows if (x['ratio'],x['lead'],x['seed'],x['model'])==(10,7,seed,m)) for m in models]
 seedtab.append([seed]+[f'{x["best_val_rmse"]:.6f}' for x in rs]+[f'{rs[2]["improvement_pct"]:+.3f}%'])
overfit=[]
for model in models:
 rs=[x for x in rows if x['model']==model]
 overfit.append([model,statistics.median(x['best_epoch'] for x in rs),f'{min(x["best_epoch"] for x in rs)}–{max(x["best_epoch"] for x in rs)}',f'{statistics.mean(x["deterioration_pct"] for x in rs):.2f}%',f'{sum(x["train_rmse_epoch80"]<x["train_rmse_at_best"] for x in rs)}/24'])
now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
text=['# Actual Epoch Oracle96 최종 Validation 결과 검토',now,'2026-10-12 13:48 KST에 96/96 완료. 새 GPU 실행·학습·Test 평가 없이 기존 개별 로그 7,680행을 재검증했다.',
'## 결론','가장 일관된 개선은 10%·Lead7 NoAdv: paired 개선율 평균 +3.045%, 3/3 seeds 개선이다. P0의 전반적인 우위는 확인되지 않았고, Adv는 모든 ratio×lead의 평균에서 Baseline보다 불리했다. 수평 이류 추가가 예측을 개선한다는 근거는 현재 Validation에서 약하다. 이는 물리식 자체의 진실성 또는 실제 해양에서 이류의 중요성을 부정하는 결과가 아니다.',
'## 학습 및 무결성','모든96개80actual epochs, 각epoch정확한sample1회/lastbatch포함, 자료량별1920/4960/9920/20000 optimizer calls, LR/physics warmup5 PASS. 모든 run 전체/post-warmup checkpoint 저장·재로드 기록192개 PASS. 같은ratio/lead/seed 입력통계·초기화·shuffle순서 일치. full Validation forcing 독립성96개 PASS. GPU7/8/9, 각최대2개, 남은worker lease0. Checkpoint 바이너리 재추론은 이번 검토에서 수행하지 않았다.',
'## Validation RMSE: 3 seeds 평균 ± 표본 표준편차',table(['Lead','Ratio',*models],wide),
'## Baseline 대비 개선율: 양수가 개선', '각 seed에서 100×(Baseline−Model)/Baseline을 계산한 뒤 평균했다. 괄호는 개선 seed 수이다. 평균 RMSE끼리 나눈 개선율과는 구분한다.',table(['Lead','Ratio','P0','NoAdv','Adv'],imp),
'## 핵심 10%·Lead7 seed별 결과',table(['Seed',*models,'NoAdv 개선율'],seedtab),
'NoAdv는 세 seed에서 개선됐으나 효과 크기는 +0.753%~+6.577%로 다르다. Seed3개뿐이고 Validation은 모델 선택에 사용했으므로 확증적 통계 우위나 Test 일반화를 주장하지 않는다. 10%·Lead14의 NoAdv 평균 개선 +0.866%는 1/3 seeds만 개선한 결과여서 일관되지 않다. 25%·Lead14는 +0.219%, 2/3 seeds로 작은 효과다. 50/100%에서는 NoAdv가 양Lead 모두 세seed 전부 악화했다.',
'## Best epoch와 과적합',table(['Model','Best epoch 중앙값','범위','Epoch80 악화율 평균(best 대비)','Train RMSE가 best시점보다 낮은 run'],overfit),
'Best 전체범위7–54, 중앙값22, epoch5 이전0/96. 모든run에서 전체best와 post-warmup best의 Val RMSE가 같았다. 따라서 warmup 완료 전에만 좋은 checkpoint가 선택되는 문제는 이번에 관찰되지 않았다. 다만 후반 Training 오차 감소와 Validation 악화는 여전히 있으므로 과적합이 완전히 해결됐다고 표현하지 않는다. Training RMSE는 online 집계이다.',
'## 기존 Virtual Epoch 대비: 동일 10%·Lead7·seed42만',table(['Model','Old Val','Actual Val','자기 기존결과 대비 개선','Old best virtual epoch','Actual best epoch'],[[x['model'],f'{x["old_val"]:.6f}',f'{x["new_val"]:.6f}',f'{x["vs_own_old_improvement_pct"]:.2f}%',x['old_best_virtual_epoch'],x['new_best_actual_epoch']] for x in comparison]),
'10% optimizer calls는38240→1920(94.98% 감소). old NoAdv의 Baseline 대비8.17% 개선은 seed42 단일조건이었다. 같은seed의 신규개선은6.58%; 신규3-seed 평균3.05%와 단순일대일 비교하면 안 된다. 네 모델 모두 자기 기존seed42 Val보다 개선했으나 epoch정의·학습량·Validation빈도·LR시간축·physics warmup0→5가 함께 바뀌어 변화 전체를 과적합 완화 하나로 귀속할 수 없다.',
'## 연구 해석과 다음 단계','자료량 증가에 따라 Baseline RMSE는 두Lead 모두 낮아졌다. NoAdv의 일관된 장점은 현재10%·Lead7에 한정되며 저자료량 전반 또는 긴lead의 우위로 일반화하지 않는다. P0의 평균 개선은Lead14의10/50%에서0.181/0.098%에 그치고 seed재현성이 제한적이다. Adv는NoAdv와의paired24조건 중2개에서만 더 낮은RMSE(22개는높음). NoAdv/Adv는이류 유무만 다른 통제비교이므로 현재구성에서이류제약추가의예측상 이점이 없다는 근거다. 물리적 열수지 적합성은 별개 평가로 유지한다.',
'현재 설정과 전체Validation-best checkpoint를 고정해 보존한다. 이번에는 Test를 실행하지 않았다. 이후 별도 승인된 Test에서 SST/MHW 및 해안/계절 오차를 평가하되, Test를 보고 λ·Kd·checkpoint 종류를 다시 선택하지 않는다. 기존Test를 확인한 뒤 변경한 프로토콜이므로 같은Test의 결과는탐색적분석이다. Oracle 추가감독을사용하므로SST-label efficiency이며total-information efficiency는입증하지않는다.',
'## 자료','![Validation overview](validation_overview.png)','[32개 조건 요약 CSV](comparison_summary.csv) · [96개 실행 감사 CSV](per_run_review.csv) · [검증 JSON](verification.json)',
'원본 결과: ../report.md, ../validation_results.csv, ../seed_improvements.csv, ../all96_gate.json, 각runs/<name>/history.json·status.json·checkpoint_audit.json·forcing_invariance_val.json. 집계코드 scripts/oracle96_actual/summarize.py; 새 독립검토 코드 analyze.py. 기존 원본 결과는 덮어쓰지 않았다.']
(OUT/'report_ko.md').write_text('\n\n'.join(text)+'\n')
with (ROOT/'docs/research_log.md').open('a') as f:f.write(f'\n- {now} — Actual Epoch96 완료 결과 CPU 재검토. `outputs/v8_actual_epoch_oracle96_20261011/results_review_20261012/`에 검토코드·한국어보고·32조건/96run CSV·검증JSON·전체비교PNG/PDF 추가. 모든96개80epochs 및checkpoint재로드192기록/샘플·스케줄·초기화·순서·forcing독립성 PASS. NoAdv10%Lead7 평균paired개선3.045%,3/3seed; P0일관우위없음, Adv모든ratio×lead평균악화. Best7–54/중앙22, warmup전0/96. Test·새학습·GPU계산 미실행, 기존결과보존.\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
