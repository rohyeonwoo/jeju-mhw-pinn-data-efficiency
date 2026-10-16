"""Validation-only reports, actual-vs-virtual comparison; never loads test artifacts."""
from mixed_budget_common import *
import math,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def csvout(name,rows,fields):
 with (STUDY/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def table(cols,rs):return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def main():
 p=protocol();records=[];epochs=[];audits=[];histories={}
 for j in p['jobs']:
  d=Path(j['run_dir']);f=d/'status.json'
  if not f.exists():continue
  s=json.loads(f.read_text())
  if s['status']!='COMPLETED':continue
  h=json.loads((d/'history.json').read_text());assert len(h)==80;histories[j['name']]=h
  records.append(dict(model=j['label'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],n_train=j['n_train'],updates_per_epoch=j['updates_per_epoch'],**s))
  epochs.extend(dict(model=j['label'],ratio=j['ratio'],lead=j['lead'],seed=j['seed'],**r) for r in h)
  a=json.loads((d/'checkpoint_audit.json').read_text());audits.append([j['name'],a['best']['epoch'],a['post_warmup_best']['epoch'],a['best']['status'],a['post_warmup_best']['status']])
 fields=['model','ratio','lead','seed','n_train','updates_per_epoch','status','epochs','steps','best_epoch','best_val_rmse','post_warmup_best_epoch','post_warmup_best_val_rmse','post_minus_overall_rmse','checkpoint_roundtrip','checkpoint_dir']
 csvout('validation_results.csv',records,fields)
 ef=['model','ratio','lead','seed','actual_epoch','train_samples','epoch_updates','steps','last_batch_samples','train_rmse','train_data_loss','train_physics_loss','train_rres_l2','adversarial_loss','train_total_loss','val_rmse','lr','physics_warmup_factor','lambda_phys_effective','gradient_norm_mean','gradient_clip_fraction','best_epoch','best_val_rmse','post_warmup_best_epoch','post_warmup_best_val_rmse','sample_order_sha256','elapsed_seconds','peak_reserved_mib']
 csvout('epoch_metrics.csv',epochs,ef)
 improvements=[];groups=[]
 for r in records:
  base=next((b for b in records if (b['ratio'],b['lead'],b['seed'],b['model'])==(r['ratio'],r['lead'],r['seed'],'Baseline')),None)
  if base:improvements.append(dict(ratio=r['ratio'],lead=r['lead'],seed=r['seed'],model=r['model'],baseline_val_rmse=base['best_val_rmse'],model_val_rmse=r['best_val_rmse'],improvement_pct=100*(base['best_val_rmse']-r['best_val_rmse'])/base['best_val_rmse']))
 csvout('seed_improvements.csv',improvements,['ratio','lead','seed','model','baseline_val_rmse','model_val_rmse','improvement_pct'])
 for key in sorted({(r['ratio'],r['lead'],r['model']) for r in records}):
  rs=[r for r in records if (r['ratio'],r['lead'],r['model'])==key];vs=[r['best_val_rmse'] for r in rs];im=[r['improvement_pct'] for r in improvements if (r['ratio'],r['lead'],r['model'])==key]
  groups.append([*key,len(rs),f'{statistics.mean(vs):.6f} ± {statistics.stdev(vs):.6f}' if len(vs)>1 else f'{vs[0]:.6f} (SD 미정)',f'{statistics.mean(im):.3f}%' if im else '미정',f'{sum(v>0 for v in im)}/{len(im)}'])
 comparisons=[]
 for r in records:
  name=f'{r["model"]}_r{r["ratio"]}_lead{r["lead"]}_seed{r["seed"]}';f=OLD_STUDY/'runs'/name/'status.json'
  if not f.exists():continue
  s=json.loads(f.read_text())
  if s['status']!='trained':continue
  comparisons.append([name,s['steps'],r['steps'],s['best_epoch'],r['best_epoch'],f'{s["best_val_rmse"]:.6f}',f'{r["best_val_rmse"]:.6f}',f'{r["best_val_rmse"]-s["best_val_rmse"]:+.6f}'])
 (STUDY/'figures').mkdir(exist_ok=True)
 for ratio,lead,seed in sorted({(r['ratio'],r['lead'],r['seed']) for r in records}):
  fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
  for r in records:
   if (r['ratio'],r['lead'],r['seed'])!=(ratio,lead,seed):continue
   h=histories[f'{r["model"]}_r{ratio}_lead{lead}_seed{seed}'];x=[v['epoch'] for v in h]
   for ax,k in zip(axs.flat,['train_rmse','val_rmse','lr','lambda_phys_effective']):ax.plot(x,[v[k] for v in h],label=r['model'])
  for ax,title in zip(axs.flat,['Online Train RMSE (K)','Validation RMSE (K)','Learning rate','Effective physics lambda']):ax.set(title=title,xlabel='Actual epoch');ax.axvline(5,ls=':',color='gray');ax.grid(alpha=.2);ax.legend()
  fig.suptitle(f'{ratio}% / Lead {lead} / Seed {seed}')
  fig.savefig(STUDY/f'figures/r{ratio}_lead{lead}_seed{seed}.png',dpi=150);plt.close(fig)
 before=sum(r['best_epoch']<5 for r in records)
 sections=['# Actual Epoch Oracle96 Validation 보고',stamp(),f'완료 {len(records)}/96. 미완료 run은 성능 집계에서 제외한다. Test 평가: 실행하지 않음.',
 '## 모델·자료량·Lead별 Validation',table(['Ratio','Lead','Model','완료 seeds','Val RMSE mean ± sample SD','Seed별 개선율 평균','개선 seeds'],groups),
 '## Best 및 warmup 진단',f'전체 best가 full-weight 시작(epoch5) 전인 run: {before}/{len(records)}. 이 run들의 best로 완전히 활성화된 물리 제약의 효과를 입증했다고 주장하지 않는다.',
 table(['Run','Overall best epoch','Post-warmup best epoch','Overall roundtrip','Post roundtrip'],audits),
 '## 기존 Virtual Epoch와 비교',table(['Run','Old updates','Actual updates','Old best virtual epoch','New best actual epoch','Old Val','New Val','New−Old K'],comparisons),
 'epoch 숫자는 서로 동일한 학습량이 아니다. 기존 10% 1 virtual epoch≈19.92순회; 신규 actual epoch=1순회. 학습 예산, Validation 빈도, LR의 update 기준 진행률, physics warmup(기존0→신규5)이 함께 바뀌었다. 변화 전체를 과적합 감소만의 인과효과로 설명하지 않는다.',
 '## 해석 및 연구 질문']
 for ratio,lead in sorted({(r['ratio'],r['lead']) for r in records}):
  sections.append(f'- {ratio}% Lead{lead}: '+ '; '.join(f'{g[2]} {g[4]}, 개선 {g[5]}, 개선 seeds {g[6]}' for g in groups if g[:2]==[ratio,lead]))
  no=[r for r in records if r['ratio']==ratio and r['lead']==lead and r['model']=='NoAdv'];ad=[r for r in records if r['ratio']==ratio and r['lead']==lead and r['model']=='Adv'];pairs=[(n,a) for n in no for a in ad if n['seed']==a['seed']]
  if pairs:sections.append(f'  Adv가 NoAdv보다 낮은 Val RMSE인 seed: {sum(a["best_val_rmse"]<n["best_val_rmse"] for n,a in pairs)}/{len(pairs)}.')
 sections+=['## 일반화 곡선', '각 조건의 80-epoch Train/Validation·LR·physics λ 곡선은 figures/에 저장했다. Training RMSE는 optimizer update 중 배치별 online 집계이다. 모델을 고정해 Train 전체를 재평가한 지표가 아니다.',
 '## 한계와 Test 계획','3 seeds의 SD와 개선 seed 수를 함께 제시하며 통계적 확증을 주장하지 않는다. Training-only oracle은 SST-label efficiency 비교이며 total-information efficiency 비교가 아니다. NOAA SST는 혼합층 수온 대리변수, 투과식은 전체 단파복사의 단일지수 근사, Kd=.06은 파일럿 상수다. 기존 Test 결과를 확인한 이후 프로토콜을 변경했으므로 향후 같은 Test 평가도 탐색적 분석이다.',
 '96개 COMPLETED 및 양 checkpoint 검증 PASS 후, 전체 Validation-best만 고정하고 별도 승인된 Test 평가를 수행한다. Post-warmup best는 진단용이며 Test로 둘 중 하나를 선택하지 않는다. Test 실행 명령은 본 관리자가 호출하지 않는다. SST/MHW·해안/계절별·forcing 독립성 평가 계획을 별도 확정한다.']
 (STUDY/'report.md').write_text('\n\n'.join(sections)+'\n')
 (STUDY/'checkpoint_audit.md').write_text('# Checkpoint 감사\n\n'+table(['Run','Overall best epoch','Post-warmup best epoch','Overall reload','Post reload'],audits)+'\n\n공식 모델: overall best.pt. post_warmup_best.pt는 진단 전용.\n')
 print(f'Validation summary: {len(records)}/96 completed; no Test',flush=True)
if __name__=='__main__':main()
