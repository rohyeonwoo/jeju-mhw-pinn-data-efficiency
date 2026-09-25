"""Read-only analysis of the 60 completed batch16 runs; writes small reports only."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mhw_subset_analysis_mpl')
import csv
import hashlib
import itertools
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/ceph_data/yeonwoo/MHW_subset/outputs/data_efficiency')
OUT = ROOT / 'docs/batch16_results'
SEEDS = [42, 7, 123]
RATIOS = ['10', '25', '50', '100', 'Full']
MODELS = ['jeju_hilam', 'pi_jeju_hilam']
METRICS = {
    'test_rmse': -1, 'test_mae': -1, 'test_acc': 1,
    'mhw_grid_csi': 1, 'mhw_grid_pod': 1, 'mhw_grid_far': -1,
    'tail_strong_plus_csi': 1, 'fixed_mhw_obs_event_rmse': -1,
    'mhw_event_onset_mae_days': -1, 'mhw_event_duration_mae_days': -1,
}


def read(p):
    return json.loads(p.read_text())


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def exact_sign_flip(d):
    # Enumerate all 2^3 paired model-label swaps. No large-sample approximation.
    means = np.array([np.mean(d * signs) for signs in itertools.product([-1, 1], repeat=len(d))])
    return float(np.mean(np.abs(means) >= abs(np.mean(d)) - 1e-14))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lookup, daily, runs, provenance = {}, {}, [], []
    references = {}
    invariant = ['batch_size', 'history', 'precision', 'hidden_dim', 'temporal_hidden_dim',
                 'processor_layers', 'mlp_hidden_layers', 'lr', 'initial_lr', 'weight_decay',
                 'S', 'max_steps', 'warmup_steps', 'selection_metric',
                 'early_stopping_patience_validations', 'data_sha256', 'source_sha256',
                 'split_sha256', 'full_statistics', 'include_residual_feature',
                 'target_mean', 'target_std', 'val_samples', 'test_samples']
    for folder in ['allocation_batch16_5plus5', 'full_batch16_matched']:
        state = read(BASE / folder / '_queue/batch_status.json')
        assert state['status'] == 'completed'
        assert all(j['status'] == 'completed' and j['returncode'] == 0 for j in state['jobs'])
    for ratio in RATIOS:
        root = BASE / ('full_batch16_matched/full' if ratio == 'Full' else f'allocation_batch16_5plus5/ratio_{ratio}')
        source = root / 'mhw_skill.json'
        payload = read(source)
        provenance.append({'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'config': payload['config']})
        rows = [r for r in payload['rows'] if r['model'] in MODELS]
        assert len(rows) == 12 and all(r['status'] == 'ok' for r in rows)
        for row in rows:
            lead, seed, model = row['lead_days'], row['seed'], row['model']
            key = ratio, lead, model, seed
            assert key not in lookup
            lookup[key] = row
            metric = read(Path(row['metrics_path']))
            d = Path(row['metrics_path']).parent
            config = read(d / 'jeju_hilam_config.json')
            progress = read(d / 'progress.json')
            assert progress['status'] == 'completed' and not config['smoke_only']
            assert config['batch_size'] == 16 and config['S'] == 478
            assert config['train_samples'] == ({7: 7644, 14: 7639}[lead] if ratio == 'Full' else {'10':376,'25':987,'50':1974,'100':3995}[ratio])
            assert metric['seed'] == seed and metric['lead'] == lead
            assert config['physics_mode'] == ('none' if model == MODELS[0] else 'p0_original')
            if ratio != 'Full':
                assert config['subset_seed'] == seed and config['subset_ratio'] == int(ratio)
            shared = {k: config[k] for k in invariant}
            if lead not in references:
                references[lead] = {'config': shared}
            assert shared == references[lead]['config'], (key, 'invariant config differs')
            with np.load(row['fields_path']) as fields:
                pred, obs, mask = (np.squeeze(fields[k], axis=1) for k in ['pred_raw', 'target_raw', 'mask'])
                idx = fields['target_index']
                assert np.all(np.diff(idx) == 1)
                hashes = {k: digest(v) for k,v in [('obs',obs),('mask',mask),('index',idx)]}
                if 'array_hashes' not in references[lead]: references[lead]['array_hashes'] = hashes
                assert hashes == references[lead]['array_hashes'], (key, 'test arrays differ')
                sea = mask[0] > 0
                assert np.all(mask == mask[0])
                assert np.isfinite(pred[:,sea]).all() and np.isfinite(obs[:,sea]).all()
                mse = np.mean((pred[:,sea].astype(np.float64) - obs[:,sea])**2, axis=1)
                assert np.isclose(np.sqrt(mse.mean()), row['test_rmse'], atol=2e-6, rtol=0)
                assert np.isclose(metric['test_rmse'], row['test_rmse'], atol=2e-6, rtol=0)
                daily[key] = mse
            run = {'ratio':ratio,'lead':lead,'seed':seed,'model':model,'train_samples':config['train_samples'],
                   'best_step':progress['best_step'],'actual_steps':progress['step'],
                   'validation_count':progress['validation_count'],'elapsed_seconds':progress['elapsed_seconds'],
                   'metrics_path':row['metrics_path']}
            run.update({m:row[m] for m in METRICS})
            runs.append(run)
        print('Audited', ratio, '12 runs', flush=True)
    assert len(runs) == 60
    summaries = []
    for ratio, lead, metric in itertools.product(RATIOS, [7,14], METRICS):
        a, p = [np.array([lookup[ratio,lead,model,seed][metric] for seed in SEEDS]) for model in MODELS]
        delta = p-a
        summaries.append({'ratio':ratio,'lead':lead,'metric':metric,'baseline_mean':float(a.mean()),
                          'baseline_sd':float(a.std(ddof=1)),'pinn_mean':float(p.mean()),'pinn_sd':float(p.std(ddof=1)),
                          'delta_pinn_minus_baseline':float(delta.mean()), 'delta_sd':float(delta.std(ddof=1)),
                          'relative_improvement_pct':float(METRICS[metric]*delta.mean()/a.mean()*100),
                          'improved_seed_count':int(np.sum(METRICS[metric]*delta>0)),
                          'delta_seed42':float(delta[0]),'delta_seed7':float(delta[1]),'delta_seed123':float(delta[2]),
                          'paired_exact_two_sided_p':exact_sign_flip(delta)})
    # Conditional test-period uncertainty: resample common time blocks across all
    # grids and all six fitted models. This does NOT estimate training-seed uncertainty.
    boots = []
    for lead in [7,14]:
        n = len(daily['10',lead,MODELS[0],42])
        for block in [30,90,180]:
            rng = np.random.default_rng(20260925 + lead + block)
            starts = rng.integers(0,n,size=(4000,int(np.ceil(n/block))))
            indices = ((starts[:,:,None]+np.arange(block)) % n).reshape(4000,-1)[:,:n]
            for ratio in RATIOS:
                samples = []
                for model in MODELS:
                    per_seed = [np.sqrt(daily[ratio,lead,model,seed][indices].mean(axis=1)) for seed in SEEDS]
                    samples.append(np.mean(per_seed,axis=0))
                delta = samples[1]-samples[0]
                lo, hi = np.quantile(delta,[.025,.975])
                boots.append({'ratio':ratio,'lead':lead,'block_days':block,'replicates':4000,
                              'conditional_delta_rmse_ci_low':float(lo),'conditional_delta_rmse_ci_high':float(hi)})
            print('Bootstrap',lead,block,flush=True)
    csv_write(OUT/'runs.csv', runs)
    csv_write(OUT/'paired_summary.csv', summaries)
    csv_write(OUT/'conditional_time_bootstrap.csv', boots)
    def get(r,l,m): return next(s for s in summaries if s['ratio']==r and s['lead']==l and s['metric']==m)
    fig, axes = plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for col, lead in enumerate([7,14]):
        for row, metric in enumerate(['test_rmse','mhw_grid_csi']):
            ax=axes[row,col]
            for model,label,color in zip(MODELS,['SeaCast baseline','SeaCast + PINN (P0)'],['#2463a6','#c74e35']):
                vals=np.array([[lookup[r,lead,model,seed][metric] for seed in SEEDS] for r in RATIOS])
                prefix='baseline' if model==MODELS[0] else 'pinn'
                ax.errorbar(np.arange(5),vals.mean(axis=1),yerr=vals.std(axis=1,ddof=1),label=label,
                            marker='o',capsize=4,color=color,linewidth=1.7)
                for j in range(3):ax.scatter(np.arange(5)+(j-1)*.055,vals[:,j],s=13,alpha=.45,color=color)
            ax.axvline(3.5,color='gray',linestyle=':',linewidth=1)
            ax.set_xticks(np.arange(5),['10%\n376','25%\n987','50%\n1974','100%\n3995','Full\n7644' if lead==7 else 'Full\n7639'])
            ax.set_title(f'Lead {lead} days')
            ax.set_ylabel('Test RMSE (K); lower is better' if row==0 else 'MHW CSI; higher is better')
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Batch 16: mean +/- sample SD across three paired seeds\n100% block subset and original Full are distinct; categorical x axis',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=160)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    report = ['# Batch16 SeaCast baseline vs PINN(P0) 결과', '',
              '생성 시각: '+datetime.datetime.now(datetime.timezone.utc).isoformat(), '',
              '## 범위와 검증', '',
              '- 본 학습 60개(48 subset + 12 Full) 모두 큐 completed, 종료 코드 0. smoke 및 중단된 batch8 결과 제외.',
              '- 모든 실행의 batch16/S478/최대38240/통계/원본 소스 해시/lead별 val·test 인덱스와 공통 하이퍼파라미터 기록 일치 확인.',
              '- 예측 파일에서 lead별 관측·마스크·target index가 완전히 같은지 검증. 해양 격자 예측은 모두 유한하며 RMSE를 재계산해 저장값과 2e-6 K 이내 일치 확인.',
              '- RMSE/CSI는 테스트 지표이며 체크포인트는 validation RMSE로 선정했다. seed 평균 ± 표본 SD(ddof=1).',
              '- 100%는 3995개 블록 subset, Full은 lead7 7644 / lead14 7639개. 비율은 원래 Full 샘플의 비율이 아니다.', '',
              '## 전체 SST RMSE', '', '| Lead | 데이터 | Baseline 평균 ± SD | PINN 평균 ± SD | RMSE 감소율 | 개선 seed |', '|---|---|---|---|---|---|']
    for lead in [7,14]:
        for ratio in RATIOS:
            s=get(ratio,lead,'test_rmse')
            report.append(f"| {lead} | {ratio} | {s['baseline_mean']:.4f} ± {s['baseline_sd']:.4f} | {s['pinn_mean']:.4f} ± {s['pinn_sd']:.4f} | {s['relative_improvement_pct']:+.2f}% | {s['improved_seed_count']}/3 |")
    report += ['', '## MHW 탐지와 관측 MHW 구간 오차', '',
               'CSI와 POD는 높을수록, FAR는 낮을수록 좋다. Δ=P0−baseline. CSI/POD/FAR 차이는 0~1 척도의 절대 차이이며 ×100하면 percentage point다. 원본 평가기 정의: 학습 기준 DOY 90백분위, 최소 5일, 해양 격자·시간 전체의 카운트를 합쳐 계산. 자세한 설정은 analysis_metadata.json.', '',
               '| Lead | 데이터 | CSI baseline→PINN | ΔPOD | ΔFAR | 관측 MHW 구간 RMSE baseline→PINN |', '|---|---|---|---|---|---|']
    for lead in [7,14]:
        for ratio in RATIOS:
            c=get(ratio,lead,'mhw_grid_csi');p=get(ratio,lead,'mhw_grid_pod');f=get(ratio,lead,'mhw_grid_far');e=get(ratio,lead,'fixed_mhw_obs_event_rmse')
            report.append(f"| {lead} | {ratio} | {c['baseline_mean']:.4f}→{c['pinn_mean']:.4f} | {p['delta_pinn_minus_baseline']:+.4f} | {f['delta_pinn_minus_baseline']:+.4f} | {e['baseline_mean']:.4f}→{e['pinn_mean']:.4f} |")
    report += ['', '## 통계 해석', '',
               '- 추론 단위는 같은 subset/training seed의 baseline–PINN 쌍이다. 각 조건은 n=3으로 작다. 영가설 아래 쌍 내 모델 라벨 교환 가능성(차이의 부호 대칭성)을 가정한 양측 paired sign-flip 검정은 8가지 부호 배치만 있어 최소 p=0.25다. 정규성은 가정하지 않는다. 모든 조건에 이 검정을 적용했고 p<0.05인 조건은 없다. 이는 효과가 없다는 증명이 아니라 현재 반복 수로 강한 유의성 주장을 할 수 없다는 뜻이다.',
               '- ratio는 nested subset이고 lead/model은 같은 seed와 테스트 시계열을 공유한다. 30개 쌍 또는 공간·날짜별 오차를 모두 독립 표본으로 풀링해 p값을 작게 만들지 않았다.',
               '- 보조 분석: 동일한 날짜 블록을 모든 모델·seed·해양 격자에 공동 적용하는 circular moving-block bootstrap 4000회. 아래는 90일 블록의 ΔRMSE 95% percentile 구간. 30/180일 민감도 결과는 CSV에 있다. 이 구간은 이미 학습된 세 모델 쌍에 조건부인 테스트 기간 변동만 나타내며, 새로운 학습 seed·새 해역·미래 기후에 대한 신뢰구간이 아니다. 다중비교 보정 없는 탐색 분석이다.', '',
               '| Lead | 데이터 | ΔRMSE(PINN−baseline) 90일 블록 95% 구간 |', '|---|---|---|']
    for s in boots:
        if s['block_days']==90:report.append(f"| {s['lead']} | {s['ratio']} | [{s['conditional_delta_rmse_ci_low']:+.4f}, {s['conditional_delta_rmse_ci_high']:+.4f}] |")
    report += ['', '## 해석과 제약', '',
               '- subset 8개 조건 모두 평균 RMSE 개선. Full 두 조건은 평균 RMSE 악화. 데이터 제한 조건에서 P0가 정규화 역할을 한다는 가설과 부합하는 탐색적 증거지만, 물리식의 인과적 기전 입증은 아니다.',
               '- 가장 선명한 작은 데이터 사례: 10%/lead7 RMSE가 세 seed 모두 개선. 10%·25%/lead14의 MHW CSI도 세 seed 모두 개선한다.',
               '- 예외를 함께 보고한다: 25%/lead7 RMSE는 세 seed 중 하나만 개선; 50%/lead14 MHW CSI는 평균 악화. Full/lead14 CSI는 세 seed 모두 악화. 보편적 우월성·데이터가 적을수록 항상 개선 폭 증가를 주장하지 않는다.',
               '- 모든 subset PINN의 평균 SST RMSE는 같은 lead의 Full baseline보다 높다. 더 적은 데이터로 Full의 종합 성능을 대체했다고 말할 수 없다. 일부 CSI가 Full에 근접·상회해도 RMSE/오경보와 함께 해석해야 한다.',
               '- 공통 Full 통계에 선택되지 않은 학습 기간의 정보가 포함된다. 이는 gradient update에 쓰는 표본 접근량을 줄인 실험이며 모든 정보 접근량을 줄인 실험은 아니다.',
               '- subset seed=training seed이므로 subset 추출 변동과 초기화 변동을 분리할 수 없다. 동일 최대 학습 예산·early stopping 규칙을 적용했지만 실제 step 수는 다르다. 저장 step은 runs.csv에 제공한다.',
               '- baseline test_physics_loss=0은 해당 손실 미계산을 의미하므로 물리 일관성 우월성의 비교 지표로 사용하지 않았다.',
               '- 후속 확인 우선순위는 결과를 숨기지 않고 사전 고정한 조건에서 추가 seed와 가능하면 다른 시간 구간 반복이다. 이번 분석은 새 학습을 실행하거나 설정을 변경하지 않았다.', '',
               '![RMSE and MHW CSI](comparison.png)', '']
    (OUT/'report.md').write_text('\n'.join(report))
    (OUT/'analysis_metadata.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sources':provenance,'audit':'60 completed runs; config invariants, paired observed arrays and indices checked; RMSE recomputed',
        'seed_order':SEEDS,'bootstrap_replicates':4000,'bootstrap_seed_formula':'20260925 + lead + block_days',
        'bootstrap_scope':'conditional on six fitted models per condition; common circular time blocks; no seed resampling',
        'references':references},indent=2)+'\n')
    print('SUMMARY_RMSE',json.dumps([s for s in summaries if s['metric']=='test_rmse']),flush=True)
    print('BOOTSTRAP_90',json.dumps([s for s in boots if s['block_days']==90]),flush=True)
    print('REPORT', OUT/'report.md',flush=True)


if __name__=='__main__':
    main()
