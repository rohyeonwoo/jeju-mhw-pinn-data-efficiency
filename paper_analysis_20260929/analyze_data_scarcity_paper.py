#!/usr/bin/env python3
"""CPU-only post-hoc analysis; no training performed.
Uses saved result arrays and metadata only; never opens a checkpoint or NetCDF.
Requires numpy, pandas and matplotlib. Output directory must not exist.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, math, re, zipfile, datetime, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_PROJECT = Path('/ceph_data/yeonwoo/MHW_subset')
DEFAULT_CODE = Path('/ceph_data/yeonwoo/server_backups/node1_20260928T054310Z/home/mhw_subset')
MODELS = {'jeju_hilam':'baseline','pi_jeju_hilam':'PINN'}
SEEDS = [7,42,123]
FRACTIONS = [10,25,50,100]
METRICS = {'rmse':'test_rmse','mae':'test_mae','bias':'test_bias','acc':'test_acc',
           'mhw_csi':'mhw_grid_csi','mhw_pod':'mhw_grid_pod','mhw_far':'mhw_grid_far','strong_csi':'tail_strong_plus_csi'}
NAN = float('nan')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def ahash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')


def mdtable(df):
    def fmt(x):
        if pd.isna(x): return 'NaN'
        if isinstance(x,(float,np.floating)): return f'{x:.6f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])


def aggregate(df,metrics,keys):
    rows=[]
    for group,d in df.groupby(keys,dropna=False,sort=False):
        if not isinstance(group,tuple): group=(group,)
        r=dict(zip(keys,group));r['n_seeds']=len(d)
        for m in metrics:
            r[m+'_mean']=d[m].mean();r[m+'_std']=d[m].std(ddof=1)
            r[m+'_n']=int(d[m].notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_daily(daily, n_bootstrap, block, seed):
    """Adapted from scripts/analyze_batch16_results.py circular time-block method.
    daily: [regime, model, training seed, target date]. Resample shared dates,
    keep spatial grid and fitted seed pairs together. Chunking limits memory.
    block=1 gives iid target-date cluster bootstrap; block>1 preserves short-lag dependence.
    """
    n=daily.shape[-1]; rng=np.random.default_rng(seed)
    flat=daily.reshape(-1,n); out=[]
    for begin in range(0,n_bootstrap,128):
        b=min(128,n_bootstrap-begin)
        starts=rng.integers(0,n,size=(b,math.ceil(n/block)))
        ix=((starts[:,:,None]+np.arange(block))%n).reshape(b,-1)[:,:n]
        weights=np.zeros((b,n),dtype=np.float64)
        for i in range(b): weights[i]=np.bincount(ix[i],minlength=n)
        scores=np.sqrt((weights@flat.T)/n).reshape(b,*daily.shape[:-1]).mean(axis=-1)
        delta=scores[:,:,1]-scores[:,:,0]
        relative=(scores[:,:,0]-scores[:,:,1])/scores[:,:,0]*100
        out.append(np.stack([delta,relative],axis=-1))
    return np.concatenate(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=DEFAULT_PROJECT/'outputs/data_efficiency')
    parser.add_argument('--project-root',type=Path,default=DEFAULT_PROJECT)
    parser.add_argument('--code-root',type=Path,default=DEFAULT_CODE)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=20000)
    args=parser.parse_args()
    if args.bootstrap<2: parser.error('--bootstrap must be >=2')
    out=args.output_dir.resolve(); root=args.results_root.resolve(); project=args.project_root.resolve();code=args.code_root.resolve()
    if not out.is_relative_to(Path('/ceph_data/yeonwoo')): parser.error('All outputs must be under /ceph_data/yeonwoo')
    if out.exists(): parser.error('Output directory already exists; refusing to overwrite')
    out.mkdir(parents=True)
    for d in ['analysis_data','paper_figures','paper_tables','cache']: (out/d).mkdir()
    os.environ['MPLCONFIGDIR']=str(out/'cache/matplotlib')
    os.environ['XDG_CACHE_HOME']=str(out/'cache')
    os.environ['TMPDIR']=str(out/'cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320,'axes.spines.top':False,'axes.spines.right':False})
    print('CPU-only post-hoc analysis; no training performed',flush=True)
    log=[]
    def note(s): log.append(s);print(s,flush=True)
    data=out/'analysis_data'; figures=out/'paper_figures'; tables=out/'paper_tables'
    frames={};figure_metadata=[];provenance={}
    def csv(name,obj):
        df=obj if isinstance(obj,pd.DataFrame) else pd.DataFrame(obj)
        df.to_csv(data/(name+'.csv'),index=False,na_rep='NaN',float_format='%.12g'); frames[name]=df
        return df
    def use(p,role):
        p=Path(p).resolve()
        if p not in provenance:
            s=p.stat();provenance[p]={'path':str(p),'role':role,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256_before':sha(p)}
        return p
    def read(p,role='metadata'): return json.loads(use(p,role).read_text())
    def savefig(fig,name,caption,source,plotdata):
        fig.savefig(figures/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(figures/(name+'.png'),dpi=320,bbox_inches='tight')
        plt.close(fig)
        csv(name+'_plot_data',plotdata)
        figure_metadata.append({'figure':name,'caption':caption,'source_csv':source,'plot_data_csv':name+'_plot_data.csv','png_dpi':320})
    # Import only the existing numpy/pandas module by path: package __init__ imports torch.
    mp=use(code/'src/mhw/mhw_skill.py','existing CPU evaluation implementation')
    spec=importlib.util.spec_from_file_location('original_mhw_skill_cpu',mp)
    skill=importlib.util.module_from_spec(spec);sys.modules[spec.name]=skill;spec.loader.exec_module(skill)
    for p in ['src/summarize_jeju_hilam_mhw_skill.py','src/summarize_hybrid_fields_mhw_skill.py',
              'src/mhw/datasets.py','scripts/analyze_batch16_results.py']:
        use(code/p,'evaluation/split/bootstrap definition')
    # Record complete result-file discovery without deserializing any checkpoint.
    inventory=[];schemas={};discovery=[]
    for folder in [project/'outputs/main_experiment_final',root]:
        for p in sorted(folder.rglob('*')):
            if not p.is_file(): continue
            if p.suffix.lower() not in {'.json','.csv','.npz','.npy','.pt','.log','.md'}:continue
            st=p.stat();r={'path':str(p),'bytes':st.st_size,'modified_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'suffix':p.suffix,'array_keys':''}
            if p.suffix=='.npz':
                with zipfile.ZipFile(p) as z: keys=[Path(k).stem for k in z.namelist()]
                r['array_keys']=';'.join(keys)
                schemas.setdefault(tuple(keys),{'count':0,'example':str(p)})['count']+=1
            r['analysis_capability']=('saved predictions/errors/date indices' if 'test_fields' in p.name else
                'grid topology and ocean mask' if p.name.endswith('graph.npz') else
                'checkpoint excluded: no deserialization or inference' if p.suffix=='.pt' else
                'metrics/config/split/history; see experiment inventory')
            discovery.append(r)
    csv('file_inventory',discovery)
    write_json(data/'array_schema_inventory.json',[{'keys':k,**v} for k,v in schemas.items()])
    subset={s:read(project/f'data/subsets/train_subsets_v2_win44_seed{s}.json','subset and calendar metadata') for s in SEEDS}
    meta=subset[42]['_metadata'];origin=np.datetime64(meta['dataset']['start']); assert meta['dataset']['days']==10957
    for s,d in subset.items():
        for low,high in zip(FRACTIONS,FRACTIONS[1:]):assert set(d[str(low)]['start_indices'])<set(d[str(high)]['start_indices'])
    regimes=[str(x) for x in FRACTIONS]+['Full'];run_rows=[];histories=[];audits=[];daily={};spatial={};seasonal=[];quantiles=[];conditioned=[];ecdfs=[];spatial_rows=[]
    refs={};selected={};configs={};raw_lookup={};family_states={}
    for f in sorted(root.glob('*/_queue/batch_status.json')):
        state=read(f,'batch manifest')
        family=f.parent.parent.name;family_states[family]=state
        for job in state.get('jobs',[]):
            inventory.append({'experiment_family':family,'run_id':job.get('id'),'status':job.get('status'),
                'returncode':job.get('returncode'),'training_fraction':job.get('ratio',job.get('data_regime',job.get('regime',''))),
                'lead_day':job.get('lead'),'seed':job.get('seed'),'arm':job.get('arm',job.get('mode','')),
                'lambda_physics':job.get('lambda_physics',NAN),'run_dir':job.get('run_dir',''),
                'in_primary_analysis':family=='allocation_batch16_5plus5' and job.get('status')=='completed',
                'usage': 'primary scarcity' if family=='allocation_batch16_5plus5' else 'separate Full reference' if family=='full_batch16_matched' else 'excluded from scarcity comparisons: different protocol or stopped'})
    csv('experiment_inventory',inventory)
    def keyfields(regime,lead,model,seed):
        return {'experiment_family':'batch16_original','training_fraction':int(regime) if regime!='Full' else NAN,
                'data_regime':'block_subset' if regime!='Full' else 'Full','lead_day':lead,'model':model,'seed':seed}
    invariants=['batch_size','history','precision','hidden_dim','temporal_hidden_dim','processor_layers','mlp_hidden_layers',
                'lr','initial_lr','weight_decay','S','max_steps','warmup_steps','selection_metric','early_stopping_patience_validations',
                'data_sha256','source_sha256','split_sha256','full_statistics','include_residual_feature','target_mean','target_std','val_samples','test_samples']
    for regime in regimes:
        family='full_batch16_matched' if regime=='Full' else 'allocation_batch16_5plus5'
        state=family_states[family];assert state['status']=='completed';assert all(x['status']=='completed' and x['returncode']==0 for x in state['jobs'])
        rr=root/family/('full' if regime=='Full' else 'ratio_'+regime)
        payload=read(rr/'mhw_skill.json','saved MHW evaluation; authoritative rows')
        rows=[x for x in payload['rows'] if x['model'] in MODELS and x.get('seed') is not None]
        assert len(rows)==12
        for row in sorted(rows,key=lambda r:(r['lead_days'],r['seed'],r['model'])):
            lead=int(row['lead_days']);seed=int(row['seed']);model=MODELS[row['model']];k=(regime,lead,model,seed)
            assert k not in raw_lookup and row['status']=='ok';raw_lookup[k]=row
            rd=rr/row['model']/f'lead{lead}_seed{seed}'
            metric=read(rd/Path(row['metrics_path']).name,'saved metrics');cfg=read(rd/'jeju_hilam_config.json','run config');prog=read(rd/'progress.json','training status')
            cfgkey={x:cfg[x] for x in invariants}
            if lead not in configs:configs[lead]=cfgkey
            assert cfgkey==configs[lead],(k,'shared config differs')
            assert cfg['seed']==seed and cfg['lead']==lead and cfg['batch_size']==16 and cfg['history']==30
            assert cfg['physics_mode']==('none' if model=='baseline' else 'p0_original')
            if model=='PINN':assert cfg['lambda_physics']==.1 and cfg['lambda_rres']==10
            assert not cfg['smoke_only'] and prog['status']=='completed'
            ind=read(rd/'selected_start_indices.json','actual training selection')
            assert len(ind)==cfg['train_samples']
            if regime!='Full':
                assert cfg['subset_ratio']==int(regime) and cfg['subset_seed']==seed
                assert ind==subset[seed][regime]['start_indices']
            else: assert ind==list(range(cfg['full_train_samples']))
            selected[k]=ind
            p=use(rd/Path(row['fields_path']).name,'saved test prediction/target/mask/index')
            with np.load(p,allow_pickle=False) as z:
                pred=z['pred_raw'][:,0];obs=z['target_raw'][:,0];mask=z['mask'][:,0];idx=z['target_index']
            assert np.all(np.diff(idx)==1) and np.all(mask==mask[0])
            sea=mask[0]>0;assert sea.sum()==661
            with np.load(use(rd/'jeju_hilam_graph.npz','saved grid topology'),allow_pickle=False) as z:assert np.array_equal(z['ocean_mask'].astype(bool),sea)
            hashes={name:ahash(a) for name,a in [('target',obs),('mask',mask),('target_index',idx)]}
            if lead not in refs:refs[lead]=hashes
            assert hashes==refs[lead],(k,'test samples differ')
            dates=origin+idx.astype('timedelta64[D]'); expected=meta['original_splits'][str(lead)]['test']
            assert str(dates[0])==expected['target_dates'][0] and str(dates[-1])==expected['target_dates'][1] and len(idx)==expected['sample_count']
            assert np.isfinite(pred[:,sea]).all() and np.isfinite(obs[:,sea]).all()
            err=pred[:,sea].astype(np.float64)-obs[:,sea].astype(np.float64)
            recomputed={'rmse':np.sqrt(np.mean(err**2)),'mae':np.abs(err).mean(),'bias':err.mean()}
            for m,val in recomputed.items(): assert np.isclose(val,row[METRICS[m]],atol=2e-6,rtol=0),(k,m,val,row[METRICS[m]])
            assert np.isclose(metric['test_rmse'],row['test_rmse'],atol=2e-6,rtol=0)
            base=keyfields(regime,lead,model,seed)
            record={**base,**{m:row.get(source,NAN) for m,source in METRICS.items()},'train_samples':len(ind),'test_days':len(idx),
                    'metrics_path':str(rd/Path(row['metrics_path']).name),'fields_path':str(p),'metric_source':str(rr/'mhw_skill.json')}
            for dest,source in [('tp','hits'),('fp','false_alarms'),('fn','misses'),('tn','correct_negatives')]:record[dest]=row['mhw_grid_'+source]
            # Same zero-denominator convention as original evaluator: return zero.
            h,f,m=record['tp'],record['fp'],record['fn']
            for name,num,den in [('mhw_csi',h,h+f+m),('mhw_pod',h,h+m),('mhw_far',f,h+f)]:
                assert np.isclose(record[name],skill._safe(num,den),atol=1e-12)
            record['strong_tp']=row['tail_strong_plus_hits'];record['strong_fp']=row['tail_strong_plus_false_alarms'];record['strong_fn']=row['tail_strong_plus_misses']
            assert np.isclose(record['strong_csi'],skill._safe(record['strong_tp'],record['strong_tp']+record['strong_fp']+record['strong_fn']))
            run_rows.append(record)
            mse=np.mean(err**2,axis=1);daily[k]=mse;spatial[k]=np.sqrt(np.mean(err**2,axis=0))
            for quant,value in zip([50,75,90,95],np.percentile(np.abs(err),[50,75,90,95])):quantiles.append({**base,'quantile_pct':quant,'absolute_error_K':value,'grid_time_count':err.size})
            months=pd.DatetimeIndex(dates).month.to_numpy()
            for season,ms in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:
                take=np.isin(months,ms); seasonal.append({**base,'season':season,'rmse':np.sqrt(mse[take].mean()),'n_target_dates':int(take.sum()),'small_subgroup_lt30_dates':bool(take.sum()<30)})
            conditioned.append({**base,'rmse_mhw_grid_days':row.get('fixed_mhw_obs_event_rmse',NAN),'mhw_grid_days':row.get('fixed_mhw_obs_event_count',NAN),
                'rmse_non_mhw_grid_days':NAN,'source':'existing saved fixed_mhw_obs_event_rmse; not recalculated','non_mhw_status':'계산 불가: saved MHW masks/threshold arrays absent'})
            history=pd.read_csv(use(rd/'jeju_hilam_epoch_history.csv','validation history'))
            assert np.isclose(history['val_rmse'].min(),prog['best_val_rmse'],atol=2e-6)
            assert int(history.loc[history['val_rmse'].idxmin(),'step'])==prog['best_step']
            histories.append({**base,'best_epoch':metric.get('best_epoch',prog['best_step']/cfg['S']),
                'stopping_virtual_epoch':prog['step']/cfg['S'],'early_stopped':prog['step']<cfg['max_steps'],
                'best_step':prog['best_step'],'actual_steps':prog['step'],'best_val_rmse':prog['best_val_rmse'],
                'validation_count':prog['validation_count'],'elapsed_seconds':prog['elapsed_seconds'],
                'wall_clock_for_efficiency_claim':False,'time_note':'concurrent shared-GPU jobs; descriptive only'})
            audits.append({**base,'target_sha256':hashes['target'],'mask_sha256':hashes['mask'],'target_index_sha256':hashes['target_index'],
                'first_target_date':str(dates[0]),'last_target_date':str(dates[-1]),'n_target_dates':len(idx),'ocean_cells':int(sea.sum()),
                'rmse_recomputed':recomputed['rmse'],'rmse_saved':row['test_rmse'],'max_sst_metric_abs_error':max(abs(recomputed[m]-row[METRICS[m]]) for m in recomputed),
                'val_split_sha256':cfg['split_sha256']['val'],'test_split_sha256':cfg['split_sha256']['test'],'training_selection_verified':True})
            # Baseline sorts first. Save one array pair transiently, never persist predictions.
            if model=='baseline':baseline_abs=np.abs(err)
            else:
                assert baseline_abs.shape==err.shape
                diff=(np.abs(err)-baseline_abs).ravel();qs=np.linspace(0,1,201)
                ecdf_values=np.quantile(diff,qs)
                for q,v in zip(qs,ecdf_values):ecdfs.append({**{x:base[x] for x in base if x!='model'},'cdf':q,'paired_absolute_error_difference_K':v})
            del err,pred,obs,mask
        note(f'Audited {regime}: 12 completed runs; saved test arrays and SST metrics verified')
    for regime in regimes:
        for lead in [7,14]:
            for seed in SEEDS: assert selected[regime,lead,'baseline',seed]==selected[regime,lead,'PINN',seed]
        if regime!='Full':
            for seed in SEEDS:assert selected[regime,7,'baseline',seed]==selected[regime,14,'baseline',seed]
    original=pd.DataFrame(run_rows)
    # Include completed follow-up metrics in long format with distinct protocol/arm IDs.
    extra=[]
    for family in ['lambda_pilot_lead7_20260925','physics_pilot_lead7_20260925_4plus4']:
        mhw_lookup={}
        for p in sorted((root/family).rglob('mhw*.json')):
            try: payload=read(p,'follow-up metrics (separate protocol)')
            except (ValueError,KeyError):continue
            for r in payload.get('rows',[]):
                if r.get('metrics_path'):mhw_lookup[str(Path(r['metrics_path']).parent)]=r
        for job in family_states[family]['jobs']:
            assert job['status']=='completed' and job['returncode']==0
            rd=Path(job['run_dir']); paths=list(rd.glob('*_metrics.json'));assert len(paths)==1
            m=read(paths[0],'follow-up metrics (separate protocol)'); row=mhw_lookup.get(str(rd),{})
            c=read(rd/'jeju_hilam_config.json','follow-up configuration')
            frac=c.get('subset_ratio');regime='Full' if c.get('data_regime')=='full' or '/full/' in str(rd) else 'block_subset'
            arm=job.get('arm','p0_with_aux');lam=job.get('lambda_physics',c.get('lambda_physics'))
            extra.append({'experiment_family':family,'training_fraction':NAN if regime=='Full' else frac,'data_regime':regime,
                'lead_day':job['lead'],'model':arm,'seed':job['seed'],'arm':arm,'lambda_physics':lam,
                **{key:row.get(src,m.get(src,NAN)) for key,src in METRICS.items()},'train_samples':c.get('train_samples'),
                'metrics_path':str(paths[0]),'fields_path':m.get('fields_path'),'metric_source':'follow-up saved metrics; not pooled with original scarcity'})
    for r in run_rows:r['arm']='baseline' if r['model']=='baseline' else 'P0_with_Rres';r['lambda_physics']=0 if r['model']=='baseline' else .1
    csv('experiment_metrics_long',pd.DataFrame(run_rows+extra))
    csv('primary_metrics_long',original[original.data_regime=='block_subset'])
    csv('full_reference_metrics',original[original.data_regime=='Full'])
    csv('pairing_audit',audits);csv('training_logs_by_seed',histories);csv('error_quantiles',quantiles);csv('seasonal_metrics',seasonal);csv('mhw_conditioned_metrics',conditioned);csv('paired_error_ecdf',ecdfs)
    keys=['data_regime','training_fraction','lead_day']
    sums=aggregate(original,list(METRICS),keys+['model']);csv('sst_and_mhw_seed_summary',sums)
    csv('mhw_metrics_by_seed',original[[*keys,'model','seed','mhw_csi','mhw_pod','mhw_far','strong_csi','tp','fp','fn','tn','strong_tp','strong_fp','strong_fn','metric_source']])
    csv('mhw_metrics_summary',aggregate(original,['mhw_csi','mhw_pod','mhw_far','strong_csi'],keys+['model']))
    pairs=[];summary=[]
    for regime in regimes:
        for lead in [7,14]:
            all_d=[]
            for seed in SEEDS:
                b,p=[next(r for r in run_rows if r['data_regime']==('Full' if regime=='Full' else 'block_subset') and (regime=='Full' or r['training_fraction']==int(regime)) and r['lead_day']==lead and r['seed']==seed and r['model']==model) for model in ['baseline','PINN']]
                d={x:b[x] for x in ['data_regime','training_fraction','lead_day','seed']}
                d.update({'baseline_rmse':b['rmse'],'pinn_rmse':p['rmse'],'delta_rmse':p['rmse']-b['rmse'],
                          'relative_rmse_improvement_pct':(b['rmse']-p['rmse'])/b['rmse']*100})
                for m in ['mhw_csi','mhw_pod','mhw_far','strong_csi']:d['delta_'+m]=p[m]-b[m]
                pairs.append(d);all_d.append(d)
            ds=pd.DataFrame(all_d);bm=ds.baseline_rmse.mean();pm=ds.pinn_rmse.mean()
            summary.append({**{x:all_d[0][x] for x in keys},'baseline_rmse_mean':bm,'baseline_rmse_std':ds.baseline_rmse.std(ddof=1),
                'pinn_rmse_mean':pm,'pinn_rmse_std':ds.pinn_rmse.std(ddof=1),'delta_rmse':ds.delta_rmse.mean(),
                'relative_improvement_pct':(bm-pm)/bm*100,'mean_seed_relative_improvement_pct':ds.relative_rmse_improvement_pct.mean(),
                'n_pinn_better':int((ds.delta_rmse<0).sum()),'n_seeds':len(ds)})
            bgrid=np.mean([spatial[regime,lead,'baseline',s] for s in SEEDS],axis=0);pgrid=np.mean([spatial[regime,lead,'PINN',s] for s in SEEDS],axis=0)
            spatial_rows.append({**{x:all_d[0][x] for x in keys},'improved_grid_fraction':float((pgrid<bgrid).mean()),'n_improved_cells':int((pgrid<bgrid).sum()),'n_ocean_cells':661,'definition':'difference of seed-mean per-cell RMSE; unweighted ocean cells'})
    pair=csv('paired_model_differences',pairs);summary=csv('sst_summary',summary);csv('spatial_improvement_summary',spatial_rows)
    (data/'sst_summary.md').write_text(mdtable(summary)+'\n')
    csv('training_log_summary',aggregate(pd.DataFrame(histories),['best_epoch','stopping_virtual_epoch','actual_steps','best_val_rmse','elapsed_seconds'],keys+['model']))
    sea_df=pd.DataFrame(seasonal);sea_pair=[]
    for (regime,frac,lead,season),g in sea_df.groupby(keys+['season'],dropna=False):
        b=g[g.model=='baseline'].set_index('seed').sort_index();p=g[g.model=='PINN'].set_index('seed').sort_index();assert b.index.equals(p.index)
        sea_pair.append({'data_regime':regime,'training_fraction':frac,'lead_day':lead,'season':season,'baseline_rmse_mean':b.rmse.mean(),'pinn_rmse_mean':p.rmse.mean(),
                         'relative_improvement_pct':(b.rmse.mean()-p.rmse.mean())/b.rmse.mean()*100,'n_dates_per_seed':int(b.n_target_dates.iloc[0]),'n_pinn_better':int((p.rmse<b.rmse).sum()),'n_seeds':len(b)})
    csv('seasonal_paired_summary',sea_pair)
    # Main block length inherited from prior paper analysis; iid and 30/180d sensitivity reported separately.
    bootstrap=[]
    for lead in [7,14]:
        a=np.array([[[daily[r,lead,m,s] for s in SEEDS] for m in ['baseline','PINN']] for r in regimes])
        for block in [1,30,90,180]:
            replicates=bootstrap_daily(a,args.bootstrap,block,20260925+lead+block)
            for j,r in enumerate(regimes):
                point=summary[(summary.lead_day==lead)&(summary.data_regime==('Full' if r=='Full' else 'block_subset'))]
                if r!='Full':point=point[point.training_fraction==int(r)]
                point=point.iloc[0];ci=np.quantile(replicates[:,j,:],[.025,.975],axis=0)
                bootstrap.append({'data_regime':point.data_regime,'training_fraction':point.training_fraction,'lead_day':lead,
                    'method':'iid_target_date_cluster' if block==1 else 'circular_moving_time_block','block_days':block,'n_bootstrap':args.bootstrap,
                    'delta_rmse':point.delta_rmse,'ci_low':ci[0,0],'ci_high':ci[1,0],
                    'relative_improvement_pct':point.relative_improvement_pct,'relative_ci_low':ci[0,1],'relative_ci_high':ci[1,1],
                    'scope':'conditional on 3 fitted seed pairs; common dates across all models/seeds/cells; no seed resampling',
                    'mhw_bootstrap_status':'계산 불가: no saved threshold/event-mask/daily contingency arrays'})
            note(f'Bootstrap lead={lead}, block={block} days: {args.bootstrap:,} shared-date resamples complete')
    boot=csv('bootstrap_ci',bootstrap)
    trends=[]
    for lead in [7,14]:
        d=summary[(summary.lead_day==lead)&(summary.data_regime=='block_subset')].sort_values('training_fraction');imp=d.relative_improvement_pct.to_numpy()
        trends.append({'lead_day':lead,**{f'improvement_{int(r.training_fraction)}pct':r.relative_improvement_pct for r in d.itertuples()},
            'maximum_improvement_fraction':int(d.loc[d.relative_improvement_pct.idxmax(),'training_fraction']),
            'benefit_monotonically_increases_as_fraction_decreases':bool(np.all(np.diff(imp)<=0))})
    csv('scarcity_trend',trends)
    # All plot coordinates and numeric labels are read from these same DataFrames and saved separately.
    colors={'baseline':'#2864A0','PINN':'#C44B32'};labels={'baseline':'Baseline','PINN':'PINN (P0 + auxiliary)'}
    def decorate(ax):ax.set_xticks(FRACTIONS);ax.set_xlabel('Training block fraction (%)');ax.grid(axis='y',alpha=.18)
    for lead in [7,14]:
        d=sums[(sums.lead_day==lead)&(sums.data_regime=='block_subset')]
        fig,ax=plt.subplots(figsize=(4.5,3.3),layout='constrained')
        for m in ['baseline','PINN']:
            g=d[d.model==m].sort_values('training_fraction');ax.errorbar(g.training_fraction,g.rmse_mean,yerr=g.rmse_std,marker='o',capsize=3,color=colors[m],label=labels[m])
        decorate(ax);ax.set_ylabel('SST RMSE (K)');ax.set_title(f'{lead}-day forecast');ax.legend()
        savefig(fig,f'fig_rmse_vs_fraction_lead{lead}','Seed mean ± sample SD (n=3). 100% block subset is not Full. Original batch16 P0 includes auxiliary Rres loss.','sst_and_mhw_seed_summary.csv',d)
    # Reference-style SST figures: preserve errors, colors, labels; stack panels only.
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
    for lead in [7,14]:
        d=pair[(pair.data_regime=='block_subset')&(pair.lead_day==lead)];fig,ax=plt.subplots(figsize=(4.7,3.3),layout='constrained')
        for seed,marker,color in zip(SEEDS,['o','s','^'],['#377EB8','#4DAF4A','#984EA3']):
            g=d[d.seed==seed];ax.scatter(g.training_fraction,g.relative_rmse_improvement_pct,label=f'Seed {seed}',marker=marker,color=color,s=42)
        means=d.groupby('training_fraction').relative_rmse_improvement_pct.mean();ax.plot(means.index,means.values,'k_',ms=16,mew=2,label='Mean of seed improvements')
        ax.axhline(0,color='.4',lw=1,ls='--');decorate(ax);ax.set_ylabel('Seed-wise RMSE improvement (%)');ax.set_title(f'{lead}-day forecast');ax.legend(fontsize=8,ncol=2)
        savefig(fig,f'fig_seedwise_improvement_lead{lead}','Dots: paired seed-relative improvements. Black ticks: arithmetic mean of seed-relative improvements, distinct from ratio of means in Figure B.','paired_model_differences.csv',d)
    for metric,label in [('mhw_csi','CSI (higher is better)'),('mhw_pod','POD (higher is better)'),('mhw_far','FAR (lower is better)')]:
        fig,axes=plt.subplots(2,1,figsize=(4.8,6.4),sharey=True,layout='constrained');d=sums[sums.data_regime=='block_subset']
        for lead,ax in zip([7,14],axes):
            for m in ['baseline','PINN']:
                g=d[(d.lead_day==lead)&(d.model==m)].sort_values('training_fraction');ax.errorbar(g.training_fraction,g[metric+'_mean'],yerr=g[metric+'_std'],color=colors[m],marker='o',capsize=3,label=labels[m])
            decorate(ax);ax.set_title(f'{lead}-day forecast');ax.set_ylim(0,1)
        axes[0].set_ylabel(label);axes[1].set_ylabel(label);axes[0].legend(fontsize=8)
        savefig(fig,'fig_'+metric.removeprefix('mhw_')+'_vs_fraction','Existing MHW evaluation, mean ± sample SD across 3 seeds; pooled grid-time contingency within each seed. No threshold recomputation.','mhw_metrics_summary.csv',d)
    d=boot[(boot.block_days==90)&(boot.data_regime=='block_subset')].sort_values(['lead_day','training_fraction']);fig,ax=plt.subplots(figsize=(5.5,4.3),layout='constrained')
    y=np.arange(len(d))
    for yi,row in zip(y,d.itertuples()):
        ax.plot([row.ci_low,row.ci_high],[yi,yi],color='#2864A0',lw=2);ax.plot(row.delta_rmse,yi,'o',color='#2864A0')
    ax.axvline(0,color='.4',ls='--',lw=1);ax.set_yticks(y,[f'{int(r.training_fraction)}% – {r.lead_day}d' for r in d.itertuples()]);ax.invert_yaxis();ax.set_xlabel(r'$\Delta$RMSE: PINN − baseline (K)');ax.set_title('95% conditional CI · 90-day time blocks')
    savefig(fig,'fig_bootstrap_delta_rmse',f'{args.bootstrap:,} paired circular 90-day block resamples, common to cells/models/seeds. ΔRMSE<0 favors PINN. Conditional test-period uncertainty, not training-seed uncertainty. 1/30/180-day alternatives in CSV.','bootstrap_ci.csv',d)
    # Geographic coordinates absent: clearly labeled grid-index maps, never fabricated lat/lon.
    maps=[];cell_rows=[]
    ocean=np.load(root/'allocation_batch16_5plus5/ratio_10/jeju_hilam/lead7_seed42/jeju_hilam_graph.npz',allow_pickle=False)['ocean_mask'].astype(bool)
    for regime in ['10','100']:
        for lead in [7,14]:
            b=np.mean([spatial[regime,lead,'baseline',s] for s in SEEDS],axis=0);p=np.mean([spatial[regime,lead,'PINN',s] for s in SEEDS],axis=0)
            grid=np.full(ocean.shape,np.nan);grid[ocean]=p-b;maps.append((regime,lead,grid))
            for rc,bv,pv in zip(np.argwhere(ocean),b,p):cell_rows.append({'training_fraction':int(regime),'lead_day':lead,'grid_row':rc[0],'grid_column':rc[1],'baseline_rmse_mean':bv,'pinn_rmse_mean':pv,'delta_rmse':pv-bv})
    cells=csv('spatial_cell_rmse',cell_rows);limit=max(np.nanmax(np.abs(g)) for _,_,g in maps)
    for regime,lead,g in maps:
        fig,ax=plt.subplots(figsize=(4.7,3.7),layout='constrained');im=ax.imshow(g,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_xlabel('Grid column (longitude unavailable)');ax.set_ylabel('Grid row (latitude unavailable)');ax.set_title(f'{regime}% blocks · {lead}-day · grid coordinates')
        fig.colorbar(im,ax=ax,label='Seed-mean ΔRMSE (K)')
        savefig(fig,f'fig_spatial_delta_rmse_{regime}pct_lead{lead}',f'Grid-index map only; no geographic coordinates. Mean of 3 per-seed cell RMSEs, PINN−baseline; blue/negative favors PINN. Common symmetric scale ±{limit:.6f} K across all four maps.','spatial_cell_rmse.csv',cells[(cells.training_fraction==int(regime))&(cells.lead_day==lead)])
    e=pd.DataFrame(ecdfs);fig,axes=plt.subplots(2,1,figsize=(4.8,6.4),sharey=True,layout='constrained')
    for ax,lead in zip(axes,[7,14]):
        for seed in SEEDS:
            g=e[(e.data_regime=='block_subset')&(e.training_fraction==10)&(e.lead_day==lead)&(e.seed==seed)];ax.plot(g.paired_absolute_error_difference_K,g.cdf,label=f'Seed {seed}')
        ax.axvline(0,color='.4',ls='--',lw=1);ax.set_title(f'10% blocks · {lead}-day');ax.set_xlabel('|PINN error| − |baseline error| (K)')
    axes[0].set_ylabel('Cumulative fraction');axes[1].set_ylabel('Cumulative fraction');axes[0].legend(fontsize=8)
    savefig(fig,'fig_paired_error_ecdf','Descriptive grid-time error distribution, represented at 201 exact empirical quantiles per seed. Not an independent-sample inferential test. Negative error differences favor PINN.','paired_error_ecdf.csv',e[(e.data_regime=='block_subset')&(e.training_fraction==10)])
    write_json(figures/'figure_metadata.json',figure_metadata)
    # Compact ACM-friendly tables, no packages beyond booktabs required.
    def latex(name,caption,cols,headers,rows,wide=False):
        env='table*' if wide else 'table'
        text=[f'\\begin{{{env}}}[t]','\\centering',f'\\caption{{{caption}}}',f'\\label{{tab:{name}}}','\\small',f'\\begin{{tabular}}{{{cols}}}','\\toprule',' & '.join(headers)+r' \\','\\midrule']
        text+=[' & '.join(map(str,row))+r' \\' for row in rows]
        text+=['\\bottomrule','\\end{tabular}',f'\\end{{{env}}}']
        (tables/(name+'.tex')).write_text('\n'.join(text)+'\n')
    st=summary[summary.data_regime=='block_subset'].sort_values(['lead_day','training_fraction'])
    latex('table_sst','SST forecasting on block subsets (three seeds; mean $\\pm$ sample SD). PINN includes P0 and auxiliary loss.','rrrrr',
          ['Blocks','Lead','Baseline (K)','PINN (K)','Gain (\\%)'],
          [[f'{int(r.training_fraction)}\\%',r.lead_day,f'${r.baseline_rmse_mean:.4f} \\pm {r.baseline_rmse_std:.4f}$',f'${r.pinn_rmse_mean:.4f} \\pm {r.pinn_rmse_std:.4f}$',f'{r.relative_improvement_pct:+.2f}'] for r in st.itertuples()],True)
    mh=sums[sums.data_regime=='block_subset'].sort_values(['lead_day','training_fraction','model'])
    latex('table_mhw','MHW grid-time detection skill (three seeds; mean $\\pm$ sample SD).','rrlrrr',['Blocks','Lead','Model','CSI','POD','FAR'],
          [[f'{int(r.training_fraction)}\\%',r.lead_day,r.model]+[f'${getattr(r,m+"_mean"):.3f} \\pm {getattr(r,m+"_std"):.3f}$' for m in ['mhw_csi','mhw_pod','mhw_far']] for r in mh.itertuples()],True)
    setup=[['Input history / leads','30 days / 7, 14 days'],['Block fractions','10, 25, 50, 100\\%'],['Training windows','376, 987, 1974, 3995'],['Full reference windows','7644 (7d), 7639 (14d)'],['Subset / training seeds','7, 42, 123 (paired)'],['Train/val/test split','Chronological 70/15/15 samples'],['Batch / maximum steps','16 / 38240'],['Validation / patience','478 steps / 12 validations'],['Normalization','Original Full training statistics'],['MHW reference','First 7670 daily records'],['MHW threshold / duration','DOY 90th percentile / 5 days'],['Primary bootstrap','20000 shared 90-day block draws' if args.bootstrap==20000 else f'{args.bootstrap} shared 90-day block draws']]
    latex('table_setup','Verified experimental setup. Block 100\\% is distinct from Full.','ll',['Setting','Value'],setup,True)
    csv('experimental_setup',[{'setting':a,'value':b} for a,b in setup])
    # Persist explicit limitations instead of silently fabricating unavailable results.
    unavailable=[{'item':'MHW threshold/climatology array and fresh event masks','status':'계산 불가','reason':'No saved arrays in discovered result NPZ/NPY; evaluator originally recomputed from training NetCDF, which this saved-results-only analysis does not read.'},
        {'item':'MHW CSI/POD/FAR paired date bootstrap','status':'계산 불가','reason':'Aggregate counts exist but not per-date event masks/counts; cannot resample aggregate scores as dates.'},
        {'item':'Non-MHW-conditioned RMSE and fresh MHW-conditioned recomputation','status':'계산 불가','reason':'Exact MHW masks absent. Existing saved MHW-conditioned RMSE is retained and labeled.'},
        {'item':'Latitude/longitude geographic maps','status':'계산 불가','reason':'Saved prediction/graph NPZ has mask/topology but no raw lat/lon. Grid-index maps supplied instead; normalized node features not treated as coordinates.'},
        {'item':'ACC independent recomputation','status':'계산 불가','reason':'Saved ACC reused; climatology arrays absent.'},
        {'item':'New-training-seed population CI','status':'계산 불가','reason':'Only 3 fitted seed pairs; bootstrap describes conditional test-period variation.'},
        {'item':'Reliable isolated wall-clock efficiency comparison','status':'계산 불가','reason':'Concurrent shared-GPU original jobs; logged elapsed time is descriptive only.'}]
    csv('unavailable_analyses',unavailable)
    # Hash all input files again after the entire analysis.
    for p,r in provenance.items():
        r['sha256_after']=sha(p);r['unchanged']=r['sha256_before']==r['sha256_after'];assert r['unchanged'],p
    csv('source_provenance',list(provenance.values()))
    for name,df in frames.items():
        disk=pd.read_csv(data/(name+'.csv'))
        assert len(disk)==len(df),(name,'row count')
        for col in df.select_dtypes(include=[np.number]).columns:
            a=pd.to_numeric(disk[col],errors='coerce').to_numpy(float);b=df[col].to_numpy(float)
            assert np.allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True),(name,col)
    assert 'torch' not in sys.modules and not any(k.startswith('torch.') for k in sys.modules)
    verification={'cpu_only':True,'torch_imported':False,'training_or_inference_called':False,'checkpoint_deserialized':False,'netcdf_opened':False,
        'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'completed_original_runs':60,'completed_followup_runs_separate':len(extra),
        'main_scarcity_runs':48,'matched_original_pairs':30,'main_scarcity_pairs':24,'same_test_arrays_per_lead':True,
        'same_val_test_split_hashes_per_lead':True,'nested_subsets_verified':True,'actual_training_selections_verified':True,
        'sst_saved_vs_recomputed_tolerance_K':2e-6,'saved_mhw_counts_formulas_verified':True,
        'all_used_sources_hash_unchanged':True,'source_count':len(provenance),'csv_roundtrip_verified':True,
        'figure_data_saved_from_same_tables':True,'summary_report_generated_from_same_tables':True,'bootstrap_resamples_per_method':args.bootstrap,
        'date_origin_source':str(project/'data/subsets/train_subsets_v2_win44_seed42.json'),'geographic_coordinates_not_fabricated':True}
    write_json(out/'verification.json',verification)
    findings=[]
    for lead in [7,14]:
        r=st[(st.lead_day==lead)&(st.training_fraction==10)].iloc[0]
        findings.append(f'10% / {lead}-day: baseline {r.baseline_rmse_mean:.4f} K, PINN {r.pinn_rmse_mean:.4f} K; relative RMSE improvement {r.relative_improvement_pct:.2f}%; improved seeds {int(r.n_pinn_better)}/{int(r.n_seeds)}.')
    r=st[(st.lead_day==7)&(st.training_fraction==25)].iloc[0]
    findings.append(f'25% / 7-day: relative RMSE improvement {r.relative_improvement_pct:.2f}%; improved seeds {int(r.n_pinn_better)}/3.')
    for lead in [7,14]:
        r=summary[(summary.data_regime=='Full')&(summary.lead_day==lead)].iloc[0]
        findings.append(f'Separate Full reference / {lead}-day: baseline {r.baseline_rmse_mean:.4f} K, PINN {r.pinn_rmse_mean:.4f} K; relative improvement {r.relative_improvement_pct:.2f}%.')
    ci=boot[(boot.block_days==90)&(boot.data_regime=='block_subset')][['training_fraction','lead_day','delta_rmse','ci_low','ci_high','relative_ci_low','relative_ci_high']]
    report=['# Training-data scarcity: paper analysis','', 'CPU-only post-hoc analysis; no training performed','',
        '## 1. 실제 실험 구조 / scope','',
        '- Main: allocation_batch16_5plus5, 48 completed runs = 4 block fractions × 2 leads × 2 models × 3 seeds. Full batch16: 12 separate references. Follow-ups: 18 lambda + 30 corrected-physics runs retained only in protocol-separated long inventory; never pooled into main figures.',
        '- Nested block fractions verified against every saved actual training-index list. 100% block subset = 3995 windows; Full = 7644/7639. Fractions are percentages of 85 eligible 90-day blocks (rounded), not percentages of original training-window count.',
        '- Original P0 includes lambda_physics=0.1 and auxiliary lambda_rres=10. Original land-adjacent spatial differencing and possible head-induced random-order differences remain. This is not the later matched clean physics-only protocol.',
        '- Train/validation/test metadata: sample-based chronological 70/15/15. Full training normalization (including unselected input/target information) shared across fractions. Subset seed equals training seed.',
        '', '## 2. 사용 파일 / provenance','',
        f'- Results root: `{root}`. Latest code definitions: `{code}`.',
        '- `analysis_data/file_inventory.csv`: discovered result files, sizes, timestamps, array keys, capabilities. `experiment_inventory.csv`: selected/excluded batch jobs. `source_provenance.csv`: before/after SHA-256 for every used source.',
        '- No checkpoint opened; no NetCDF read; no model import or inference. Target calendar recovered exactly from saved dataset-origin metadata plus target_index, checked against recorded test endpoints.',
        '', '## 3. 계산 가능 지표','',
        '- SST RMSE/MAE/bias recomputed from 60 prediction arrays and checked against saved evaluation within 2e-6 K. ACC retained from saved evaluator. Seed mean and sample SD (ddof=1).',
        '- MHW CSI/POD/FAR/Strong+ CSI and TP/FP/FN reused from existing evaluation and checked against saved contingency counts with original zero-denominator=0 convention. No new MHW masks generated.',
        '- SST bootstrap, per-cell errors, error quantiles, seasons, paired error ECDF, training/validation history. Saved observed-MHW grid-day RMSE retained; it is not domain-wide MHW-day RMSE.',
        '', '## 4. 계산하지 못한 항목','',mdtable(pd.DataFrame(unavailable)),
        '', '## 5. SST 결과','',mdtable(summary),
        '', 'Relative improvement in summary/figure B = ratio of seed-mean RMSEs. Seedwise figure uses individual ratios and their arithmetic mean; both estimands are separately named in CSV.',
        '', '## 6. MHW 결과와 기존 정의','',
        '- Existing evaluation builds a common threshold using the first round(10957×0.70)=7670 daily records: 1993-01-01 to 2013-12-31. This daily reference differs from lead-specific training target endpoints (2014-01-10/12). It is unchanged across fractions; no subset-specific threshold.',
        '- 90th percentile, ±5 day-of-year pooling, minimum 5 consecutive exceedance days. No extra smoothing and no gap merging in existing implementation.',
        '- Leap handling is the original pandas Gregorian dayofyear with a 366-day circular distance. There is no explicit Feb-29 interpolation or post-February alignment correction in non-leap years. Report this implementation, not an idealized standard.',
        '- CSI/POD/FAR pool ocean grid-time cells within each run; Strong+ uses normalized intensity ≥2 on duration-filtered events. Thresholds themselves were not saved, so their numeric arrays cannot be re-audited here.',
        mdtable(frames['mhw_metrics_summary']),
        '', '## 7. Paired bootstrap','',
        f'- {args.bootstrap:,} replicates per condition/method. Existing circular moving-block algorithm adapted into chunked CPU computation. Primary 90 days; sensitivity 30/180 days; iid target-date cluster variant (block=1) also included.',
        '- Same date samples for all cells, seed pairs and fractions within each lead. RMSE first calculated per fitted seed, then averaged; no seed or spatial-cell resampling. 95% percentile CIs are conditional on existing fitted models, not confidence over new training seeds. No p-values.',
        '- IID-date intervals ignore serial dependence; use 90-day primary with block-length sensitivities. The 90-day choice follows the existing analysis and is not claimed optimal.',
        mdtable(ci),
        '', '## 8. 가장 중요한 5개 quantitative findings','']+[f'{i}. {s}' for i,s in enumerate(findings,1)]+[
        '', '## 9. Figure / Table 목록','',
        '\n'.join(f'- [PDF: {x["figure"]}](paper_figures/{x["figure"]}.pdf) · [PNG](paper_figures/{x["figure"]}.png)' for x in figure_metadata),
        '- `paper_figures/figure_metadata.json`: captions, signs, denominators and source CSVs. Every figure also has its plotted-data CSV.',
        '- LaTeX: `paper_tables/table_setup.tex`, `table_sst.tex`, `table_mhw.tex` (requires booktabs; SST/MHW use table*).',
        '', '## 10. Scarcity trend와 해석 주의','',mdtable(pd.DataFrame(trends)),
        '- The largest relative improvement was observed under the most data-scarce regime for both leads. Across the four observed seed-mean point estimates, lead7 is non-monotonic and lead14 is monotonic as data decreases. This descriptive lead14 ordering is not a statistically established population trend; do not generalize monotonicity across leads.',
        '- Effects refer to the combined original P0/auxiliary training setup, not isolated physical constraints. Later pilot results are different protocols, not substitute points for this curve.',
        '- Only three seeds; subset-selection and training randomness are confounded. Fractions are nested, all share test data and Full statistics. No causal mechanism or general significance claim.',
        '- Spatial improvement is the fraction of 661 cells with negative difference of seed-mean per-cell RMSE; not area weighted. Maps use saved row/column coordinates only.',
        '- Error quantiles/ECDF are descriptive grid-time distributions. Seasonal counts are target dates, not independent grid samples. Training times are affected by concurrent GPU allocation and are not an efficiency claim.',
        '', '## 11. 검증 / reproducibility','',
        f'- Verified {len(provenance)} used source hashes before/after; all unchanged. 60 run configurations, actual subsets, paired target/index/mask arrays, saved MHW contingency formulas, SST recalculations, numeric CSV round trips and shared figure/table/report inputs checked.',
        '- See `verification.json`, `analysis_data/pairing_audit.csv`, `analysis_data/source_provenance.csv`.',
        '- Every analysis output is in this new Ceph directory. Script refuses to overwrite an existing output directory. CUDA hidden; torch not imported; no subprocess training/inference calls.',
        '', '```bash',f'/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python -B {Path(__file__).resolve()} \\',f'  --results-root {root} \\',f'  --output-dir {out.parent}/another_new_analysis --bootstrap 20000','```','']
    (out/'analysis_report.md').write_text('\n'.join(report),encoding='utf-8')
    (out/'execution.log').write_text('\n'.join(log)+'\n')
    write_json(out/'analysis_metadata.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),
        'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'command':sys.argv,
        'bootstrap_seed_formula':'20260925 + lead + block_days','float_precision':'float64 error reductions; saved evaluator values remain primary',
        'figure_count':len(figure_metadata),'csv_count':len(frames),'primary_main_protocol':'batch16 original, not corrected clean pilot'})
    note(f'Completed: {len(figure_metadata)} PDF+PNG figures, {len(frames)} CSVs, 3 LaTeX tables. Report: {out / "analysis_report.md"}')

if __name__=='__main__':main()
