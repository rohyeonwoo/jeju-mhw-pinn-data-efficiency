"""Inventory all saved configs/checkpoints, strictly audit reuse, freeze48 conditions."""
import os,json,csv,math,copy
from pathlib import Path
import train_v3_study as w
ROOT=w.ROOT;OUT=ROOT/'outputs/v3_physics_ratio_lead_lambda001_20261009'
torch=w.torch;np=w.np
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  c=csv.DictWriter(f,keys);c.writeheader();c.writerows(rows)
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
 for name in ['runtime','logs','evaluations','figures']: (OUT/name).mkdir()
 ref=json.loads((ROOT/'outputs/aux_weight_selection_20261009/runs/lead7_seed42/baseline_aux0/config.json').read_text())
 index=json.loads((ROOT/'data/subsets/preprocessing_subset_v3/index.json').read_text());manifests={};hashes={};manifestrows=[]
 def checked(path,expected):
  path=Path(path).resolve();key=str(path)
  if key not in hashes:hashes[key]=w.sha(path)
  assert hashes[key]==expected,key
 for item in index['datasets']:
  path=ROOT/'data/subsets/preprocessing_subset_v3'/item['manifest'];checked(path,item['sha256']);m=json.loads(path.read_text());assert m['schema']=='subset_normalization_v3'
  for key,h in [('data_path','data_sha256'),('raw_thetao_path','raw_thetao_sha256'),('subset_path','subset_sha256')]:checked(m[key],m[h])
  checked(path.parent/m['climatology_file'],m['climatology_sha256']);checked(m['parent_manifest'],m['parent_manifest_sha256'])
  for name,h in m['source_sha256'].items():checked(ROOT/name,h)
  parent=json.loads(Path(m['parent_manifest']).read_text())
  for k in ['statistics','climatology_sha256','input_time_indices','train_start_indices','val_start_indices','test_start_indices']:assert m[k]==parent[k],k
  assert len(m['train_start_indices'])=={10:376,25:987,50:1974,100:3995}[item['ratio']]
  assert math.isfinite(m['physics_rhs_scale']) and m['physics_rhs_scale']>0
  key=(item['ratio'],item['lead'],item['seed']);manifests[key]=(path,m)
  manifestrows.append(dict(ratio=key[0],lead=key[1],seed=key[2],manifest=str(path),n_train=len(m['train_start_indices']),unique_input_dates=len(m['input_time_indices']),rhs_scale=m['physics_rhs_scale'],target_std=m['statistics']['target'][1],v2_inputs_identical=True))
 for lead in [7,14]:
  reference=None
  for seed in [42,7,123]:
   prev=set()
   for ratio in [10,25,50,100]:
    m=manifests[ratio,lead,seed][1];starts=set(m['train_start_indices']);assert prev<=starts;prev=starts
    assert m['train_start_indices']==manifests[ratio,21-lead,seed][1]['train_start_indices']
    sig=(m['val_start_indices'],m['test_start_indices'])
    if reference is None:reference=sig
    else:assert reference==sig
 allconfigs=list((ROOT/'outputs').rglob('config.json'));allcheckpoints=list((ROOT/'outputs').rglob('*.pt'))
 csvsave(OUT/'checkpoint_inventory.csv',[dict(path=str(p),bytes=p.stat().st_size) for p in allcheckpoints])
 protocols={}
 for pp in (ROOT/'outputs').rglob('protocol.json'):
  try:
   q=json.loads(pp.read_text());protocols[w.digest(q)]=(pp,q)
  except (ValueError,OSError):pass
 allowed={'subset_manifest','output_root','lead','seed','lambda_physics','physics_rhs_scale'}
 audits=[];eligible={};state_shapes=None
 for cp in allconfigs:
  c=json.loads(cp.read_text());record=dict(config=str(cp),eligible=False,reason='not compatible native completed config')
  if not all(k in c for k in ['settings','job','parameters','config_sha256','protocol_sha256']):audits.append(record);continue
  s=c['settings'];j=c['job'];weight=s.get('lambda_physics');key=(j.get('ratio'),j.get('lead'),j.get('seed'))
  if key not in manifests or weight not in [0,.01] or s.get('lambda_rres')!=0:
   record['reason']='ratio/lead/seed or active loss weights outside fixed design';audits.append(record);continue
  if c['parameters']!=ref['parameters']:
   record['reason']='different predictor/head structure';audits.append(record);continue
  diffs={k:[ref['settings'].get(k),s.get(k)] for k in set(ref['settings'])|set(s) if k not in allowed and ref['settings'].get(k)!=s.get(k)}
  if diffs:record.update(reason='training settings differ',differences=diffs);audits.append(record);continue
  if not (cp.parent/'status.json').exists():audits.append(record);continue
  status=json.loads((cp.parent/'status.json').read_text())
  if status.get('status')!='trained':record['reason']='not completed native training';audits.append(record);continue
  try:
   assert c['protocol_sha256'] in protocols
   pp,prot=protocols[c['protocol_sha256']]
   for name,h in prot['code_sha256'].items():checked(ROOT/name,h)
   assert w.digest({k:v for k,v in c.items() if k!='config_sha256'})==c['config_sha256']
   mp,m=manifests[key];assert c['manifest_sha256']==w.sha(mp) and c['stats']==m['statistics'] and c['applied_rhs_scale']==m['physics_rhs_scale']
   olddir=ROOT/f'outputs/data_efficiency/v3_baseline_p0_20261008_r1/runs/ratio_{key[0]}/lead{key[1]}_seed{key[2]}/p0'
   rc=json.loads((olddir/'config.json').read_text())
   for k in ['parameters','shared_initial_sha256','split_sha256','tensor_fingerprints','physics_stencil']:assert c[k]==rc[k],k
   h=json.loads((cp.parent/'history.json').read_text());rh=json.loads((olddir/'history.json').read_text());best=min(h,key=lambda x:x['val_rmse'])
   assert len(h)==status['epochs'] and [x['steps'] for x in h]==[478*(i+1) for i in range(len(h))]
   assert status['steps']==478*len(h) and len(h)<=80
   assert len(h)==80 or len(h)-best['epoch']==12
   assert all(x['batch_order_sha256']==y['batch_order_sha256'] for x,y in zip(h,rh))
   assert all(x['rres_calls']==0 and ((x['physics_calls']==0) if weight==0 else x['physics_calls']>0) for x in h)
   state=torch.load(cp.parent/'best.pt',map_location='cpu',weights_only=False)
   assert state['config_sha256']==c['config_sha256'] and state['manifest_sha256']==c['manifest_sha256']
   assert state['best_epoch']==best['epoch']==status['best_epoch'] and state['best_value']==best['val_rmse']==status['best_val_rmse']
   assert {k:list(v) for k,v in state['stats'].items()}==c['stats']
   shapes={k:(list(v.shape),str(v.dtype)) for k,v in state['model_state'].items()}
   if state_shapes is None:state_shapes=shapes
   assert shapes==state_shapes and all(torch.isfinite(v).all() for v in state['model_state'].values())
   last=torch.load(cp.parent/'last.pt',map_location='cpu',weights_only=False)
   assert last['epoch']==len(h) and last['stream']['total']==status['steps']
   assert all(tuple(g['betas'])==(.9,.95) and g['weight_decay']==.1 for g in last['optimizer_state']['param_groups'])
   assert all(k in last for k in ['rng_cpu','rng_cuda','rng_numpy','rng_python','optimizer_state','stream'])
   record.update(eligible=True,reason='all12 reuse criteria verified; strict state/config/source/manifest/optimizer/history/batch audit',checkpoint=str(cp.parent/'best.pt'),checkpoint_sha256=w.sha(cp.parent/'best.pt'),config_sha256=c['config_sha256'],best_update=best['steps'],end_update=status['steps'],validation_rmse=best['val_rmse'])
   ekey=(*key,'baseline' if weight==0 else 'physics_only')
   eligible.setdefault(ekey,[]).append(record)
   del state,last
  except Exception as ex:record['reason']='reuse integrity failed: '+repr(ex)
  audits.append(record)
 jobs=[]
 for ratio in [10,25,50,100]:
  for lead in [7,14]:
   for seed in [42,7,123]:
    for arm in ['baseline','physics_only']:
     key=(ratio,lead,seed);mp,m=manifests[key];matches=eligible.get((*key,arm),[])
     # Multiple compatible snapshots must be identical; never choose on Test.
     if len(matches)>1:assert len({x['checkpoint_sha256'] for x in matches})==1,matches
     reuse=matches[0] if matches else None;name=f'ratio_{ratio}/lead{lead}_seed{seed}/{arm}';newdir=OUT/'runs'/name
     run_dir=Path(reuse['config']).parent if reuse else newdir
     jobs.append(dict(id=len(jobs),name=name,ratio=ratio,lead=lead,seed=seed,arm=arm,lambda_physics=0. if arm=='baseline' else .01,lambda_rres=0.,manifest=str(mp),manifest_sha256=w.sha(mp),run_dir=str(run_dir),reuse=bool(reuse),reuse_evidence=reuse,reference_p0_run=str(ROOT/f'outputs/data_efficiency/v3_baseline_p0_20261008_r1/runs/ratio_{ratio}/lead{lead}_seed{seed}/p0'),evaluation_dir=str(OUT/'evaluations'/name)))
 core=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py'))+['scripts/train_v3_study.py','scripts/train_v3_ablation.py','scripts/ratio001_worker.py']
 protocol=dict(created_at=w.stamp(),experiment='fixed_lambda001_ratio_lead',jobs=jobs,code_sha256={n:w.sha(ROOT/n) for n in core},artifact_hashes=hashes,settings=dict(lambda_physics=.01,lambda_rres=0,epochs=80,updates_per_validation=478,patience=12,batch_size=16,optimizer='AdamW',betas=[.9,.95],weight_decay=.1,gradient_clip=1.,precision='fp32'),resources=dict(gpu_ids=[1,2,3],max_total_per_gpu=4,reserve_mib=4096,memory_multiplier=1.2),evaluation=dict(scope='exploratory replication on previously inspected Test',test_after_all48_checkpoint_gate=True,bootstrap_primary=90,bootstrap_sensitivity=[30,180],bootstrap_repetitions=4000,random_seed_rule='20260925+lead+block',mhw_reference='unchanged prior v3 first70% raw dates,window5,q.9,duration5,area.15,onset3'),reuse_count=sum(j['reuse'] for j in jobs),new_count=sum(not j['reuse'] for j in jobs))
 w.save(OUT/'protocol.json',protocol);w.save(OUT/'reuse_audit.json',dict(configs_scanned=len(allconfigs),checkpoints_scanned=len(allcheckpoints),records=audits));w.save(OUT/'manifest_audit.json',dict(status='PASS',rows=manifestrows,hashes=hashes))
 csvsave(OUT/'run_manifest.csv',[dict(ratio=j['ratio'],lead=j['lead'],seed=j['seed'],model=j['arm'],reuse_new='Reuse' if j['reuse'] else 'New',checkpoint=str(Path(j['run_dir'])/'best.pt'),config=str(Path(j['run_dir'])/'config.json'),evidence=j['reuse_evidence']['reason'] if j['reuse'] else 'no eligible existing checkpoint',status='reused_verified' if j['reuse'] else 'preflight_pending') for j in jobs])
 (OUT/'reuse_audit.md').write_text(f"# Reuse audit\n\nScanned {len(allconfigs)} config files and {len(allcheckpoints)} checkpoints in outputs. Strict audit accepted {protocol['reuse_count']} of48; new training required {protocol['new_count']}. Previous9/39 estimate {'confirmed' if protocol['reuse_count']==9 else 'changed'}. See reuse_audit.json for every candidate and exclusion; run_manifest.csv enumerates48. No selection used Test results. Headless Baselines, dense-validation checkpoints, λ=.1 and auxiliary-active runs excluded.\n")
 w.log(f"Fixedλ=.01 ratio/lead:reuse audit {protocol['reuse_count']}/48, new{protocol['new_count']}; all48manifest prepared; GPU1/2/3 only; no Test before global48checkpoint gate.")
 print(json.dumps(dict(status='REUSE_AUDIT_COMPLETE',reused=protocol['reuse_count'],new=protocol['new_count'])),flush=True)
if __name__=='__main__':main()
