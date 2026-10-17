"""Final artifact and preservation checks; no model or dataset writes."""
import json,hashlib,datetime,zoneinfo
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v3_generalization_20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 protocol=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'results.json').read_text());assert r['status']=='COMPLETE'
 assert len(r['rows'])==12 and len(r['bootstrap'])==60
 assert json.loads((OUT/'inference_status.json').read_text())['status']=='COMPLETE'
 audit=json.loads((OUT/'checkpoint_audit.json').read_text());assert audit['all_12_validation_gate'] and len(audit['rows'])==12
 assert json.loads((OUT/'analysis_tests.json').read_text())['status']=='PASS'
 assert json.loads((OUT/'mhw_reference_audit.json').read_text())['status']=='PASS'
 verified=[]
 for row in protocol['rows']:
  assert sha(row['checkpoint'])==row['checkpoint_sha256']
  assert sha(row['manifest'])==row['manifest_sha256']
  assert sha(row['config_file'])==row['config_file_sha256']
  d=OUT/'runs'/f"{row['arm']}_seed{row['seed']}"
  for split,n in [('train',376),('val',1638),('test',1639)]:
   m=json.loads((d/(split+'_metrics.json')).read_text())
   with np.load(d/(split+'_fields.npz')) as f:
    assert len(f['target_index'])==n and int(f['seed'])==row['seed'] and str(f['model'])==row['arm']
    assert np.all(f['daily_count']==661)
    sea=f['mask'].astype(bool);e=f['pred_raw'].astype(float)-f['target_raw'].astype(float)
    assert np.isfinite(e[sea]).all()
    assert abs(np.sqrt(np.mean(e[sea]**2))-m['rmse'])<1e-12
    np.testing.assert_allclose(np.where(sea,e**2,0).reshape(n,-1).sum(1),f['daily_sse'],rtol=0,atol=1e-10)
  with np.load(d/'test_mhw_fields.npz') as f:assert f['pred_mhw'].dtype==bool and f['obs_mhw'].dtype==bool
  verified.append(dict(model=row['arm'],seed=row['seed'],checkpoint_manifest_config_preserved=True,raw_fields_metrics_consistent=True))
 for stem in ['figure1_validation_test','figure2_seed_improvement','figure3_bootstrap','figure4_generalization']:
  for ext in ['png','pdf']:assert (OUT/f'{stem}.{ext}').stat().st_size>1000
 for name,h in protocol['code_sha256'].items():assert sha(ROOT/name)==h
 extra=json.loads((OUT/'analysis_source_hashes.json').read_text())
 for name,h in extra.items():assert sha(ROOT/name)==h
 sources=[Path('scripts')/name for name in ['evaluate_v3_generalization.py','analyze_v3_generalization.py','plan_v3_generalization_extension.py','compare_v3_generalization_prior.py','verify_v3_generalization.py']]
 final=dict(status='PASS',timestamp=datetime.datetime.now(zoneinfo.ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST'),rows=verified,checkpoints=12,split_field_sets=36,bootstrap_rows=60,figures=4,source_sha256={str(p):sha(ROOT/p) for p in sources},new_training=False,test_based_reselection=False)
 (OUT/'final_integrity.json').write_text(json.dumps(final,indent=2)+'\n')
 groups={g['model']:g for g in r['groups']}
 with (ROOT/'docs/research_log.md').open('a') as f:
  f.write(f"\n- {final['timestamp']} — v3 일반화 평가 완료:12 Validation-best checkpoint의 Train/Val/Test 고정추론,36개필드/12개MHW/90일4000회부트스트랩(30/180민감도),4그림,6RQ보고 및후속계획 저장. 신규학습/데이터변경/Test재선택없음. 무결성PASS. Test RMSE baseline={groups['baseline']['test_rmse_mean']:.6f}, physics_only={groups['physics_only']['test_rmse_mean']:.6f}, auxiliary_only={groups['auxiliary_only']['test_rmse_mean']:.6f}, full={groups['full_pinn']['test_rmse_mean']:.6f}. 경로 outputs/v3_generalization_20261009/report.md; 평가/분석/계획/비교/검증 스크립트 sourcehash final_integrity.json. 탐색적평가로한정.\n")
 print(json.dumps(final,indent=2))
if __name__=='__main__':main()
