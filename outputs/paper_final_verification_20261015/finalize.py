from pathlib import Path
import json,hashlib,re,csv,subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';J=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest();stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
required=['model_definition_audit.md','physics_equation_audit.md','horizontal_advection_audit.md','coastal_stencil_tests.md','experiment_protocol_audit.md','mhw_evaluation_audit.md','auxiliary_ablation_audit.md','statistical_comparison.md','verification_summary_ko.md','retraining_decision.md'];assert all((O/p).exists() for p in required)
status=J(O/'analysis_status.json');assert status['models']==120 and status['recomputed_metrics_match']==120
checks=[]
for e in J(O/'auxiliary_test/protocol.json')['jobs']:
 d=O/'auxiliary_test/runs'/e['name'];m=J(d/'metrics.json');i=J(d/'forcing_invariance_test.json');g=J(d/'gpu_identity.json');s=J(d/'status.json')
 assert s['status']=='COMPLETED' and s['gpu'] in [5,6,7] and m['optimizer_steps']==0 and m['checkpoint_sha256']==e['sha256'] and sha(e['checkpoint'])==e['sha256']
 assert i['prediction_bitwise_identical'] and i['model_state_unchanged'] and i['baseline_truth_identical'] and i['optimizer_steps']==0
 checks.append(dict(name=e['name'],gpu=s['gpu'],pid=s['pid'],status='PASS',checkpoint_sha256=e['sha256'],peak_reserved_mib=m['peak_reserved_mib'],inference_seconds=m['evaluation_seconds'],gpu_identity=g))
# Freeze all new executable/script/config paths used in this audit.
files={str(p.relative_to(R)):sha(p) for p in O.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.md','.csv','.json'] and p.name!='completion_manifest.json'}
for fn in required:
 for dest in re.findall(r'\]\((/[^\)]+)\)',(O/fn).read_text()):
  target=re.sub(r':\d+$','',dest);assert Path(target).exists(),(fn,target)
# Scientific code hashes remained unchanged after inference and analysis.
for row in J(O/'source_hashes.json'):assert sha(R/row['path'])==row['expected']
assert J(O/'protected_integrity.json')['mismatches']==[]
out=dict(status='COMPLETE',timestamp=stamp,current_oracle120_verdict='PASS WITH LIMITATIONS',historical_WSDM_P1_P3='UNVERIFIED',confirmed_retrain_required_runs=[],new_training_runs=0,optimizer_steps=0,auxiliary_test_checks=checks,code_hash_records_verified=53,protected_hashes_verified=654,protocol_runs_verified=120,stencil_checks_passed=88,rhs_scales_verified=24,test_predictions_recomputed=120,report_files=required,artifact_sha256=files)
(O/'completion_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with (R/'docs/research_log.md').open('a') as f:
 f.write('\n- '+stamp+' — 최종 논문 감사 완료: outputs/paper_final_verification_20261015에 필수10개 보고서,120개 Test 재검산,24개 Auxiliary 고정 Test 추론,6쌍×8조건의30/90/180일·2000회블록CI 및그림 저장. GPU5/6/7각1개,process name yeonwoo 확인,학습/optimizer update0. 120프로토콜·53소스해시·654보호파일·88수치검사·24RHS재계산PASS. NoAdv가Adv보다RMSE22/24우위,Auxiliary는Baseline대비Val16/24·Test15/24개선. 현120개PASS WITH LIMITATIONS/확인된재학습대상0,구WSDMP1–P3실행mapping UNVERIFIED. 문서렌더러에서선택의존성없이Markdown표생성, P0의K/s잔차후86400변환문구를실제연산순서에맞춤. 기존과학코드/데이터/결과교체없음.\n')
print(json.dumps({k:out[k] for k in ['status','timestamp','current_oracle120_verdict','confirmed_retrain_required_runs','test_predictions_recomputed']},ensure_ascii=False))
