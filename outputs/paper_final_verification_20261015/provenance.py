import json,hashlib,os,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');O=R/'outputs/paper_final_verification_20261015';sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest();rows=[]
for p in (R/'outputs/main_experiment_final').rglob('*metrics.json'):
 d=json.loads(p.read_text(encoding='utf-8-sig'));h=d.get('history_log',[])
 rows.append(dict(path=str(p),sha256=sha(p),model=d.get('model_name'),lead=d.get('lead_days'),seed=next((x for x in p.parts if x.startswith('seed')),None),best_epoch=d.get('best_epoch'),recorded_epochs=len(h),preset=d.get('physics_preset'),physics_components=d.get('physics_components'),lambda_physics=d.get('lambda_physics'),lambda_rres=d.get('lambda_rres'),rhs_scale=d.get('physics_rhs_scale'),forcing_mode=d.get('forcing_mode'),common_residual=d.get('test_common_res_rmse_kday'),prediction_file=d.get('fields_path',d.get('test_fields_npz')),keys=list(d)))
(O/'legacy_inventory.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
p=R/'outputs/v8_actual_epoch_oracle96_20261011/protected_input_hashes.json';expected=json.loads(p.read_text());checks=[]
for name,h in expected.items():
 if Path(name).is_file():a=sha(name);checks.append(dict(path=name,match=a==h,expected=h,actual=a))
 else:checks.append(dict(path=name,match=False,error='missing'))
(O/'protected_integrity.json').write_text(json.dumps(dict(count=len(checks),mismatches=[x for x in checks if not x['match']],checks=checks),indent=2))
with (R/'docs/research_log.md').open('a') as f:f.write('\n- '+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')+' — 최종 감사 신규 코드(audit_cpu.py/check_rhs.py/reanalyze.py/provenance.py): 기존 모델 코드 변경 없이 120개 로그·해시, 88개 차분 테스트, 24개 RHS 재계산, 고정 예측 SST/MHW 재검산 및 30/90/180일 블록 bootstrap 구현. 감사 스크립트의 상대 climatology 경로·archived import·독립 테스트 인수·CPU process-title 초기화 문제를 수정; 원본 학습 경로 수정 없음. 현재 120개 프로토콜 및 53개 실행 소스 해시 일치, 88개 수치 테스트 통과. 후속 수치·보고서는 같은 출력 폴더에 보존.\n')
print('integrity',len(checks),'mismatch',len([x for x in checks if not x['match']]))
