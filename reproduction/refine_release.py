"""Finalize the reviewable minimal-retention proposal after prepare_release.py. No deletion."""
from pathlib import Path
import csv,json,collections,hashlib,shutil,tarfile,datetime,subprocess,sys
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');D=R/'reproduction';BASE=Path('/ceph_data/yeonwoo/_work/subset_github_20261016');E=BASE/'MHW_subset'
assert (D/'release_validation.json').is_file(),'Wait for initial export to finish'
def read(p):return list(csv.DictReader(p.open()))
def put(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
rows=read(D/'retention_inventory.csv');lookup={x['path']:x for x in rows};final=[];extra=[]
prior={x['name']:x for x in read(R/'outputs/paper_final_verification_20261015/protocol120.csv')}
for proto in ['outputs/v8_actual_epoch_oracle96_20261011/experiment_protocol.json','outputs/v8_actual_epoch_auxiliary24_20261015/training_protocol.json']:
 for j in json.loads((R/proto).read_text())['jobs']:
  rd=Path(j['run_dir']);status=json.loads((rd/'status.json').read_text());assert status['status']=='COMPLETED',rd
  c=json.loads((rd/'config.json').read_text());cp=Path(c['checkpoint_dir']);best=cp/'best.pt';assert best.is_file()
  final.append(dict(name=j['name'],checkpoint=str(best.relative_to(R)),bytes=best.stat().st_size,checkpoint_sha256_prior=prior[j['name']]['checkpoint_sha256'],hash_scope='Prior final audit hash; not rehashed during export',status='KEEP validation-best for primary result'))
  for name in ['last.pt','post_warmup_best.pt']:
   p=cp/name
   if not p.is_file():continue
   rel=str(p.relative_to(R));row=lookup[rel];assert not row['reference_ledgers'],rel
   row['decision']='PROPOSE_DELETE';row['reason']='Completed final run: restart/alternate-selection checkpoint; validation-best retained; no primary evaluation dependency';extra.append(row)
assert len(final)==120
put(D/'final_checkpoint_inventory.csv',final);put(D/'retention_inventory.csv',rows)
candidates=[x for x in rows if x['decision']=='PROPOSE_DELETE'];put(D/'cleanup_candidates.csv',candidates)
by=collections.defaultdict(lambda:[0,0])
for x in candidates:
 key=x['path'].split('/')[1];by[key][0]+=1;by[key][1]+=int(x['bytes'])
summ=[dict(directory=k,files=v[0],bytes=v[1],GiB=round(v[1]/1024**3,4)) for k,v in sorted(by.items(),key=lambda kv:-kv[1][1])];put(D/'cleanup_summary.csv',summ)
total=sum(int(x['bytes']) for x in rows);amount=sum(int(x['bytes']) for x in candidates);extras=sum(int(x['bytes']) for x in extra)
plan=f'''# 재현 필수 파일 보존 및 정리 제안 — 삭제 미실행\n\n- 조사한 원본: {total/1024**3:.3f} GiB\n- 정리 후보: **{len(candidates)}개 파일, {amount/1024**3:.3f} GiB**\n- 적용 후 원본 예상 잔여량: **{(total-amount)/1024**3:.3f} GiB**\n- 이 중 완료된 최종 학습의 last/post-warmup checkpoint: {len(extra)}개, {extras/1024**3:.3f} GiB\n\n논리 파일 크기 합계이며 Ceph 스냅샷/할당/압축에 따른 물리적 반환량과 다를 수 있다. 새 GitHub export/압축본은 별도 공간을 사용한다.\n\n## 반드시 남김\n\n1. `src/`, `scripts/`, `tests/`, subset 설계·전처리(v1/v2/v3)·원자료 및 config/manifest/학습 history\n2. **최종120 Validation-best checkpoint 전부**: `final_checkpoint_inventory.csv`\n3. 최종120 저장 예측·MHW 라벨·평가, 최종 검증과 논문 Figure/Table/plot data/통계/보고서\n4. 실제 해시 보호 목록이 참조하는 과거 파일과 v7 물리 feature 배열\n5. 기존 WSDM/Full 자료와 이전 논문분석: 별도 근거로 유지\n6. 외부3D GLORYS 원자료(프로젝트 밖3.19GiB): `external_dependencies.csv`; 이 작업의 정리 범위 밖\n\n## 삭제 대상의 정확한 범위\n\n폴더 전체가 아닌 `cleanup_candidates.csv`의 경로만 대상이다.\n\n- 최종120에 쓰이지 않는 중간·pilot checkpoint와 예측/진단 배열(보호 해시 참조 파일 제외)\n- 완료된120개 학습의 `last.pt`, 및 Oracle96의 `post_warmup_best.pt`. 최종 `best.pt`는 항상 남긴다. 삭제 후 중단 지점의 optimizer/RNG 상태 재개와 이 대체 checkpoint로 평가하는 기능은 잃는다. 처음부터 재학습하거나 최종 선택 모델로 평가하는 데는 사용하지 않는다.\n- 중간 pilot을 saved predictions/weights에서 다시 분석하는 기능은 해당 후보 삭제 시 잃을 수 있다. 작은 수치/설정/보고서와 실행기 코드는 유지한다.\n\n| 폴더 | 후보 파일 | GiB |\n|---|---:|---:|\n'''+''.join(f"| `{x['directory']}` | {x['files']} | {x['GiB']:.3f} |\n" for x in summ)+'''\n## 현재 상태\n\n**삭제·이동 및 GitHub 업로드 모두 미실행.** GitHub 준비 사본은 코드/작은 설정·결과·그림만 포함한다. 대용량 데이터와 모델의 대체 백업이 아니다. 최종120 가중치와 데이터까지 Ceph에서 지우면 GitHub만으로 기존 모델을 그대로 복구할 수 없다.\n\nAGENTS.md의 “파일 삭제, 이동, 모델 코드 수정은 반드시 목록을 먼저 보여주고 승인을 받은 뒤 진행한다” 규칙에 따라 이 구체 목록의 최종 승인을 기다린다.\n'''
(D/'cleanup_plan_ko.md').write_text(plan)
# Ignore administrative files deliberately changed by this packaging operation.
validator="""from pathlib import Path
import argparse,csv,json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();missing=[];changed=[];n=0
with (a.root/'reproduction/retention_inventory.csv').open() as f:
 for r in csv.DictReader(f):
  rel=r['path']
  if r['decision']!='KEEP' or rel.startswith('reproduction/') or rel=='docs/research_log.md' or rel.endswith('.log'):continue
  n+=1;q=a.root/rel
  if not q.is_file():missing.append(rel)
  elif q.stat().st_size!=int(r['bytes']):changed.append(rel)
print(json.dumps(dict(kept_scientific_files_checked=n,missing=missing,changed_size=changed),indent=2))
raise SystemExit(1 if missing or changed else 0)
"""
(D/'check_retained.py').write_text(validator)
result=subprocess.run([sys.executable,str(D/'check_retained.py'),'--root',str(R)],capture_output=True,text=True,check=True);(D/'retention_check.json').write_text(result.stdout)
validation=json.loads((D/'release_validation.json').read_text());validation.update(proposed_cleanup_files=len(candidates),proposed_cleanup_bytes=amount,remaining_bytes=total-amount,final_validation_best_preserved=120,completed_resume_alternate_candidates=len(extra),kept_scientific_files_check=json.loads(result.stdout),input_files_deleted=0)
(D/'release_validation.json').write_text(json.dumps(validation,indent=2));(BASE/'release_validation.json').write_text(json.dumps(validation,indent=2))
readme=(D/'README_reproduction.md').read_text();readme=readme.replace('outside final120 and recorded protected-input dependencies.','outside primary final120 results and recorded protected-input dependencies, plus completed-run restart/alternate-selection checkpoints. All120 Validation-best weights remain on Ceph.');readme+='\n## Reviewed minimal retention proposal\n\nSee reproduction/cleanup_plan_ko.md for exact candidates. Completed last.pt/post_warmup_best.pt are proposed for deletion, but primary best.pt120/120 and all frozen final predictions remain on Ceph. Exact optimizer/RNG resume and old alternate/pilot checkpoint reanalysis would be sacrificed.\n';(D/'README_reproduction.md').write_text(readme)
now=datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
with (R/'docs/research_log.md').open('a') as f:f.write(f'\n- {now} — subset 최소 보존 목록 확정: 최종120개 Validation-best 전부 보존 확인. 완료된 last/post-warmup {len(extra)}개를 후보에 추가, 총{len(candidates)}개 {amount/1024**3:.3f}GiB 제안. 보존 scientific파일 존재/크기 검사 통과. 코드·데이터 삭제/이동 없음, 구체 목록 승인 대기.\n')
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.py','.md','.csv','.json','.txt'}:shutil.copy2(p,E/'reproduction'/p.name)
shutil.copy2(R/'docs/research_log.md',E/'docs/research_log.md');(E/'README.md').write_text(readme)
files=[p for p in E.rglob('*') if p.is_file() and p.name!='GITHUB_EXPORT_SHA256SUMS']
with (E/'GITHUB_EXPORT_SHA256SUMS').open('w') as f:
 for p in sorted(files):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(E))+'\n')
archive=BASE/'MHW_subset_github_20261016.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(E,arcname='MHW_subset')
# Tar member sizes match export, no raw/weight arrays can enter accidentally.
with tarfile.open(archive) as tar:
 members=[x for x in tar.getmembers() if x.isfile()];assert len(members)==len(files)+1
 for x in members:assert Path(x.name).suffix not in {'.nc','.pt','.pth','.ckpt','.npy','.npz','.pkl','.h5'}
validation.update(archive_bytes=archive.stat().st_size,export_files=len(files)+1,export_bytes=sum(p.stat().st_size for p in E.rglob('*') if p.is_file()),archive_validation='member count and binary exclusions checked')
(D/'release_validation.json').write_text(json.dumps(validation,indent=2));(BASE/'release_validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2),flush=True)
