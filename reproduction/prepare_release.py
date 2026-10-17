"""Build a reviewable GitHub export and cleanup proposal. Never deletes/moves inputs."""
from pathlib import Path
import os,json,csv,hashlib,collections,shutil,subprocess,tarfile,datetime
from zoneinfo import ZoneInfo
R=Path('/ceph_data/yeonwoo/MHW_subset');D=R/'reproduction';RELEASE=Path('/ceph_data/yeonwoo/_work/subset_github_20261016');EXPORT=RELEASE/'MHW_subset'
D.mkdir(exist_ok=True);RELEASE.mkdir(parents=True,exist_ok=True)
if EXPORT.exists():raise RuntimeError('Existing export: preserve it and inspect before another build')
protected=set();reforigin=collections.defaultdict(set)
# Preserve actual files named in protected-input hash maps. Hash ledgers are evidence, not trash.
for p in (R/'outputs').rglob('*hash*.json'):
 try:x=json.loads(p.read_text())
 except (ValueError,OSError):continue
 def walk(v):
  if isinstance(v,dict):
   for k,w in v.items():
    if isinstance(k,str) and k.startswith('/'):
     q=Path(k)
     if q.is_file():protected.add(str(q));reforigin[str(q)].add(str(p.relative_to(R)))
    walk(w)
  elif isinstance(v,list):
   for w in v:walk(w)
 walk(x)
# Explicit core artifacts and outputs: keep all final checkpoints/predictions, not only tables.
core=['outputs/v8_actual_epoch_oracle96_20261011','outputs/v8_actual_epoch_auxiliary24_20261015','outputs/paper_final_verification_20261015','outputs/paper_analysis_20261016','outputs/v7_gpu_mixed_budget_comparison_20261011/features','outputs/main_experiment_final','paper_analysis_20260929','data','src','scripts','tests','docs','reproduction']
binext={'.pt','.pth','.ckpt','.npy','.npz','.pkl','.pickle','.bin','.h5','.hdf5'}
text_ext={'.py','.sh','.ps1','.md','.txt','.csv','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.tex','.bib','.r','.rst','.html','.css','.js'}
fig_ext={'.png','.pdf','.svg'}
rows=[];candidates=[];external=[]
print('Inventory and dependency retention',flush=True)
for root,dirs,files in os.walk(R):
 dirs[:]=[s for s in dirs if s not in {'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}]
 for name in files:
  p=Path(root)/name;rel=str(p.relative_to(R))
  if p.is_symlink() or not p.is_file():continue
  st=p.stat();suf=p.suffix.lower();iscore=any(rel==x or rel.startswith(x+'/') for x in core)
  if iscore:decision='KEEP';why='Final120, scientific inputs, evidence, source, or project documentation'
  elif str(p) in protected:decision='KEEP';why='Referenced in retained hash/protection ledger'
  elif rel.startswith('outputs/') and suf in binext:decision='PROPOSE_DELETE';why='Intermediate/pilot binary outside final120 and protected dependencies'
  else:decision='KEEP';why='Small evidence/metadata/history or unclassified: conservative retention'
  row=dict(path=rel,bytes=st.st_size,mtime_ns=st.st_mtime_ns,decision=decision,reason=why,reference_ledgers=';'.join(sorted(reforigin[str(p)])))
  rows.append(row)
  if decision=='PROPOSE_DELETE':candidates.append(row)
for s in sorted(protected):
 p=Path(s)
 if not p.is_relative_to(R):external.append(dict(path=s,bytes=p.stat().st_size,reference_ledgers=';'.join(sorted(reforigin[s]))))
def csvwrite(p,rs,fields=None):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rs[0]) if rs else (fields or ['path','bytes']));w.writeheader();w.writerows(rs)
csvwrite(D/'retention_inventory.csv',rows);csvwrite(D/'cleanup_candidates.csv',candidates);csvwrite(D/'external_dependencies.csv',external)
by=collections.defaultdict(lambda:[0,0])
for x in candidates:by[x['path'].split('/')[1]][0]+=1;by[x['path'].split('/')[1]][1]+=x['bytes']
summary=[dict(directory=k,files=n,bytes=b,GiB=round(b/1024**3,4)) for k,(n,b) in sorted(by.items(),key=lambda kv:-kv[1][1])];csvwrite(D/'cleanup_summary.csv',summary)
# Freeze observed package metadata without importing torch or initializing a device.
py=Path('/ceph_data/yeonwoo/conda_envs/mhw_subset/bin/python')
r=subprocess.run([str(py),'-B','-c',"import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'packages':sorted([(d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']])},indent=2))"],capture_output=True,text=True,check=True)
(D/'observed_environment.json').write_text(r.stdout)
(D/'requirements-observed.txt').write_text('# Observed environment on export date; not claimed to be the original run-time lockfile.\n'+'\n'.join(n+'=='+v for n,v in json.loads(r.stdout)['packages'])+'\n')
# Build a named external-input list with stored hashes, without rereading multi-GB ocean data.
inputs=[]
for p in [R/'outputs/v7_gpu_mixed_budget_comparison_20261011/features/manifest.json',R/'outputs/v8_actual_epoch_oracle96_20261011/protected_input_hashes.json']:
 x=json.loads(p.read_text())
 if 'source' in x:
  for path,h in x['source'].items():
   q=Path(path);inputs.append(dict(path=path,exists=q.is_file(),bytes=q.stat().st_size if q.is_file() else None,sha256_prior=h,hash_source=str(p)))
(D/'scientific_inputs.json').write_text(json.dumps(inputs,indent=2))
existing=(R/'requirements.txt').read_text();(D/'requirements-reproduction.txt').write_text('-r ../requirements.txt\nsetproctitle\nscipy\npillow\nnetCDF4\n')
removed=sum(x['bytes'] for x in candidates);total=sum(x['bytes'] for x in rows)
report=f'''# 재현 보존 및 정리 제안 — 삭제 미실행\n\n프로젝트 파일 합계: {total/1024**3:.3f} GiB.\n정리 후보: {len(candidates)}개 파일, {removed/1024**3:.3f} GiB.\n제안 적용 후 계산상 잔여량: {(total-removed)/1024**3:.3f} GiB. 이는 논리 파일 크기이며 Ceph 실제 할당/압축/스냅샷 반환량과 다를 수 있다.\n\n## 보존\n\n- 원자료, subset JSON 및 v1/v2/v3 전처리 전부\n- `src/`, `scripts/`, `tests/`, 문서 및 모든 작은 과거 설정/지표/보고서\n- 최종 Oracle96 + Auxiliary24: checkpoint 전체, 학습 history, 예측·평가 및 설정\n- 최종 verification과 논문 분석 산출물 전체\n- v7 혼합층 feature 배열 및 생성 코드\n- 과거 해시/보호 목록에서 직접 참조하는 파일: 원래 무결성 검사와 증거 연결 보존\n- WSDM/Full 결과 및 20260929 분석: 별도 기존 근거 보존\n- 외부 GLORYS 원자료: `external_dependencies.csv`, `scientific_inputs.json` 참조. 이 프로젝트 밖의 파일은 정리 대상 아님.\n\n## 정리 후보\n\n아래 각 폴더 전체가 아니라 `cleanup_candidates.csv`의 파일만 대상이다. 중간·pilot 바이너리 중 최종120 결과나 보호 해시 의존 파일이 아닌 항목이다. 참조 증거는 남기지만 이 후보를 삭제하면 해당 과거 pilot의 checkpoint/예측을 이용한 재분석은 불가능할 수 있다. 그 pilot의 기록 자체를 숨기거나 삭제하지 않는다.\n\n| 폴더 | 파일 수 | GiB |\n|---|---:|---:|\n'''+''.join(f"| `{x['directory']}` | {x['files']} | {x['GiB']:.3f} |\n" for x in summary)+'''\n## 승인 및 업로드 상태\n\n- 어떤 원본 파일도 삭제/이동하지 않았다. 이 목록을 확인하고 삭제를 승인하면 해당 CSV만 적용한다.\n- 프로젝트 AGENTS.md의 파일 삭제/이동 사전 목록·승인 규칙에 따른 최종 확인 단계다.\n- GitHub 업로드는 아직 수행하지 않았다. 별도 export 사본과 압축 파일은 다운로드/업로드 준비용이며 원본 대용량 보관의 백업이 아니다.\n- GitHub에는 대용량 raw data/checkpoint/예측 배열을 포함하지 않는다. 삭제 후보가 아닌 최종 checkpoint나 raw data까지 지우면 GitHub 사본만으로 기존 결과를 바로 복구할 수 없다.\n'''
(D/'cleanup_plan_ko.md').write_text(report)
README='''# Oracle120 subset experiment: reproduction bundle\n\nThis export contains the frozen scientific source, subset/protocol manifests, configuration and training histories, final metrics, statistical evidence, publication figures and analysis code. Large ocean data, checkpoints and prediction arrays remain outside GitHub. This is a curated publication source snapshot, not a complete backup of Ceph.\n\n## Start here\n\n- Final analysis: `outputs/paper_analysis_20261016/README.md`\n- Seven research questions: `outputs/paper_analysis_20261016/reports/analysis_summary_ko.md`\n- Protocol: `outputs/paper_final_verification_20261015/experiment_protocol_audit.md`\n- Exact retained and proposed-cleanup files: `reproduction/retention_inventory.csv`, `cleanup_candidates.csv`\n- Exported file checksums: `GITHUB_EXPORT_SHA256SUMS`\n\n## Scope\n\n5 models (Baseline, Auxiliary-only, P0, NoAdv, Adv), 4 nested subsets (10/25/50/100%), leads 7/14, seeds42/7/123. 120 runs,80 actual epochs, Validation-best. The 100% subset is3995 retained windows, not all original training samples. Test has been examined previously; conclusions are exploratory. Older WSDM experiments remain separate.\n\n## Reproduction levels\n\n1. **Review the reported results:** final CSV/LaTeX tables and figures/captions are included. No training/data download is needed to inspect them.\n2. **Recompute full analysis from frozen predictions:** requires the Ceph-retained final120 predictions/daily MHW labels, original metadata and physical features, in addition to this repository. `outputs/paper_analysis_20261016/code/analyze.py`, `refine_figures.py`, `finalize_findings.py` reproduce the analysis.\n3. **Repeat training/evaluation:** requires budget2d raw NetCDF, external3D GLORYS source (or the exact retained features), v3 training manifests plus referenced v2 climatologies, and the environment. Keep retained protocol/config/indices/checkpoints and the recorded source unchanged for exact provenance. GPU scheduling is server-specific. No training was run as part of preparing this export.\n\n## Environment and paths\n\nUse `requirements.txt` plus `reproduction/requirements-reproduction.txt`. Install an appropriate PyTorch CUDA build explicitly. `requirements-observed.txt`/`observed_environment.json` capture the environment at export time; they are not asserted to be the original execution lockfile. `setproctitle` is required by Oracle runners (not optional there).\n\nOriginal scripts deliberately preserve absolute Ceph paths, physical GPU IDs and hash guards. This snapshot is **not a portable one-command clone-and-train release**. On the original server use the recorded `/ceph_data/yeonwoo/MHW_subset` layout and environment. Elsewhere provide the same dataset paths/mount layout or perform an explicit documented path/scheduler adaptation and regenerate new-run manifests without overwriting historical ones. Do not silently edit frozen source then claim its original hashes still verify.\n\nPrimary training code: `scripts/oracle96_actual/worker.py` and `scripts/oracle96_auxiliary24/aux_worker.py`. Historical `scripts/oracle96/run_all.py` uses a different virtual-epoch protocol: do not confuse it with the final actual-epoch experiment. Queue controllers have side effects; read protocols before launching.\n\nInput provenance: `reproduction/scientific_inputs.json`, `external_dependencies.csv`, feature manifest and recorded source hashes. `reproduction/check_retained.py` is a read-only file-existence/metadata check; it does not import trainers or initialize GPUs. Use `--root` to inspect a full retained Ceph tree. A GitHub-only copy is expected to report missing large files.\n\n## Storage policy\n\nThe files proposed for cleanup are intermediate binary artifacts outside final120 and recorded protected-input dependencies. Deleting them sacrifices reanalysis of those specific pilot predictions/checkpoints; small historical evidence is retained. No cleanup has occurred until explicit approval. GitHub uploading alone is not a backup of external inputs or trained models.\n\nThe export preserves source files byte-for-byte. Its top-level README/.gitignore are export-specific; the original README is `reproduction/README_original.md`. Runtime caches, credentials, machine logs, data arrays and model weights are excluded.\n'''
(D/'README_reproduction.md').write_text(README)
# Read-only validator consumes the reviewed manifest; it never launches entrypoints.
validator='''from pathlib import Path\nimport argparse,csv,json\np=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();missing=[];changed=[];n=0\nwith (a.root/'reproduction/retention_inventory.csv').open() as f:\n for r in csv.DictReader(f):\n  if r['decision']!='KEEP':continue\n  n+=1;q=a.root/r['path']\n  if not q.is_file():missing.append(r['path'])\n  elif q.stat().st_size!=int(r['bytes']):changed.append(r['path'])\nprint(json.dumps(dict(kept_files_checked=n,missing=missing,changed_size=changed),indent=2))\nraise SystemExit(1 if missing or changed else 0)\n'''
(D/'check_retained.py').write_text(validator)
# Export only scientific text and final figures; do not recursively include cache directories or symlinks.
EXPORT.mkdir();exported=[];omitted=[]
for root,dirs,files in os.walk(R):
 dirs[:]=[x for x in dirs if x not in {'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','runtime','tmp'}]
 for name in files:
  p=Path(root)/name;rel=p.relative_to(R);suf=p.suffix.lower()
  if p.is_symlink() or not p.is_file():continue
  include=(suf in text_ext or name in {'.gitignore','.gitattributes','AGENTS.md','TRANSFER_SHA256SUMS'})
  if suf in fig_ext:include=str(rel).startswith(('outputs/paper_analysis_20261016/','paper_analysis_20260929/'))
  if name.startswith('.env') or any(x in name.lower() for x in ['auth.json','credentials','token']):include=False
  if p.stat().st_size>40*1024**2:include=False
  if not include:continue
  dest=EXPORT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);exported.append(str(rel))
shutil.copy2(R/'README.md',EXPORT/'reproduction/README_original.md')
(EXPORT/'README.md').write_text(README)
(EXPORT/'.gitignore').write_text('''# Large scientific inputs and trained/predicted arrays stay outside GitHub.\n*.nc\n*.npy\n*.npz\n*.pt\n*.pth\n*.ckpt\n*.pkl\n*.pickle\n*.h5\n*.hdf5\n*.tar\n*.tar.gz\n*.zip\n*.log\n__pycache__/\n*.py[cod]\n**/runtime/\n**/tmp/\n.venv/\n.env\n.env.*\n!.env.example\n.vscode/\n.idea/\n''')
# Integrity of exported source and maximum blob size; no execution of training modules.
checks=[]
for base in ['src','scripts','tests']:
 for p in (R/base).rglob('*'):
  if p.is_file() and p.suffix=='.py':
   q=EXPORT/p.relative_to(R);assert q.is_file(),q;assert p.read_bytes()==q.read_bytes(),p
   compile(q.read_bytes(),str(q),'exec');checks.append(str(q.relative_to(EXPORT)))
files=[p for p in EXPORT.rglob('*') if p.is_file()];assert all(p.stat().st_size<50*1024**2 for p in files)
with (EXPORT/'GITHUB_EXPORT_SHA256SUMS').open('w') as f:
 for p in sorted(files):f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(EXPORT))+'\n')
# Confirm all KEEP files still exist, without reading checkpoint payloads.
assert all((R/x['path']).is_file() for x in rows if x['decision']=='KEEP')
archive=RELEASE/'MHW_subset_github_20261016.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(EXPORT,arcname='MHW_subset')
validation=dict(project_bytes=total,proposed_cleanup_files=len(candidates),proposed_cleanup_bytes=removed,remaining_bytes=total-removed,export_files=len(files)+1,export_bytes=sum(p.stat().st_size for p in files),archive_bytes=archive.stat().st_size,source_files_verified=len(checks),input_files_deleted=0,input_files_moved=0,github_upload_performed=False,protected_paths=len(protected),external_referenced_files=len(external))
(D/'release_validation.json').write_text(json.dumps(validation,indent=2));(RELEASE/'release_validation.json').write_text(json.dumps(validation,indent=2))
now=datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
with (R/'docs/research_log.md').open('a') as f:f.write(f'\n\n### {now} — subset GitHub 배포 사본 및 Ceph 정리 검토 준비\n- 대상: {D}, {RELEASE}\n- 변경: 보존/정리 후보 inventory, 외부 의존성, 관측 환경, 재현 안내, 읽기전용 검사기, byte-identical source 포함 GitHub export와 tar.gz 생성.\n- 이유: 사용자 요청에 따른 재현 필수 파일 보존 및 중간 산출물 경량화 준비.\n- 검증: {len(checks)} Python source 복사 동일성/compile, export 대용량/바이너리 제외 확인. 후보 {len(candidates)}개 {removed/1024**3:.3f}GiB. 원본 삭제/이동 및 GitHub업로드 0. AGENTS.md의 구체 목록 승인 규칙에 따라 삭제는 최종 검토 대기.\n')
print(json.dumps(validation,indent=2),flush=True)
