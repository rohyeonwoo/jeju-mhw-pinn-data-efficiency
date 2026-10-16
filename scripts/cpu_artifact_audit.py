from cpu_diagnostics_common import *
def main():
 assert json.loads((ADAPT/'user_pause.json').read_text())['supervisor_stopped']
 launch=json.loads((ADAPT/'supervisor_launch.json').read_text());proc=Path('/proc')/str(launch['pid'])
 if proc.exists():assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=launch['start_ticks'],'Old GPU queue still alive'
 rows=[];hashes={}
 def checked(path,expected=None):
  path=str(Path(path).resolve())
  if path not in hashes:hashes[path]=sha(path)
  if expected is not None:assert hashes[path]==expected,path
  return hashes[path]
 pr=json.loads((OLD/'protocol.json').read_text())
 for n,v in pr['code_sha256'].items():checked(ROOT/n,v)
 for j in jobs():
  checked(j['manifest'],j['manifest_sha256']);m=json.loads(Path(j['manifest']).read_text());checked(Path(j['manifest']).parent/m['climatology_file'],m['climatology_sha256']);met=json.loads((Path(j['evaluation_dir'])/'metrics.json').read_text());checked(Path(j['run_dir'])/'best.pt',met['checkpoint_sha256']);checked(Path(j['run_dir'])/'config.json');checked(Path(j['run_dir'])/'history.json')
  for split,rec in met['fields'].items():
   checked(rec['path'],rec['sha256'])
   with np.load(rec['path']) as z:
    assert all(k in z for k in ['pred_raw','target_raw','target_index','we_raw','mask'])
    assert np.array_equal(z['target_index'],np.asarray(m[split+'_start_indices'])+29+j['lead'])
    rows.append(dict(**meta(j),model=j['arm'],split=split,path=rec['path'],sha256=rec['sha256'],samples=len(z['target_index']),we_stored=True,rhs_stored='rhs' in z,residual_stored='residual' in z,pred_shape=str(z['pred_raw'].shape),scope='saved Validation-best predictions'))
  if j['arm']=='physics_only':
   path=Path(j['evaluation_dir'])/'mhw_fields.npz';checked(path)
 for key in ['data','raw_thetao']:checked(m[key+'_path'],m[key+'_sha256'])
 for path in [OLD/'bootstrap_replicates.npz',OLD/'seed_improvements.csv',OLD/'climatology_diagnostics.csv',ADAPT/'gradient_diagnostics.csv']:
  checked(path)
 mhw=json.loads((OLD/'mhw_protocol.json').read_text());checked(mhw['reference'],mhw['sha256'])
 csvsave('artifact_inventory.csv',rows);save(OUT/'protected_input_hashes.json',hashes)
 save(OUT/'analysis_protocol.json',dict(timestamp=stamp(),cpu_only=True,threads=2,address_space_limit_gib=16,new_model_inference=False,new_backward=False,new_training=False,test_scope='Exploratory post-hoc analysis of already stored predictions; no Test-based hyperparameter selection',physics='all24 ratio/lead/seed pairs, Train/Validation/Test; native linear observed and predicted endpoints, actual daily comparator; cached Fixed we',statistics='4000 paired circular block resamples,30/90/180days; fixed3 seeds; centered bootstrap null p-values and Holm family8; MHW event labels built before resampling',climatology='read-only existing subset artifacts; no refit; unique-history and tensor-occurrence summaries',adaptive='offline rho .05/.10/.25; no change to paused pilot protocol',mhw=mhw))
 finish('audit',saved_prediction_files=len(rows),protected_paths=len(hashes))
if __name__=='__main__':main()
