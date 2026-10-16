from eval_common import *
import numpy as np
from metrics import score

def main():
 p=protocol();verify_code(p)
 assert json.loads((TRAIN/'run_state.json').read_text())['status']=='COMPLETED'
 assert json.loads((TRAIN/'all96_gate.json').read_text())['status']=='PASS'
 # Known grid-day contingency table and regression error; 5-day persistence is native.
 dates=np.arange(np.datetime64('2020-01-01'),np.datetime64('2021-01-01'))
 obs=np.full((366,1,2,2),5.);pred=obs.copy();obs[:6]=15;pred[:5]=15
 mask=np.ones_like(obs);mask[:,:,0,0]=0
 fields=dict(pred_raw=pred,target_raw=obs,mask=mask,target_index=np.arange(366))
 r,_=score(fields,dates,np.full((367,2,2),10.),np.full((367,2,2),10.))
 for k,v in dict(rmse=np.sqrt(100/366),bias=-10/366,mhw_grid_hits=15,mhw_grid_misses=3,mhw_grid_false_alarms=0,mhw_grid_precision=1.,mhw_grid_pod=5/6,mhw_grid_csi=5/6,mhw_grid_f1=10/11).items():assert np.isclose(r[k],v), (k,r[k],v)
 bad=dict(fields,pred_raw=pred.copy());bad['pred_raw'][0,0,1,1]=np.nan
 try:score(bad,dates,np.full((367,2,2),10.),np.full((367,2,2),10.))
 except AssertionError:pass
 else:raise AssertionError('nonfinite ocean values accepted')
 jobs=[]
 for j in p['jobs']:
  d=Path(j['run_dir']);s=json.loads((d/'status.json').read_text());a=json.loads((d/'checkpoint_audit.json').read_text())['best'];cp=TRAIN/'checkpoints'/j['name']/'best.pt'
  assert s['status']=='COMPLETED' and a['status']=='PASS' and s['best_epoch']==a['epoch']
  h=sha(cp);assert h==a['sha256']
  jobs.append(dict(j,checkpoint=str(cp),checkpoint_sha256=h,best_epoch=s['best_epoch'],best_val_rmse=s['best_val_rmse']))
  if len(jobs)%24==0:print('checkpoints verified',len(jobs),flush=True)
 assert len(jobs)==96
 ref=sha(TRAIN/'evaluation_reference.npz');assert ref==p['evaluation_reference_sha256']
 result=dict(created_at=stamp(),jobs=jobs,training_protocol_sha256=sha(TRAIN/'experiment_protocol.json'),reference_sha256=ref,
 checkpoint_selection='overall minimum Validation SST RMSE only; frozen before Test inference',gpus=[7,8,9],max_processes_per_gpu=2,
 scientific_policy='No training, tuning, checkpoint selection or normalization fitting on Test. Existing MHW 90th percentile / DOY convention / 5-day persistence / area fraction 0.15 unchanged. Historical Test exposure: exploratory evaluation.',
 evaluation_code_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
 assert not (EVAL/'evaluation_protocol.json').exists(),'Preserve existing frozen evaluation protocol'
 save(EVAL/'evaluation_protocol.json',result);save(EVAL/'preflight.json',dict(status='PASS',checkpoint_count=96,scoring_regression='PASS',nonfinite_rejection='PASS',frozen_scientific_code='PASS',timestamp=stamp()))
 log('scripts/oracle96_test/*.py 신규 생성: 고정 Validation-best 96개 Test SST/MHW 추론·미래 forcing 불변성 검사·별도 결과 집계. 기존 학습 코드/통계/체크포인트 보존. CPU 지표 회귀 및 96 체크포인트 SHA 검증 PASS. GPU7/8/9 각 최대2, 다른 작업 메모리 고려.')
 print('CPU preflight PASS',flush=True)
if __name__=='__main__':main()
