"""Run the unchanged all-head ablation trainer for fixed lambda=.01 extension."""
import argparse,os,traceback,json
from pathlib import Path
import train_v3_ablation as a
w=a.w

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);o=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1','2']
 p=json.loads((o.study/'protocol.json').read_text());j=p['jobs'][o.job]
 assert not j['reuse'] and j['lambda_rres']==0 and j['lambda_physics']==(0 if j['arm']=='baseline' else .01)
 assert json.loads((o.study/'preflight.json').read_text())['status']=='PASS'
 try:a.worker(o.study,o.job,'train')
 except Exception:
  d=Path(j['run_dir']);d.mkdir(parents=True,exist_ok=True);w.save(d/'failure.json',dict(error=traceback.format_exc(),timestamp=w.stamp()));raise
if __name__=='__main__':main()
