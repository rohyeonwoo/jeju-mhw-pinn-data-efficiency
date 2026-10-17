"""Native P0 lambda sensitivity; only the CLI physics coefficient is changed."""
import argparse,json,os,traceback
from pathlib import Path
import train_v3_study as w

def install():
 original_arguments=w.arguments;original_setup=w.setup
 def arguments(job):
  args,argv=original_arguments(job)
  value=float(job['lambda_physics']);assert 0<=value<=.1
  assert job['arm']!='baseline' or value==0
  args.lambda_physics=value;argv[argv.index('--lambda-physics')+1]=str(value)
  return args,argv
 def setup(job,protocol,device):
  args,bundle,model,physics,cfg=original_setup(job,protocol,device)
  assert args.lambda_physics==job['lambda_physics']
  cfg['test_evaluation']='disabled for this training/validation-only lambda study'
  cfg['optimization']='native scalar data + lambda_effective * physics; no gradient projection/norm-ratio cap'
  cfg.pop('config_sha256');cfg['config_sha256']=w.digest(cfg)
  return args,bundle,model,physics,cfg
 w.arguments=arguments;w.setup=setup

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--job',type=int,required=True);opt=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','2','3'],'GPU0 prohibited; one physical GPU per worker'
 p=json.loads((opt.study/'protocol.json').read_text());assert p['phase']=='training_validation_only'
 assert p['experiment']=='native_p0_lambda_sensitivity'
 install()
 try:w.worker(opt.study,opt.job,'train')
 except Exception:
  j=p['jobs'][opt.job];w.save(Path(j['run_dir'])/'failure.json',dict(timestamp=w.stamp(),error=traceback.format_exc()))
  w.log('native P0 lambda validation failed: '+j['name']);raise
if __name__=='__main__':main()
