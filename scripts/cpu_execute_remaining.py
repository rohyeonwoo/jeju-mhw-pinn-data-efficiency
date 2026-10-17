"""Sequential CPU analyses only; never starts the paused GPU manager."""
from cpu_diagnostics_common import *
import subprocess

def main():
 pid=734749
 while not (OUT/'physics_status.json').exists():
  proc=Path('/proc')/str(pid)
  if not proc.exists():raise RuntimeError('Physics worker exited without PASS; inspect logs/physics.log')
  time.sleep(5)
 assert json.loads((OUT/'physics_status.json').read_text())['status']=='PASS'
 for phase,script in [('adaptive_offline','cpu_offline_adaptive.py'),('mhw_statistics','cpu_mhw_statistics.py'),('climatology','cpu_climatology_diagnostics.py'),('report','cpu_diagnostics_report.py'),('figures','cpu_diagnostics_figures.py')]:
  save(OUT/'execution_state.json',dict(phase=phase,status='running',timestamp=stamp(),cuda_disabled=True))
  with (OUT/'logs'/(phase+'.log')).open('w') as f:
   result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/script)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:
   save(OUT/'execution_state.json',dict(phase=phase,status='failed',returncode=result.returncode,timestamp=stamp(),log=str(OUT/'logs'/(phase+'.log'))));raise RuntimeError(f'{phase} failed; no further phase started')
 save(OUT/'execution_state.json',dict(status='completed',timestamp=stamp(),cuda_disabled=True,new_training=0))
if __name__=='__main__':main()
