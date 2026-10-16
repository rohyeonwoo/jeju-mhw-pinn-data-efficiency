"""Refill free GPU1/2/3 slots serially; preserve immutable training protocol."""
import json,time,os
from pathlib import Path
import manage_aux_selection as m
_query=m.query
last_offer=0.

def query(kind):
 global last_offer
 if kind!='gpu':return _query(kind)
 # Once every dense job is either completed or already running, its launcher
 # has no pending allocation with which this launcher could race.
 root=m.ROOT/'outputs/p0_dense_validation_20261009'
 if (root/'pilot_state.json').exists():
  state=json.loads((root/'pilot_state.json').read_text());active={}
  for row in state.get('active',[]):
   q=Path('/proc')/str(row['pid'])
   try:
    bits=q.joinpath('stat').read_text().rsplit(')',1)[1].split()
    if q.stat().st_uid==os.getuid() and bits[19]==row['start_ticks'] and bits[0]!='Z':active[row['job']]=row
   except FileNotFoundError:pass
  protocol=json.loads((root/'protocol.json').read_text());pending=[]
  for j in protocol['jobs']:
   sf=Path(j['run_dir'])/'status.json'
   done=sf.exists() and json.loads(sf.read_text()).get('status')=='trained'
   if not done and j['id'] not in active:pending.append(j['id'])
  if pending:
   m.w.save(m.OUT/'resource_wait.json',dict(timestamp=m.w.stamp(),waiting=True,reason='Other supervisor still has not-yet-started jobs; avoid simultaneous allocation',pending_dense_jobs=pending));return []
 # Offer inventory only once per scheduler poll, so one child is started,
 # then the next allocation uses a fresh GPU inventory on the next poll.
 if time.monotonic()-last_offer<15:return []
 last_offer=time.monotonic()
 m.w.save(m.OUT/'resource_wait.json',dict(timestamp=m.w.stamp(),waiting=False,reason='Refilling available GPU1/2/3 slots one worker per poll; existing processes and memory checked',max_total_processes_per_gpu=4,max_study_workers_per_gpu=3))
 return _query(kind)

if __name__=='__main__':
 m.query=query;m.main()
