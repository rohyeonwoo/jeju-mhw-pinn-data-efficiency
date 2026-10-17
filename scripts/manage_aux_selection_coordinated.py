"""Resume the same experiment; prevent two independent launchers racing for free GPU slots."""
import json
import manage_aux_selection as m
_original_query=m.query

def query(kind):
 if kind=='gpu':
  state=m.ROOT/'outputs/p0_dense_validation_20261009/pilot_state.json'
  if state.exists() and json.loads(state.read_text()).get('phase')!='completed':
   # Already launched pilot workers continue; further launches await dense completion.
   m.w.save(m.OUT/'resource_wait.json',dict(timestamp=m.w.stamp(),reason='existing dense supervisor retains launch priority; active pilot workers continue',gpu_ids=[1,2,3]))
   return []
 return _original_query(kind)

if __name__=='__main__':
 m.query=query;m.main()
