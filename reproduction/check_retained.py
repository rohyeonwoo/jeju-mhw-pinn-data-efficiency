from pathlib import Path
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
