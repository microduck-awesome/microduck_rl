"""Read-only inventory of the exact ten-run ancestor path to baseline3762."""
from pathlib import Path
import hashlib,json,tarfile,shutil,re
ROOT=Path(__file__).resolve().parents[1]
HISTORY=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline')
EXTRACT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762/historical-sources')
NAMES=['standup-allocation-001','standup-allocation-002','standup-allocation-003','head-precision-allocation-001','head-precision-allocation-002','head-precision-allocation-003','head-precision-allocation-004','head-precision2-allocation-001','head-foundation4-allocation-001','paired-feedback-allocation-001']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):
 if isinstance(x,dict):return {k:canonical(v)for k,v in x.items()}
 if isinstance(x,list):return [canonical(v)for v in x]
 if isinstance(x,str):return re.sub(r' at 0x[0-9a-fA-F]+(?=>)','',x)
 return x
def digest(x):return hashlib.sha256(json.dumps(canonical(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
rows=[];previous={}
for index,name in enumerate(NAMES):
 src=HISTORY/name;dst=ROOT/'evidence'/name;dst.mkdir(parents=True,exist_ok=True);tree=EXTRACT/name;tree.mkdir(parents=True,exist_ok=True)
 run=json.loads((src/'run.json').read_text());status=json.loads((src/'status.json').read_text())
 for fn in ('run.json','status.json','source.tar.gz'):
  shutil.copy2(src/fn,dst/fn)
 preflight=Path(run['preflight']);assert preflight.exists(),preflight
 shutil.copy2(preflight,dst/'historical-preflight.json')
 raw={}
 with tarfile.open(src/'source.tar.gz')as tar:
  for m in tar.getmembers():
   assert m.isfile() and not Path(m.name).is_absolute() and '..'not in Path(m.name).parts,m.name
   data=tar.extractfile(m).read();raw[m.name]=hashlib.sha256(data).hexdigest();target=tree/m.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 assert raw==run['source_sha256'],name
 evaluations=[]
 for f in sorted(src.glob('evaluation_*.json')):
  shutil.copy2(f,dst/f.name);e=json.loads(f.read_text());evaluations.append(dict(file=f.name,sha256=sha(f),iteration=e.get('iteration'),decision=e.get('decision')))
 checkpoints={}
 for f in sorted([*src.glob('checkpoint_*.pt'),*src.glob('resume_*.pt')]):checkpoints[f.name]=dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)
 pre=json.loads(preflight.read_text())
 row=dict(index=index,name=name,historical_directory=str(src),run_sha256=sha(src/'run.json'),source_archive_sha256=sha(src/'source.tar.gz'),source_files=raw,changed_source_files=[n for n,h in raw.items()if previous.get(n)!=h],removed_source_files=sorted(set(previous)-set(raw)),environment_canonical_sha256=digest(run['environment']),agent_canonical_sha256=digest(run['agent']),task=run['task'],seed=run['seed'],device=run['device'],num_envs=run['num_envs'],allocated_updates=run['allocation_updates'],completed_updates=status['completed_updates'],resume=run.get('resume'),evaluation_interval=run['evaluation_interval'],evaluation_samples_per_bucket_per_seed=run.get('evaluation_samples_per_bucket_per_seed'),head_precision_recipe=run.get('head_precision_recipe'),position_robustness_recipe=run.get('position_robustness_recipe'),dependencies=run['dependencies'],historical_preflight=dict(path=str(preflight),sha256=sha(preflight),robot_xml=pre.get('robot_xml'),robot_xml_sha256=pre.get('robot_xml_sha256'),keys=sorted(pre)),checkpoints=checkpoints,evaluations=evaluations)
 rows.append(row);previous=raw
(ROOT/'history-inventory.json').write_text(json.dumps(dict(schema=1,status='INVENTORIED_NOT_REVIEWED',target='rise_baseline_paired3762',phases=rows),indent=2)+'\n')
print(json.dumps(dict(phases=len(rows),source_hashes_exact=True,asset_hashes=[r['historical_preflight']['robot_xml_sha256']for r in rows],final_checkpoint=rows[-1]['checkpoints']['checkpoint_003762.pt']['sha256']),indent=2))
