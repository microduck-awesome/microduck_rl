"""Reconstruct one historical config in a fresh interpreter, with no environment/learning."""
import argparse,json,re,hashlib,inspect,importlib.metadata,sys,os,platform
from pathlib import Path
from dataclasses import asdict
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];SOURCE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762/sources')/a.phase
history=json.loads((ROOT/'evidence'/a.phase/'run.json').read_text())
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
cfg=load_env_cfg(history['task']);recipe=None
if history.get('head_precision_recipe'):
 from mjlab_microduck.custom_v2.training import configure_head_precision_learning
 expected=history['head_precision_recipe'];stage=expected.get('stage',int(expected['id'].rsplit('-',1)[1]))
 recipe=configure_head_precision_learning(cfg,stage)if 'stage'in inspect.signature(configure_head_precision_learning).parameters else configure_head_precision_learning(cfg)
cfg.seed=history['seed'];cfg.scene.num_envs=history['num_envs'];agent=load_rl_cfg(history['task']);agent.logger='tensorboard';agent.upload_model=False;agent.save_interval=250;agent.seed=cfg.seed
actual=json.loads(json.dumps(dict(environment=asdict(cfg),agent=asdict(agent),head_precision_recipe=recipe),default=str))
expected={k:history.get(k)for k in actual}
def canonical(x):
 if isinstance(x,dict):return {k:canonical(v)for k,v in x.items()}
 if isinstance(x,list):return [canonical(v)for v in x]
 if isinstance(x,str):return re.sub(r' at 0x[0-9a-fA-F]+(?=>)','',x)
 return x
def digest_resource(p):
 if p.is_file():return {'':hashlib.sha256(p.read_bytes()).hexdigest()}
 return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(p.rglob('*'))if f.is_file()and '__pycache__'not in f.parts}
diffs=[];remaps=[]
def compare(x,y,path=''):
 if isinstance(x,dict)and isinstance(y,dict):
  for k in sorted(set(x)|set(y)):
   if k not in x or k not in y:diffs.append(dict(path=path+'.'+k,actual=x.get(k),expected=y.get(k)))
   else:compare(x[k],y[k],path+'.'+k)
 elif isinstance(x,list)and isinstance(y,list):
  if len(x)!=len(y):diffs.append(dict(path=path,actual=x,expected=y))
  else:
   for i,(v,w)in enumerate(zip(x,y)):compare(v,w,f'{path}[{i}]')
 elif x!=y:
  if isinstance(x,str)and isinstance(y,str)and x.startswith('/')and y.startswith('/')and Path(x).exists()and Path(y).exists():
   dx,dy=digest_resource(Path(x)),digest_resource(Path(y))
   if dx==dy:remaps.append(dict(path=path,historical=y,materialized=x,resource_hashes=dx));return
  diffs.append(dict(path=path,actual=x,expected=y))
compare(canonical(actual),canonical(expected))
origins={n:str(Path(m.__file__).resolve())for n,m in sys.modules.items()if n.startswith('mjlab_microduck')and getattr(m,'__file__',None)}
assert all(Path(v).is_relative_to(SOURCE/'src')for v in origins.values()),origins
versions={n:importlib.metadata.version(n)for n in history['dependencies']};assert versions==history['dependencies']
ppo_hash=None
if history['agent']['algorithm']['class_name']!='PPO':
 from mjlab_microduck.ppo_diagnostics import UPSTREAM_UPDATE_SHA256
 from rsl_rl.algorithms import PPO
 ppo_hash=hashlib.sha256(inspect.getsource(PPO.update).encode()).hexdigest();assert ppo_hash==UPSTREAM_UPDATE_SHA256
# Only the exact new XML geom mask entries are permitted, never whole sections.
import xml.etree.ElementTree as ET
materialized=json.loads((ROOT/'materialized-sources.json').read_text())
identity=next(x for x in materialized['phases']if x['phase']==a.phase)
xml=SOURCE/'src/mjlab_microduck/custom_v2/assets/robot.xml'
assert hashlib.sha256(xml.read_bytes()).hexdigest()==identity['corrected_xml_sha256']
for name,h in history['source_sha256'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h,name
for name,h in identity['added_meshes'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h,name
shells=[n for n in ET.parse(xml).iter('geom')if n.get('name','').startswith('sim2real_shell_')]
assert len(shells)==32
allowed_paths={f'.environment.scene.entities.robot.collisions[0].{field}.^{node.get("name")}$':int(node.get(field))for field in ('contype','conaffinity','condim')for node in shells}
allowed=[d for d in diffs if d['path']in allowed_paths and d['actual']==allowed_paths[d['path']] and d['expected']is None]
assert {d['path']for d in allowed}==set(allowed_paths)
unexpected=[d for d in diffs if d not in allowed]
from triton import knobs as triton_knobs
phase_index=next(x['index']for x in json.loads((ROOT/'history-inventory.json').read_text())['phases']if x['name']==a.phase)
compiler_environment={k:os.environ.get(k)for k in ('TRITON_DEFAULT_FP_FUSION','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','WORLD_SIZE','RANK','LOCAL_RANK','MICRODUCK_WARM_START')}
assert compiler_environment['TRITON_DEFAULT_FP_FUSION']==(None if phase_index<3 else '0')
assert bool(triton_knobs.language.default_fp_fusion)==(phase_index<3)
assert platform.python_version()=='3.12.14'
report=dict(status='PASS'if not unexpected else'DIFFERENT',phase=a.phase,differences=unexpected,allowed_collision_differences=allowed,identical_resource_path_remaps=remaps,imported_source_modules=origins,dependencies=versions,upstream_ppo_update_sha256=ppo_hash,formal_updates=0,compiler_environment=compiler_environment,triton_enable_fp_fusion=bool(triton_knobs.language.default_fp_fusion),python_version=platform.python_version())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.config.json').write_text(json.dumps(actual,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],phase=a.phase,unexpected_differences=unexpected,allowed_collision_entries=len(allowed),resource_remaps=len(remaps))))
raise SystemExit(bool(unexpected))
