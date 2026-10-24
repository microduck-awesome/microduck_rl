"""Use one fresh interpreter per historical phase; do not run environments or PPO."""
from pathlib import Path
import os,json,subprocess
ROOT=Path(__file__).resolve().parents[1];SOURCES=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762/sources');rows=[]
for phase in json.loads((ROOT/'history-inventory.json').read_text())['phases']:
 name=phase['name'];source=SOURCES/name;env=os.environ.copy()
 for key in ('WORLD_SIZE','RANK','LOCAL_RANK','CUDA_VISIBLE_DEVICES','MICRODUCK_WARM_START','TRITON_DEFAULT_FP_FUSION','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):env.pop(key,None)
 env['SC0090_CUDA_COMPAT_DIR']='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat'
 output=ROOT/'config-checks'/f'{name}.json';output.parent.mkdir(exist_ok=True)
 with output.with_suffix('.log').open('w')as log:
  result=subprocess.run(['bash',str(source/'scripts/v2_python.sh'),str(ROOT/'tools/check_phase_config.py'),'--phase',name,'--output',str(output)],env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  print(output.with_suffix('.log').read_text()[-12000:],flush=True);raise SystemExit(result.returncode)
 d=json.loads(output.read_text());rows.append(dict(phase=name,status=d['status'],allowed_collision_entries=len(d['allowed_collision_differences']),resource_remaps=len(d['identical_resource_path_remaps'])));print(rows[-1],flush=True)
(ROOT/'config-review.json').write_text(json.dumps(dict(status='PASS',phases=rows,formal_updates=0),indent=2)+'\n')
