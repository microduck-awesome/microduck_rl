"""Cross-check document stage table against original argv, manifests and checkpoint states."""
from pathlib import Path
import ast,argparse,shlex,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1];inv=json.loads((ROOT/'history-inventory.json').read_text());launch=json.loads((ROOT/'historical-launch-evidence.json').read_text());cps=json.loads((ROOT/'checkpoint-history.json').read_text());doc=(ROOT/'STRICT_3762_SINGLE_GPU.md').read_text()
EXTRACT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762/historical-sources')
checks=[];previous=0
for phase,cp in zip(inv['phases'],cps['checkpoints']):
 name=phase['name'];tree=ast.parse((EXTRACT/name/'scripts/train_v2_standup.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');parser=argparse.ArgumentParser(add_help=False)
 for stmt in main.body:
  if not isinstance(stmt,ast.Expr)or not isinstance(stmt.value,ast.Call):continue
  call=stmt.value
  if not isinstance(call.func,ast.Attribute)or call.func.attr!='add_argument':continue
  kw={k.arg:({'int':int,'float':float,'str':str,'Path':Path}[k.value.id]if k.arg=='type'else ast.literal_eval(k.value))for k in call.keywords}
  parser.add_argument(*[ast.literal_eval(x)for x in call.args],**kw)
 argv=shlex.split(launch['commands'][name]['launch_command'].split(' > ')[0]);a=parser.parse_args(argv[2:])
 for key,argkey in [('task','task'),('seed',None),('device','device'),('num_envs','num_envs'),('allocated_updates','iterations'),('evaluation_interval','eval_interval'),('evaluation_samples_per_bucket_per_seed','eval_samples')]:
  if argkey:assert getattr(a,argkey)==phase[key],(name,key)
 assert (str(a.resume)if a.resume else None)==phase['resume']
 assert cp['completed_updates']==phase['completed_updates'] and cp['common_step_counter']==24*phase['completed_updates']
 assert cp['cuda_rng_states']==2
 row=next(line for line in doc.splitlines()if line.startswith(f"| {phase['index']} |"));cells=[x.strip()for x in row.strip('|').split('|')]
 assert cells[1]==name and cells[2]==f"{previous}→{phase['completed_updates']}"
 assert int(re.match(r'\d+',cells[3])[0])==a.iterations
 if phase['index']>0:assert Path(phase['resume']).name in cells[5]
 assert int(re.match(r'\d+',cells[6])[0])==a.eval_interval
 if phase['index']>0:assert previous+a.iterations==phase['completed_updates']
 previous=phase['completed_updates'];checks.append(dict(phase=name,argv_manifest_table_and_checkpoint='PASS'))
assert previous==3762
result=dict(status='PARTIAL_REVIEW_NOT_LAUNCH_APPROVAL',document_sha256=hashlib.sha256((ROOT/'STRICT_3762_SINGLE_GPU.md').read_bytes()).hexdigest(),checks=checks,remaining=['materialized source and static resource identity','effective config re-instantiation per phase','compiler/runtime compatibility and original source fingerprints','deterministic update12 stop event validation','fresh corrected-collision qualification after document review'])
(ROOT/'document-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],phase_checks=len(checks),final_update=previous)))
