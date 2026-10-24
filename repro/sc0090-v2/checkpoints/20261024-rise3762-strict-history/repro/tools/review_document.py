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

materialized=json.loads((ROOT/'materialized-sources.json').read_text())
plan=json.loads((ROOT/'execution-plan.json').read_text())
assert len(plan['phases'])==len(materialized['phases'])==10
for phase, built, row in zip(inv['phases'], materialized['phases'], plan['phases']):
 assert phase['name']==built['phase']==row['name']
 source=Path(built['path'])
 expected={**materialized['static_file_hashes'], **phase['source_files'], **built['added_meshes']}
 expected[built['only_modified_static_file']]=built['corrected_xml_sha256']
 for name,sha in expected.items():
  assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,(phase['name'],name)
 actual={str(f.relative_to(source)) for f in source.glob('src/**/*') if f.suffix in ('.py','.cpp')}
 assert actual=={n for n in phase['source_files'] if n.startswith('src/')}
 config=json.loads((ROOT/'config-checks'/f"{phase['name']}.json").read_text())
 assert config['status']=='PASS' and not config['differences']
 assert len(config['allowed_collision_differences'])==96
 assert config['triton_enable_fp_fusion']==(phase['index']<3)
 restored=row['replay_trainer_argv'][1:].copy()
 for change in row['path_remaps_only']:restored[restored.index(change['flag'])+1]=change['historical']
 assert restored==row['historical_argv'][2:]
 assert row['source']==built['path'] and row['expected_completed_updates']==phase['completed_updates']
 assert row['expected_terminal_status']==('STOPPED' if phase['index']==0 else 'ALLOCATION_COMPLETE')
 assert '--resume' not in row['launch_argv'] if phase['index']==0 else True
for name in ('config-review.json','collision-identity-review.json','stop-boundary-unit-check.json','materialized-sources.json'):
 assert json.loads((ROOT/name).read_text())['status']=='PASS',name
locked=[ROOT/'EXECUTION_REVIEW.md',ROOT/'execution-entry-review.json',ROOT/'STRICT_3762_SINGLE_GPU.md',ROOT/'execution-plan.json',ROOT/'history-inventory.json',ROOT/'materialized-sources.json',ROOT/'historical-launch-evidence.json',ROOT/'checkpoint-history.json',ROOT/'config-review.json',ROOT/'collision-identity-review.json',ROOT/'stop-boundary-unit-check.json',ROOT/'qualification-harness-provenance.json',*sorted((ROOT/'tools').glob('*.py')),*sorted((ROOT/'config-checks').glob('*.json'))]
result=dict(status='PASS_FOR_QUALIFICATION_ONLY',document_sha256=hashlib.sha256((ROOT/'STRICT_3762_SINGLE_GPU.md').read_bytes()).hexdigest(),checks=checks,locked_files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in locked},formal_training_permitted=False,remaining_formal_gates=['fresh corrected-collision smoke64 and actual8192 reports; original throughput threshold unchanged','real GPU environment update12 SIGTERM boundary and absence of extra eval','native parent checkpoint and curriculum prerequisites per segment'])
(ROOT/'document-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],phase_checks=len(checks),final_update=previous)))
