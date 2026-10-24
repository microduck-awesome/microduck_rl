"""Translate verified historical argv into isolated output paths, without numerical overrides."""
from pathlib import Path
import json,shlex,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762');SOURCES=BASE/'sources';QUAL=BASE/'qualification';FORMAL=BASE/'formal'
inv=json.loads((ROOT/'history-inventory.json').read_text());commands=json.loads((ROOT/'historical-launch-evidence.json').read_text())['commands'];rows=[]
for phase in inv['phases']:
 name=phase['name'];old=shlex.split(commands[name]['launch_command'].split(' > ')[0]);assert old[:2]==['scripts/v2_python.sh','scripts/train_v2_standup.py'];args=old[2:].copy();changes=[]
 for flag in ('--output','--preflight','--resume'):
  if flag not in args:continue
  index=args.index(flag)+1;before=args[index]
  if flag=='--output':after=FORMAL/name
  elif flag=='--preflight':after=QUAL/name/'batch/report.json'
  else:
   pieces=Path(before).parts;assert pieces[0]=='runs-v2';after=FORMAL.joinpath(*pieces[1:])
  args[index]=str(after);changes.append(dict(flag=flag,historical=before,replay=str(after)))
 # Reversing only these three path remaps must reproduce the original argv exactly.
 restored=args.copy()
 for change in changes:restored[restored.index(change['flag'])+1]=change['historical']
 assert restored==old[2:]
 source=SOURCES/name;original_train=source/'scripts/train_v2_standup.py';pre=json.loads((ROOT/'evidence'/name/'historical-preflight.json').read_text());head=phase['head_precision_recipe'];head_stage=head.get('stage',int(head['id'].rsplit('-',1)[1]))if head else None
 train=[str(original_train),*args]
 if phase['index']==0:train=[str(ROOT/'tools/historical_entry.py'),'--script',str(original_train),'--event-report',str(FORMAL/name/'historical-stop-event.json'),'--stop-after','12','--',*args]
 qhead=['--head-precision','--head-precision-stage',str(head_stage)]if head else[]
 smoke=[str(ROOT/'tools/qualify_phase.py'),'--task',phase['task'],'--device',phase['device'],'--num-envs','64','--iterations','5','--output-dir',str(QUAL/name/'smoke64'),'--report',str(QUAL/name/'smoke64/report.json'),*qhead]
 batch=[str(ROOT/'tools/qualify_phase.py'),'--task',phase['task'],'--device',phase['device'],'--num-envs',str(phase['num_envs']),'--iterations',str(pre['iterations']),'--output-dir',str(QUAL/name/'batch'),'--report',str(QUAL/name/'batch/report.json'),*qhead]
 if pre.get('terminal_bootstrap_samples', 0):
  batch.append('--force-timeouts')
 rows.append(dict(index=phase['index'],name=name,source=str(source),wrapper=str(source/'scripts/v2_python.sh'),historical_argv=old,replay_trainer_argv=[str(original_train),*args],launch_argv=train,path_remaps_only=changes,smoke_argv=smoke,batch_argv=batch,qualification_iterations=pre['iterations'],output=str(FORMAL/name),expected_completed_updates=phase['completed_updates'],expected_terminal_status='STOPPED'if phase['index']==0 else'ALLOCATION_COMPLETE',head_stage=head_stage,source_archive_sha256=phase['source_archive_sha256']))
plan=dict(schema=1,status='PLAN_REVIEW_REQUIRED',only_training_model_delta='robot.xml plus 32 collision meshes',device='cuda:1',gpu_training_processes=1,formal_root=str(FORMAL),qualification_root=str(QUAL),environment_contract=dict(unset=['WORLD_SIZE','RANK','LOCAL_RANK','CUDA_VISIBLE_DEVICES','MICRODUCK_WARM_START','TRITON_DEFAULT_FP_FUSION','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'],SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat',SC0090_V2_PYTHON='/home/l40-vm01/data2/experiments/microduck-v2-20261015/venv-p04/bin/python',wrapper_applied_after_environment_cleanup=True),phases=rows)
(ROOT/'execution-plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(dict(phases=len(rows),final_update=rows[-1]['expected_completed_updates'],path_only_argv_changes=True))
