"""Snapshot committed formal PPO control counts, preserving unresolved coverage."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch

ROOT = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015')
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
OUTPUT=args.output
assert not OUTPUT.exists()
torch.set_num_threads(2)
cache = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def checkpoint_info(path):
    key = str(path.resolve())
    if key not in cache:
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        info = loaded.get('infos', {}).get('v2', {})
        assert 'completed_updates' in info, ('missing actual parent iteration', path)
        cache[key] = dict(completed_updates=int(info['completed_updates']),
                          sha256=sha(path), task=info.get('task'), controls=info.get('controls'))
        del loaded
    return cache[key]

paths = [p.resolve() for branch in ['baseline', 'p04'] for p in (ROOT / branch).glob('*/run.json')]
runs = {p.parent: read(p) for p in paths}

def parent_path(directory, run):
    lineage = run.get('lineage') or {}
    value = lineage.get('checkpoint') or run.get('resume') or run.get('arguments', {}).get('resume') or run.get('arguments', {}).get('warm_start')
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts[0] == 'runs-v2':
        return (directory.parent / Path(*path.parts[1:])).resolve()
    raise ValueError(('ambiguous relative checkpoint', directory, value))

def route(directory, visited=()):
    assert directory not in visited, ('lineage cycle', directory)
    if directory not in runs:
        return 'unresolved_external_parent'
    run = runs[directory]
    parent = parent_path(directory, run)
    if parent is None:
        if directory.parent == ROOT / 'baseline':
            return 'baseline_rooted_including_p04_adaptation'
        if directory.parent == ROOT / 'p04':
            return 'p04_random_root_including_early_legacy_runs'
        return 'unresolved_root'
    return route(parent.parent, (*visited, directory))

records, exclusions = [], []
for directory, run in runs.items():
    status_path = directory / 'status.json'
    arguments = run.get('arguments', {})
    num_envs = run.get('num_envs', arguments.get('num_envs'))
    allocation_updates = run.get('allocation_updates', arguments.get('iterations'))
    if run.get('allocation_kind') == 'validation' or arguments.get('validation_only') or run.get('validation_only'):
        exclusions.append(dict(directory=str(directory), reason='explicit validation allocation, independent of environment count'))
        continue
    if num_envs is None or allocation_updates is None:
        exclusions.append(dict(directory=str(directory), reason='unresolved accounting evidence', detail='missing explicit N or allocation length'))
        continue
    if num_envs < 8192 or not status_path.exists():
        exclusions.append(dict(directory=str(directory), reason='small-N validation' if num_envs < 8192 else 'no committed training status; diagnostic/copied/failed startup is not presumed formal training'))
        continue
    status_bytes=status_path.read_bytes()
    status=json.loads(status_bytes)
    status_captured_unix=time.time()
    try:
        parent = parent_path(directory, run)
        parent_info = checkpoint_info(parent) if parent is not None else None
        if arguments.get('warm_start'):
            initialization = read(directory/'initialization.json')
            assert initialization['warm_start'] and initialization['new_skill_clock'] == 0
            assert parent_info is not None and parent_info['task'] != run['task']
            start = 0
        else:
            start = parent_info['completed_updates'] if parent_info else 0
        completed = int(status['completed_updates'])
        n, t = int(num_envs), int(run['agent']['num_steps_per_env'])
        actual = (completed-start)*n*t
        assert 0 <= completed-start <= allocation_updates
        if 'allocation_controls' in status:
            assert actual == status['allocation_controls'], directory
        inherited = status.get('inherited_controls')
        if inherited is not None:
            assert status['controls'] == inherited + actual
        pid = status.get('pid')
        proc_path = Path(f'/proc/{pid}/cmdline') if pid else None
        command = proc_path.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc_path and proc_path.exists() else ''
        live = any(name in command for name in ('train_v2_standup.py', 'train_v2_walking.py', 'train_v2_sitstand.py','train_v2_velstand.py','train_v2_groundpick.py','train_v2_ground_pick.py','train_v2_roulade.py','train_v2_ball_kick.py','train_v2_ballkick.py')) and directory.name in command
        live_evidence = 'direct trainer argv and allocation directory' if live else None
        # Camera workers run the trainer in-process to reuse expensive imports.
        # Verify the actual /proc argv plus the fixed arm/output mapping; a
        # status PID or pipeline marker by itself is never proof of liveness.
        for camera_arm in ('mild', 'control'):
            if directory.name != f'velstand-camera-{camera_arm}-allocation-001':
                continue
            worker = ROOT/'p04/run-velstand-camera-arm-001.py'
            marker = read(ROOT/f'p04/velstand-camera-{camera_arm}-pipeline-status.json')
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            expected_tail = [str(worker).encode(), camera_arm.encode(), b'']
            if argv[1:] == expected_tail and marker['pid'] == pid and marker['status'] == 'FORMAL_TRAINING':
                live = True
                live_evidence = dict(kind='verified live in-process trainer wrapper', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        roll_workers = {
            'roulade-teacher-allocation-001': ('run-roulade-teacher-formal-002.py', 'roulade-teacher-formal-pipeline-status.json', 'FORMAL_TRAINING'),
            'roulade-control-allocation-001': ('run-roulade-control-pair-001.py', 'roulade-control-pipeline-status.json', 'ROOT_PAIR_MARKER_ACCEPTED_STARTING_FORMAL'),
            'velstand-head-free-bc-allocation-001': ('run-velstand-head-free-bc-001.py', 'velstand-head-free-bc-pipeline-status-001.json', 'FORMAL_TRAINING'),
            'velstand-head-bias-allocation-001': ('run-velstand-head-bias-001.py', 'velstand-head-bias-pipeline-status-001.json', 'FORMAL_TRAINING'),
        }
        if directory.name in roll_workers and status['status'] == 'RUNNING':
            script, marker_name, expected_phase = roll_workers[directory.name]
            worker = ROOT/'p04'/script
            marker = read(ROOT/'p04'/marker_name)
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if argv[1:] == [str(worker).encode(), b''] and marker['pid'] == pid and marker['status'] == expected_phase:
                live = True
                live_evidence = dict(kind='verified live in-process bounded trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        if directory.name in {f'velstand-head-bias-allocation-{i:03d}' for i in (2,3,4)} and status['status']=='RUNNING':
            worker=ROOT/'p04/run-velstand-head-bias-learning-002.py'
            marker=read(ROOT/'p04/velstand-head-bias-learning-status-002.json')
            argv=proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if (argv[1:]==[str(worker).encode(),b''] and marker['pid']==pid
                    and marker['status']=='FORMAL_TRAINING' and marker['allocation']==directory.name):
                live=True
                live_evidence=dict(kind='verified live fixed-recipe head learning program',argv=command,
                    worker_sha256=sha(worker),pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        sit_workers = {
            'sitstand-transition-release-allocation-001': (('run-sitstand-transition-release-001.py','resume-sitstand-transition-release-002.py'), 'sitstand-transition-release-pipeline-status.json'),
            'sitstand-fresh-release-allocation-001': (('run-sitstand-fresh-release-001.py',), 'sitstand-fresh-release-pipeline-status.json'),
            'sitstand-history-base-allocation-001': (('run-sitstand-history-base-001.py','run-sitstand-history-base-002.py'), 'sitstand-history-base-pipeline-status.json'),
            'sitstand-history-base-allocation-002': (('run-sitstand-history-base-002.py',), 'sitstand-history-base-pipeline-status.json'),
            'sitstand-history-hold320-allocation-001': (('run-sitstand-history-hold320-001.py',), 'sitstand-history-hold320-pipeline-status.json'),
            'sitstand-near-teacher-lr3e6-allocation-001': (('run-sitstand-near-teacher-lr3e6-001.py',), 'sitstand-near-teacher-lr3e6-pipeline-status.json'),
        }
        if directory.name in sit_workers and status['status'] == 'RUNNING':
            names, marker_name = sit_workers[directory.name]
            marker = read(ROOT/'p04'/marker_name)
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            approved_workers = [ROOT/'p04'/name for name in names]
            worker = next((p for p in approved_workers if argv[1:] == [str(p).encode(), b'']), None)
            if (worker is not None and marker['pid'] == pid
                    and marker['status'] == 'RUNNING' and marker.get('phase') == 'allocation'):
                live = True
                live_evidence = dict(kind='verified live bounded SitStand trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        if directory.name == 'roulade-history-formal-001' and status['status'] == 'RUNNING':
            worker = ROOT/'p04/run-roulade-history-001.py'
            marker = read(ROOT/'p04/roulade-history-status-001.json')
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if (argv[1:] == [str(worker).encode(), b''] and marker['pid'] == pid
                    and marker['status'] == 'RUNNING' and marker.get('phase') == 'formal'
                    and Path(marker['output']).resolve() == directory):
                live = True
                live_evidence = dict(kind='verified live continuous historical Roulade trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        if directory.name == 'ballkick-health-history-allocation-001' and status['status'] == 'RUNNING':
            worker = ROOT/'p04/run-ballkick-health-history-001.py'
            marker = read(ROOT/'p04/ballkick-health-history-pipeline-status.json')
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if (argv[1:] == [str(worker).encode(), b''] and marker['pid'] == pid
                    and marker['status'] == 'RUNNING' and marker.get('phase') == 'allocation'
                    and Path(marker['output']).resolve() == directory):
                live = True
                live_evidence = dict(kind='verified live continuous healthy BallKick trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        if directory.name == 'ballkick-target-precision-allocation-001' and status['status'] == 'RUNNING':
            worker = ROOT/'p04/run-ballkick-target-precision-001.py'
            marker = read(ROOT/'p04/ballkick-target-precision-pipeline-status.json')
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if (argv[1:] == [str(worker).encode(), b''] and marker['pid'] == pid
                    and marker['status'] == 'RUNNING' and marker.get('phase') == 'allocation'
                    and Path(marker['output']).resolve() == directory):
                live = True
                live_evidence = dict(kind='verified live continuous mature target-precision BallKick trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        if directory.name == 'groundpick-health-history-allocation-001' and status['status'] == 'RUNNING':
            worker = ROOT/'p04/run-groundpick-health-history-001.py'
            marker = read(ROOT/'p04/groundpick-health-history-pipeline-status.json')
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if (argv[1:] == [str(worker).encode(), b''] and marker['pid'] == pid
                    and marker['status'] == 'RUNNING' and marker.get('phase') == 'allocation'
                    and Path(marker['output']).resolve() == directory):
                live = True
                live_evidence = dict(kind='verified live continuous healthy GroundPick trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        if directory.name == 'groundpick-target-precision-allocation-001' and status['status'] == 'RUNNING':
            worker = ROOT/'p04/run-groundpick-target-precision-002.py'
            marker = read(ROOT/'p04/groundpick-target-precision-pipeline-status-002.json')
            argv = proc_path.read_bytes().split(b'\0') if proc_path and proc_path.exists() else []
            if (argv[1:] == [str(worker).encode(), b''] and marker['pid'] == pid
                    and marker['status'] == 'RUNNING' and marker.get('phase') == 'allocation'
                    and Path(marker['output']).resolve() == directory):
                live = True
                live_evidence = dict(kind='verified live mature GroundPick height precision trainer', argv=command,
                    worker_sha256=sha(worker), pipeline_marker=marker,
                    process_start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])
        terminal_record = status['status'] in ['ALLOCATION_COMPLETE','STOPPED']
        skill = ('velstand' if 'VelStand' in run['task'] else 'groundpick' if 'GroundPick' in run['task'] else 'ballkick' if 'BallKick' in run['task'] else 'roulade' if 'Roulade' in run['task'] else 'walking' if 'Velocity' in run['task'] else 'sitstand' if 'SitStand' in run['task'] else 'rise_stand' if 'StandUp' in run['task'] else 'other')
        record = dict(directory=str(directory), route=route(directory), skill=skill, task=run['task'],
            status=status['status'], live_training_process_verified=live, live_training_evidence=live_evidence, pid=pid,
            num_envs=n, steps_per_env=t, start_update=start, committed_end_update=completed,
            committed_formal_controls=actual,
            control_scope='terminal allocation count' if terminal_record else 'committed status lower bound; may omit updates since last status',
            inherited_selected_lineage_controls=inherited, selected_lineage_controls=status.get('controls'),
            parent_checkpoint=str(parent) if parent else None, parent_checkpoint_info=parent_info,
            run_sha256=sha(directory/'run.json'), status_sha256=hashlib.sha256(status_bytes).hexdigest(),
            status_captured_unix=status_captured_unix, status_snapshot=status)
        wall_path=directory/'actual-wall-time.json'
        if wall_path.exists():
            record['existing_wall_report']=dict(path=str(wall_path), sha256=sha(wall_path), data=read(wall_path))
        records.append(record)
    except (AssertionError,ValueError,KeyError,FileNotFoundError) as exc:
        exclusions.append(dict(directory=str(directory),reason='unresolved accounting evidence',detail=str(exc)))

totals={}
for record in records:
    category=totals.setdefault(record['route'],dict(allocations=0,committed_formal_controls=0,nonterminal_allocations=0))
    category['allocations']+=1
    category['committed_formal_controls']+=record['committed_formal_controls']
    category['nonterminal_allocations']+=record['status'] not in ['ALLOCATION_COMPLETE','STOPPED']
by_skill = {}
for record in records:
    category = by_skill.setdefault(record['skill'], dict(allocations=0, committed_formal_controls=0))
    category['allocations'] += 1
    category['committed_formal_controls'] += record['committed_formal_controls']
result=dict(schema=13,snapshot_unix=time.time(),source_sha256=sha(Path(__file__)),
    scope='Direct artifact-root run/status records with >=8192 environments; Same-task start updates read from real parent metadata; cross-task warm starts require explicit initialization clock0, never infer from filenames. Each output directory counted once; parent controls are not added again.',
    limitation='Committed formal controls only, not all project GPU work or wall time. Preflight, evaluation, missing-status/copy diagnostics and unresolved records excluded explicitly. Early legacy P04 roots included; this scope differs from the dedicated from-zero agent lineage. Nonterminal status can lag optimizer progress.',
    totals=totals,by_skill=by_skill,records=records,exclusions=exclusions,checkpoint_metadata=cache)
OUTPUT.write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(dict(totals=totals,exclusions=exclusions),indent=2))
