"""Adopt live priority continuations before resuming the unchanged full24 protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def process(pid):
    path = Path('/proc')/str(pid)
    if not path.exists(): return None
    try:
        command = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        status = (path/'status').read_text()
    except FileNotFoundError:
        return None
    state = next(row for row in status.splitlines() if row.startswith('State:'))
    # An exited child of the held supervisor can remain as an unreaped zombie.
    if not command or '\tZ' in state: return None
    return {'command': command, 'state': state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--previous-queue', type=Path, required=True)
    parser.add_argument('--previous-supervisor-pid', type=int, required=True)
    parser.add_argument('--pending-runs', type=Path, required=True)
    parser.add_argument('--migration-audit', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    ws = Path(__file__).resolve().parents[2]
    study = ws/'sc0090-datasets/v5-testbench-sim2real-20261011'
    protocol = json.loads(args.protocol.read_text())
    migration = json.loads(args.migration_audit.read_text())
    if not migration['passed'] or not migration['R0_duration_veto_was_unreachable'] or migration['R1_inherited_prefixes_equivalent']:
        raise RuntimeError('Matched migration requires proved R0 equivalence and fresh R1 sampling')
    if migration['current_stream_source_sha256'] != sha(ws/'microduck_rl/src/mjlab_microduck/sc0090/stream.py'):
        raise RuntimeError('Read-equivalence audit no longer matches the sampler')
    for path, value in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha(path) != value: raise RuntimeError('Frozen training contract changed: '+path)
    previous = process(args.previous_supervisor_pid)
    if not previous or 'sc0090_mainline_matrix.py' not in previous['command'] or 'T' not in previous['state']:
        raise RuntimeError('Previous supervisor must be positively identified and held during adoption')
    state = json.loads(args.previous_queue.read_text())
    if state['training_protocol'] != str(args.protocol.resolve()): raise RuntimeError('Protocol mismatch')
    jobs = state['jobs']; pending = json.loads(args.pending_runs.read_text())['runs']
    root = study/'runs'/args.label; root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root/'matrix-source.py')
    shutil.copyfile(args.protocol, root/'training-protocol.json')
    dump(root/'pre-run.json', {'source_sha256': sha(__file__), 'protocol_sha256': sha(args.protocol),
        'previous_queue_sha256': sha(args.previous_queue), 'pending_runs_sha256': sha(args.pending_runs),
        'matched_migration_audit_sha256': sha(args.migration_audit),
        'previous_supervisor': previous, 'sampling_or_PPO_contract_changed': False,
        'scope': 'Preserve live mainline continuations. Reuse proved-equivalent R0 progress; preserve P04 as diagnosis and start all formal R1 policies at zero with the corrected read contract.',
        'hardware_executed': False})
    state.update(status='adopting_live_mainline_runs', stage='mainline_priority_then_full24',
        supervisor_pid=os.getpid(), previous_supervisor_pid=args.previous_supervisor_pid,
        matched_migration_audit=str(args.migration_audit.resolve()),
        R1_prior_policies_scope='diagnostic_only; formal corrected R1 matrix restarts from zero')
    for job in jobs: job['formal_restart_required'] = int(job['policy'][1:]) >= 4
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat(); dump(root/'queue.json', state)
    for item in pending:
        job = next(j for j in jobs if (j['policy'], j['seed']) == (item['policy'], item['seed']))
        live = process(item['pid'])
        if live and str(Path(item['output']).resolve()) not in live['command']:
            raise RuntimeError('Pending training process identity changed')
        job.update(status='running', current_output=str(Path(item['output']).relative_to(study)),
                   adopted_training_pid=item['pid'], adoption_target_iterations=item['target_iterations'])
    save()

    def ingest(job, output, previous_iterations, target):
        report = json.loads((output/'report.json').read_text())
        manifest = json.loads((output/'manifest.json').read_text())
        if not report['passed'] or report['iterations'] != target or manifest['stage'] != 'train':
            raise RuntimeError('Completed training report does not match adoption')
        if (manifest['policy'], manifest['seed'], manifest['num_envs']) != (job['policy'], job['seed'], protocol['num_envs']):
            raise RuntimeError('Actor identity differs from the matrix job')
        if manifest['training_protocol_sha256'] != sha(args.protocol): raise RuntimeError('Actor protocol differs')
        for file, key in [('policy.onnx', 'onnx_sha256'), (report['checkpoint'], 'checkpoint_sha256')]:
            if sha(output/file) != manifest[key]: raise RuntimeError('Actor artifact hash changed')
        history = json.loads((output/'convergence-evaluations.json').read_text())
        existing = {item['completed_iterations'] for item in job['history']}
        for item in history:
            if item['completed_iterations'] in existing: raise RuntimeError('Duplicate adopted validation iteration')
            item['output'] = str(output.relative_to(study)); job['history'].append(item)
        job.update(status='block_completed', completed_iterations=target,
            checkpoint=str(output/report['checkpoint']), output=str(output.relative_to(study)),
            new_contract_iterations=job['new_contract_iterations']+target-previous_iterations)

    try:
        waiting = list(pending)
        while waiting:
            for item in waiting.copy():
                output = Path(item['output']); live = process(item['pid'])
                report = output/'report.json'
                if live and str(output.resolve()) not in live['command']: raise RuntimeError('Adopted process changed identity')
                if report.is_file() and not live:
                    job = next(j for j in jobs if (j['policy'], j['seed']) == (item['policy'], item['seed']))
                    ingest(job, output, item['from_iterations'], item['target_iterations'])
                    waiting.remove(item); save()
                    print(json.dumps({'adopted': job['policy'], 'iterations': job['completed_iterations']}), flush=True)
                elif not live:
                    raise RuntimeError('Adopted training stopped without complete export/report: '+item['policy'])
            if waiting: time.sleep(5)
        previous = process(args.previous_supervisor_pid)
        if not previous or 'sc0090_mainline_matrix.py' not in previous['command'] or 'T' not in previous['state']:
            raise RuntimeError('Held scheduling supervisor changed before final adoption')
        os.kill(args.previous_supervisor_pid, signal.SIGTERM)
        os.kill(args.previous_supervisor_pid, signal.SIGCONT)
        obsolete = json.loads(args.previous_queue.read_text())
        obsolete.update(status='superseded_mainline_priority_adoption', adopted_queue=str(root/'queue.json'))
        dump(args.previous_queue, obsolete)
        # The duration veto was reachable in R1. Unequal inherited prefixes
        # would confound the paired S/O/R experiment, so preserve them only as
        # diagnostics. R0's complete eligible donor set proves equivalence.
        for job in jobs:
            if job['formal_restart_required']:
                job['retained_diagnostic_training'] = {key: job.get(key) for key in
                    ['checkpoint', 'completed_iterations', 'history', 'output', 'inherited_training', 'new_contract_iterations']}
                job.update(status='pending', completed_iterations=0, checkpoint=None,
                           history=[], new_contract_iterations=0)
                job.pop('current_output', None); job.pop('output', None)
                job['formal_restart_required'] = False
        state['status'] = 'running'; save()
        rules = protocol['convergence']
        for target in range(rules['matrix_block_iterations'], rules['maximum_iterations']+1, rules['matrix_block_iterations']):
            state['target_iterations'] = target; save()
            for job in jobs:
                if job['completed_iterations'] >= target: continue
                previous_iterations = job['completed_iterations']
                name = f"{job['policy']}-seed{job['seed']}-n{protocol['num_envs']}"
                output = root/f'iteration{target:04}'/name; output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(Path(__file__).with_name('sc0090_train.py')), '--stage', 'train',
                    '--policy', job['policy'], '--seed', str(job['seed']), '--num-envs', str(protocol['num_envs']),
                    '--iterations', str(target), '--device', protocol['device'], '--output', str(output),
                    '--training-protocol', str(args.protocol.resolve())]
                if job['checkpoint']: command += ['--resume-checkpoint', job['checkpoint']]
                job.update(status='running', current_output=str(output.relative_to(study))); save()
                print(json.dumps({'started': name, 'from_iterations': previous_iterations, 'target_iterations': target}), flush=True)
                with (output.parent/(name+'.log')).open('w') as log:
                    result = subprocess.run(command, cwd=ws/'microduck_rl', stdout=log, stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError('Training failed; partial checkpoints retained: '+name)
                ingest(job, output, previous_iterations, target); save()
            state['common_completed_iterations'] = min(j['completed_iterations'] for j in jobs); save()
        state.update(status='resource_cap_reached_convergence_review_required', stop_reason='resource_cap_is_not_convergence_or_task_success'); save()
    except Exception as exc:
        state.update(status='failed', failure=type(exc).__name__+': '+str(exc)); save(); raise


if __name__ == '__main__': main()
