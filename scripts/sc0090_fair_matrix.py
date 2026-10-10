"""Advance the least-trained SC0090 policies in synchronized checkpoint rounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers not in [1, 2]: raise ValueError('Only qualified one/two-process GPU execution is supported')
    ws = Path(__file__).resolve().parents[2]
    study = ws/'sc0090-datasets/v5-testbench-sim2real-20261011'
    protocol = json.loads(args.protocol.read_text()); interval = protocol['convergence']['evaluation_interval']
    for path, value in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if digest(path) != value: raise RuntimeError('Frozen training contract changed: '+path)
    initial = json.loads(args.initial_state.read_text())
    jobs = initial['jobs']
    expected = {(policy, seed) for policy in protocol['policy_groups'] for seed in protocol['seeds']}
    if {(job['policy'], job['seed']) for job in jobs} != expected or len(jobs) != 24:
        raise RuntimeError('Fair scheduling requires exactly the full24 identities')
    for job in jobs:
        if job['completed_iterations'] % interval: raise RuntimeError('Inherited progress is not a saved evaluation boundary')
        if job.get('checkpoint') and digest(job['checkpoint']) != job['checkpoint_sha256']:
            raise RuntimeError('Inherited checkpoint identity changed')
        job.update(status='pending', history=[])
    root = study/'runs'/args.label; root.mkdir(exist_ok=False)
    shutil.copyfile(args.protocol, root/'training-protocol.json')
    shutil.copyfile(args.initial_state, root/'initial-state.json')
    shutil.copyfile(__file__, root/'matrix-source.py')
    dump(root/'pre-run.json', {'source_sha256': digest(__file__), 'protocol_sha256': digest(args.protocol),
        'initial_state_sha256': digest(args.initial_state), 'workers': args.workers,
        'round_iterations': interval, 'scheduler': 'Only jobs below minimum_completed+evaluation_interval can advance; synchronize the entire cohort before the next round.',
        'hardware_executed': False})
    state = {'status': 'running', 'stage': 'fair24_checkpoint_rounds', 'jobs': jobs,
        'device': protocol['device'], 'num_envs': protocol['num_envs'],
        'workers': args.workers, 'round_iterations': interval,
        'training_protocol': str(args.protocol.resolve()),
        'initial_leading_iterations': max(job['completed_iterations'] for job in jobs),
        'leader_advance_until_lowest_catches_up': False,
        'current_plateaus': 0, 'automatic_convergence_decision': False,
        'formal_convergence_claimed': False, 'hardware_executed': False,
        'sim2real_benefit_verified': False, 'initialization_decision': initial['decision']}
    lock = threading.Lock()
    def save():
        values = [job['completed_iterations'] for job in jobs]
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),
            common_completed_iterations=min(values), leading_completed_iterations=max(values),
            progress_spread_iterations=max(values)-min(values))
        dump(root/'queue.json', state)
    save()

    def run(job, target):
        previous = job['completed_iterations']
        if target-previous != interval: raise RuntimeError('Fair job would skip a synchronization round')
        name = f"{job['policy']}-seed{job['seed']}-n{protocol['num_envs']}"
        output = root/f'iteration{target:04}'/name; output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).with_name('sc0090_train.py')), '--stage', 'train',
            '--policy', job['policy'], '--seed', str(job['seed']), '--num-envs', str(protocol['num_envs']),
            '--iterations', str(target), '--device', protocol['device'], '--output', str(output),
            '--training-protocol', str(args.protocol.resolve())]
        if job.get('checkpoint'): command += ['--resume-checkpoint', job['checkpoint']]
        with lock:
            job.update(status='running', current_output=str(output.relative_to(study))); save()
        print(json.dumps({'started': name, 'from_iterations': previous, 'target_iterations': target}), flush=True)
        with (output.parent/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ws/'microduck_rl', stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            with lock: job.update(status='failed', exit_code=result.returncode); save()
            raise RuntimeError('Training failed; checkpoints and raw validation retained: '+name)
        report = json.loads((output/'report.json').read_text())
        manifest = json.loads((output/'manifest.json').read_text())
        if not report['passed'] or report['iterations'] != target or manifest['num_envs'] != protocol['num_envs']:
            raise RuntimeError('Completed job does not match the fair matrix budget')
        for file, key in [('policy.onnx', 'onnx_sha256'), (report['checkpoint'], 'checkpoint_sha256')]:
            if digest(output/file) != manifest[key]: raise RuntimeError('Completed actor hash changed')
        evaluations = json.loads((output/'convergence-evaluations.json').read_text())
        if len(evaluations) != 1 or evaluations[0]['completed_iterations'] != target:
            raise RuntimeError('Expected exactly one checkpoint evaluation per fair scheduling round')
        with lock:
            evaluations[0]['output'] = str(output.relative_to(study)); job['history'].extend(evaluations)
            job.update(status='round_completed', completed_iterations=target,
                checkpoint=str(output/report['checkpoint']), checkpoint_sha256=manifest['checkpoint_sha256'],
                output=str(output.relative_to(study)), new_contract_iterations=job.get('new_contract_iterations', 0)+interval)
            save()
        print(json.dumps({'completed': name, 'iterations': target, 'validation_passed': evaluations[0]['passed']}), flush=True)

    try:
        maximum = protocol['convergence']['maximum_iterations']
        while min(job['completed_iterations'] for job in jobs) < maximum:
            target = min(job['completed_iterations'] for job in jobs)+interval
            cohort = [job for job in jobs if job['completed_iterations'] < target]
            state['target_iterations'] = target; save()
            # Fresh, corrected O1/R1 variants were previously at zero and
            # have completed a dedicated smoke; fill those missing routes first.
            cohort.sort(key=lambda job: (job['seed'] != 42, job['policy'] not in ['P06', 'P07'], job['seed'], job['policy']))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run, job, target) for job in cohort]
                for future in as_completed(futures): future.result()
            if any(job['completed_iterations'] < target for job in jobs): raise RuntimeError('A synchronization round was incomplete')
            dump(root/f'round-{target:04}.json', {'all24_at_least_iterations': target,
                'leading_iterations': state['leading_completed_iterations'],
                'formal_convergence_claimed': False, 'hardware_results_used': False})
        state.update(status='resource_cap_reached_convergence_review_required',
            stop_reason='Common resource cap is not convergence or task success'); save()
    except Exception as exc:
        state.update(status='failed', failure=type(exc).__name__+': '+str(exc)); save(); raise


if __name__ == '__main__': main()
