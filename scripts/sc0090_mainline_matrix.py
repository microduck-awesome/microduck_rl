"""Resume the full GPU matrix after the measured P04 read-contract correction."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    ws = Path(__file__).resolve().parents[2]
    study = ws/'sc0090-datasets/v5-testbench-sim2real-20261011'
    protocol = json.loads(args.protocol.read_text())
    rules = protocol['convergence']
    root = study/'runs'/args.label; root.mkdir(exist_ok=False)
    shutil.copyfile(args.protocol, root/'training-protocol.json')
    shutil.copyfile(__file__, root/'matrix-source.py')
    jobs = []
    for seed in protocol['seeds']:
        for policy in protocol['policy_groups']:
            previous = protocol['initial_checkpoint_inventory'].get(f'{policy}-seed{seed}', {})
            jobs.append({'policy': policy, 'seed': seed, 'status': 'pending',
                'completed_iterations': previous.get('completed_iterations', 0),
                'checkpoint': previous.get('checkpoint'), 'history': [],
                'inherited_training': previous, 'new_contract_iterations': 0})
    state = {'status': 'running', 'stage': 'mainline_read_contract_correction', 'jobs': jobs,
        'device': protocol['device'], 'num_envs': protocol['num_envs'],
        'training_protocol': str(args.protocol.resolve()), 'common_completed_iterations': 0,
        'current_plateaus': 0, 'automatic_convergence_decision': False,
        'convergence_note': 'Former arbitrary absolute cutoffs withdrawn; measured convergence assessment remains open. Resource cap does not imply success.',
        'hardware_executed': False, 'sim2real_benefit_verified': False}
    lock = threading.Lock()
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        dump(root/'queue.json', state)
    save()

    def run(job, target):
        previous = job['completed_iterations']
        if previous >= target: return
        name = f"{job['policy']}-seed{job['seed']}-n{protocol['num_envs']}"
        output = root/f'iteration{target:04}'/name
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).with_name('sc0090_train.py')),
            '--stage', 'train', '--policy', job['policy'], '--seed', str(job['seed']),
            '--num-envs', str(protocol['num_envs']), '--iterations', str(target),
            '--device', protocol['device'], '--output', str(output),
            '--training-protocol', str(args.protocol.resolve())]
        if job['checkpoint']: command += ['--resume-checkpoint', job['checkpoint']]
        with lock:
            job.update(status='running', current_output=str(output.relative_to(study)))
            save()
        print(json.dumps({'started': name, 'from_iterations': previous, 'target_iterations': target}), flush=True)
        with (output.parent/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ws/'microduck_rl', stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            with lock:
                job.update(status='failed', exit_code=result.returncode)
                state['status'] = 'failed'; save()
            raise RuntimeError('Training failed; retained checkpoints and logs: '+name)
        report = json.loads((output/'report.json').read_text())
        if not report['passed']: raise RuntimeError('Training/export verification failed: '+name)
        evaluations = json.loads((output/'convergence-evaluations.json').read_text())
        for evaluation in evaluations: evaluation['output'] = str(output.relative_to(study))
        with lock:
            job['history'].extend(evaluations)
            job.update(status='block_completed', completed_iterations=target,
                output=str(output.relative_to(study)), checkpoint=str(output/report['checkpoint']),
                new_contract_iterations=job['new_contract_iterations']+target-previous)
            save()
        print(json.dumps({'completed': name, 'iterations': target,
                          'last_validation_passed': evaluations[-1]['passed']}), flush=True)

    # The corrected P04 and the main-condition P00 are the immediate priority.
    priority = [next(j for j in jobs if j['policy'] == p and j['seed'] == 42) for p in ['P04', 'P00']]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, job, protocol['mainline_priority_iterations']) for job in priority]
        for future in as_completed(futures): future.result()
    # Reuse all saved PPO state; each later matrix stage reaches one common
    # iteration count. Sampler changes in inherited prefixes remain explicit.
    for target in range(rules['matrix_block_iterations'], rules['maximum_iterations']+1,
                        rules['matrix_block_iterations']):
        state['target_iterations'] = target; save()
        for job in jobs: run(job, target)
        state['common_completed_iterations'] = min(j['completed_iterations'] for j in jobs)
        save()
    state['status'] = 'resource_cap_reached_convergence_review_required'
    state['stop_reason'] = 'resource_cap_is_not_convergence_or_task_success'; save()


if __name__ == '__main__': main()
