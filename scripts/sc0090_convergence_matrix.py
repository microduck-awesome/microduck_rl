"""Train all24 with a common GPU budget and simulation-only stopping rule."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from servo_bam.testbench.convergence import assess


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    study = workspace/'sc0090-datasets/v5-testbench-sim2real-20261011'
    protocol = json.loads(args.protocol.read_text()); rules = protocol['convergence']
    root = study/'runs'/args.label; root.mkdir(exist_ok=False)
    shutil.copyfile(args.protocol, root/'training-protocol.json')
    shutil.copyfile(__file__, root/'matrix-source.py')
    jobs = [{'policy': f'P{i:02}', 'seed': seed, 'status': 'pending',
             'completed_iterations': 0, 'history': []} for seed in [42, 7, 1] for i in range(8)]
    state = {'stage': 'all24_gpu_convergence', 'device': protocol['device'],
             'num_envs': protocol['num_envs'], 'jobs': jobs, 'status': 'running',
             'common_completed_iterations': 0, 'current_plateaus': 0,
             'training_protocol': str(args.protocol.resolve()),
             'hardware_executed': False, 'sim2real_benefit_verified': False}
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat(); dump(root/'queue.json', state)
    save()
    for target in range(rules['matrix_block_iterations'], rules['maximum_iterations']+1, rules['matrix_block_iterations']):
        state['target_iterations'] = target; save()
        for job in jobs:
            name = job['policy']+'-seed'+str(job['seed'])+'-n'+str(protocol['num_envs'])
            output = root/f'iteration{target:04}'/name
            command = [sys.executable, str(Path(__file__).with_name('sc0090_train.py')),
                       '--stage', 'train', '--policy', job['policy'], '--seed', str(job['seed']),
                       '--num-envs', str(protocol['num_envs']), '--iterations', str(target),
                       '--device', protocol['device'], '--output', str(output),
                       '--training-protocol', str(args.protocol.resolve())]
            if job.get('checkpoint'):
                command += ['--resume-checkpoint', str(study/job['checkpoint'])]
            job['status'] = 'running'; job['current_output'] = str(output.relative_to(study)); save()
            output.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({'started': name, 'target_iterations': target}), flush=True)
            with (output.parent/(name+'.log')).open('w') as log:
                result = subprocess.run(command, cwd=workspace/'microduck_rl', stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                job['status'] = 'failed'; state['status'] = 'failed'; state['exit_code'] = result.returncode; save()
                raise SystemExit('Failed job; partial checkpoints, validation and logs retained: '+name)
            report = json.loads((output/'report.json').read_text())
            if not report['passed']: raise RuntimeError('Training/export numeric check failed')
            evaluations = json.loads((output/'convergence-evaluations.json').read_text())
            for evaluation in evaluations:
                evaluation['output'] = str(output.relative_to(study))
                job['history'].append(evaluation)
            job.update(status='block_completed', completed_iterations=target,
                       output=str(output.relative_to(study)),
                       checkpoint=str((output/report['checkpoint']).relative_to(study)),
                       plateau=assess(job['history'], rules))
            state['current_plateaus'] = sum(j.get('plateau', {}).get('plateau', False) for j in jobs)
            save()
            print(json.dumps({'completed': name, 'iterations': target, 'plateau': job['plateau']['plateau']}), flush=True)
        state['common_completed_iterations'] = target
        all_stable = all(j['plateau']['plateau'] for j in jobs)
        dump(root/f'stopping-decision-{target:04}.json', {
            'common_iterations': target, 'all24_plateau': all_stable,
            'policies': {j['policy']+'-seed'+str(j['seed']): j['plateau'] for j in jobs},
            'hardware_results_used': False})
        if all_stable:
            state['status'] = 'passed'; state['stop_reason'] = 'all24_simulation_plateau'
            for job in jobs: job['status'] = 'passed'
            # Publish only the common final matrix. Earlier checkpoints stay
            # available as diagnostics and are never hardware-selected.
            with (study/'policy-index.csv').open() as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
            for row in rows:
                job = next(j for j in jobs if j['policy'] == row['group'] and j['seed'] == int(row['seed']))
                manifest = json.loads((study/job['output']/'manifest.json').read_text())
                row.update(status='trained_pending_runtime_qualification', checkpoint=job['checkpoint'],
                           onnx=str(Path(job['output'])/'policy.onnx'), onnx_sha256=manifest['onnx_sha256'],
                           interaction_steps=manifest['train_interactions'], simulation_MAE_deg='')
            index = root/'policy-index.csv'
            with index.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            shutil.copyfile(index, study/'policy-index.csv')
            budget = {**protocol, 'iterations': target, 'steps_per_iteration': 24,
                      'interactions_per_policy': protocol['num_envs']*24*target,
                      'status': 'final_common_budget_after_predeclared_simulation_stopping_rule',
                      'selection': 'common final checkpoint, all24; no best-checkpoint or Sunday selection'}
            dump(study/'reports/frozen-training-budget.json', budget)
            save(); return
        save()
    state['status'] = 'budget_exhausted_without_convergence'
    state['stop_reason'] = 'resource_cap_is_not_convergence'; save()
    print(json.dumps({'status': state['status'], 'common_iterations': state['common_completed_iterations']}), flush=True)


if __name__ == '__main__': main()
