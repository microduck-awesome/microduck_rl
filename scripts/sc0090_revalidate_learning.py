"""Re-evaluate bound training checkpoints with a repaired, unchanged-training validator."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from servo_bam.testbench.learning_evidence import digest, require
from servo_bam.testbench.learning_review import review_window


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Use a fresh output directory')
    history = json.loads(args.history_report.read_text())
    require(history['passed'] and len(history['policies']) == 24, 'Full24 ancestry was not verified')
    for path, sha in history['evidence_sha256'].items():
        require(digest(path) == sha, 'Bound historical evidence changed: ' + path)
    protocol = json.loads(args.protocol.read_text())
    current_sha = digest(args.protocol)
    require(protocol.get('evaluation_refresh_revision', {}).get('training_semantics_unchanged'),
            'The current revision is not an evaluator-only repair')
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen active file changed: ' + path)
    validator = Path(__file__).with_name('sc0090_validate_checkpoint.py').resolve()
    require(digest(validator) == protocol['source_sha256'][str(validator)], 'Unexpected evaluator source')
    selected = [row for row in history['policies'] if row['policy'] == args.policy and row['seed'] == args.seed]
    require(len(selected) == 1, 'Policy/seed must select one verified ancestry')
    iterations = args.iterations
    interval = protocol['convergence']['evaluation_interval']
    require(len(iterations) == protocol['convergence']['window_evaluations']
            and all(b - a == interval for a, b in zip(iterations, iterations[1:])),
            'Use a complete, ordered evaluation window')
    records = []
    for completed in iterations:
        matches = [row for row in selected[0]['history'] if row['completed_iterations'] == completed]
        require(len(matches) == 1 and matches[0]['passed'] is True, 'Missing original completed evaluation')
        row = matches[0]
        old_path = Path(row['protocol'])
        old = json.loads(old_path.read_text())
        require(digest(old_path) == row['protocol_sha256'], 'Historical training protocol changed')
        # Metadata and evaluator source may change; all training sources,
        # sampling settings and frozen inputs must retain their exact identity.
        for path, sha in old['source_sha256'].items():
            if Path(path).resolve() != validator:
                require(protocol['source_sha256'].get(path) == sha,
                        'Training source changed across the proposed window: ' + path)
        require(all(protocol['input_sha256'].get(path) == sha for path, sha in old['input_sha256'].items()),
                'A historical training input changed')
        for key in ['device', 'num_envs', 'steps_per_iteration', 'network', 'normalization',
                    'convergence', 'environment_resume', 'frozen_M6_sha256', 'frozen_O1_sha256']:
            require(old[key] == protocol[key], 'Training or validation settings changed: ' + key)
        checkpoint = Path(row['checkpoint'])
        require(digest(checkpoint) == row['checkpoint_sha256'], 'Historical checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(state['iter'] == completed - 1 and state['infos']['policy'] == args.policy
                and state['infos']['seed'] == args.seed
                and state['infos']['training_protocol_sha256'] == row['protocol_sha256'],
                'Saved training identity differs from its ancestry record')
        onnx = Path(row['validation_report']).parent.parent / 'policy.onnx'
        original_report = json.loads(Path(row['validation_report']).read_text())
        require(digest(onnx) == original_report['onnx_sha256'], 'Original export changed')
        records.append({'completed_iterations': completed, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': row['checkpoint_sha256'], 'onnx': str(onnx),
            'onnx_sha256': digest(onnx), 'training_protocol': str(old_path),
            'training_protocol_sha256': row['protocol_sha256'],
            'original_validation_report': row['validation_report'],
            'original_validation_report_sha256': digest(row['validation_report']),
            'reward_mean': row['reward_mean']})
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / 'source.py')
    shutil.copyfile(validator, output / 'validator-source.py')
    for name in ['learning_evidence.py', 'learning_review.py']:
        import servo_bam.testbench.learning_evidence as dependency
        shutil.copyfile(Path(dependency.__file__).with_name(name), output / name)
    shutil.copyfile(args.history_report, output / 'history-snapshot.json')
    shutil.copyfile(args.protocol, output / 'evaluation-protocol.json')
    pre = {'timestamp': datetime.now(timezone.utc).isoformat(), 'policy': args.policy, 'seed': args.seed,
        'history_report': str(args.history_report.resolve()), 'history_report_sha256': digest(args.history_report),
        'evaluation_protocol': str(args.protocol.resolve()), 'evaluation_protocol_sha256': current_sha,
        'validation_executor_sha256': digest(validator), 'source_sha256': digest(__file__),
        'unchanged_training_sources_verified': True, 'unchanged_training_inputs_verified': True,
        'original_checkpoint_training_identities_preserved': True, 'checkpoints': records,
        'automatic_convergence_decision': False, 'hardware_executed': False}
    dump(output / 'pre-run.json', pre)
    rows = []
    for record in records:
        completed = record['completed_iterations']
        folder = output / f'iteration{completed:04}'
        folder.mkdir()
        # The review helper expects the exact export beside the validation.
        shutil.copyfile(record['onnx'], folder / 'policy.onnx')
        command = [sys.executable, str(validator), '--policy', args.policy,
            '--checkpoint', record['checkpoint'], '--onnx', str(folder / 'policy.onnx'),
            '--protocol', str(args.protocol.resolve()), '--output', str(folder / 'validation')]
        dump(folder / 'command.json', {'command': command})
        with (folder / 'validation.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, 'Validation process failed; retained log: ' + str(folder))
        report_path = folder / 'validation/report.json'
        report = json.loads(report_path.read_text())
        require(report['passed'] and report['fault_count'] == 0
                and report['source_sha256'] == pre['validation_executor_sha256']
                and report['checkpoint_sha256'] == record['checkpoint_sha256']
                and report['onnx_sha256'] == record['onnx_sha256']
                and report['protocol_sha256'] == current_sha,
                'Revalidation did not pass with the declared actor and executor')
        row = {key: report[key] for key in ['policy', 'passed', 'fault_count', 'checkpoint_sha256',
                                           'protocol_sha256', 'mae_deg', 'p99_deg', 'zero_hold_std_deg']}
        row.update(seed=args.seed, completed_iterations=completed, checkpoint=record['checkpoint'],
            protocol=str(args.protocol.resolve()), validation_report=str(report_path),
            validation_report_sha256=digest(report_path), training_protocol=record['training_protocol'],
            training_protocol_sha256=record['training_protocol_sha256'], reward_mean=record['reward_mean'])
        rows.append(row)
        dump(output / 'evaluations.json', rows)
        print(json.dumps({'completed_revalidation': completed, 'passed': True,
                          **{key: report[key] for key in ['mae_deg', 'p99_deg', 'zero_hold_std_deg']}}), flush=True)
    review = review_window(rows, len(rows), 2048, 91141)
    require(review['measured'], 'A complete measured window was not produced')
    for path, sha in frozen.items():
        require(digest(path) == sha, 'Frozen active file changed during revalidation: ' + path)
    result = {'passed': True, 'policy': args.policy, 'seed': args.seed,
        'evaluation_protocol_sha256': current_sha, 'training_protocols_preserved': True,
        'checkpoints': rows, 'review': review, 'pre_run_sha256': digest(output / 'pre-run.json'),
        'source_sha256': digest(__file__), 'validator_sha256': digest(validator),
        'frozen_contract_files_unchanged': len(frozen), 'formal_convergence_claimed': False,
        'automatic_stopping_enabled': False, 'hardware_executed': False,
        'sim2real_benefit_verified': False,
        'scope': 'Matched current-executor simulation learning window. The protocol field in analysis rows identifies the evaluation contract; separate training_protocol fields retain the original saved checkpoint identity. No weights, training logs or historical scores are relabeled.'}
    dump(output / 'report.json', result)
    print(json.dumps({'passed': True, 'iterations': iterations,
        'formal_convergence_claimed': False, 'metrics': [
            {key: row[key] for key in ['metric', 'observed_change', 'slope_deg_per100iterations']}
            for row in review['measured_learning_changes']]}), flush=True)


if __name__ == '__main__':
    main()
