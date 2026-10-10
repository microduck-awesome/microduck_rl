"""Refresh selected, ancestry-bound saved actors with the current frozen evaluator."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from servo_bam.testbench.learning_evidence import FAMILIES, digest, require


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def recompute(report_path, protocol):
    report = json.loads(report_path.read_text())
    rule = protocol['convergence']
    require(report['passed'] and report['fault_count'] == 0, 'A refreshed evaluation did not pass')
    require([(t['seed'], t['trajectory_seed'], t['ticks'], t['failure']) for t in report['trials']] ==
            [(s, t, rule['validation_ticks'], None) for s, t in
             zip(rule['environment_seeds'], rule['trajectory_seeds'], strict=True)],
            'Incomplete or different validation trials')
    raw = report_path.parent / 'traces.npz'
    require(digest(raw) == report['raw_sha256'], 'Raw revalidation trace changed')
    errors, families, holds = [], [], []
    with np.load(raw, allow_pickle=False) as archive:
        for trial in range(len(report['trials'])):
            q = archive[f'q_target_trial{trial}']
            family = archive[f'family_trial{trial}']
            require(q.shape == (rule['validation_ticks'], rule['validation_envs'], 2)
                    and np.isfinite(q).all(), 'Invalid refreshed raw trace')
            require(np.array_equal(family, np.arange(rule['validation_envs']) % len(FAMILIES)),
                    'Different balanced trajectory families')
            errors.append(np.abs(q[:, :, 0] - q[:, :, 1]) * 180 / np.pi)
            families.append(family)
            holds.extend(np.std(q[rule['hold_scoring_start_tick']:, family == 0, 0]
                                * 180 / np.pi, axis=0).tolist())
    values = np.concatenate([e.reshape(-1) for e in errors])
    scores = {'mae_deg': float(values.mean()), 'p99_deg': float(np.quantile(values, .99)),
              'zero_hold_std_deg': float(np.mean(holds))}
    pairs = [(scores[k], report[k]) for k in scores]
    for i, name in enumerate(FAMILIES):
        values = np.concatenate([e[:, f == i].reshape(-1) for e, f in zip(errors, families, strict=True)])
        pairs.extend([(float(values.mean()), report['families'][name]['mae_deg']),
                      (float(np.quantile(values, .99)), report['families'][name]['p99_deg'])])
    difference = float(np.abs(np.asarray(pairs)[:, 0] - np.asarray(pairs)[:, 1]).max())
    require(difference <= 1e-6, 'Raw revalidation scores differ from the stored report')
    return difference


def verify_unvalidated_export(state, onnx):
    """Bind an existing interrupted export to its saved, unnormalized 4D actor."""
    import onnxruntime as ort
    actor = state['actor_state_dict']
    expected_keys = {f'mlp.{i}.{key}' for i in [0, 2, 4, 6] for key in ['weight', 'bias']}
    require(set(actor) == expected_keys | {'distribution.std_param'},
            'Unevaluated export has an unexpected or normalized actor architecture')
    widths = [4, 256, 128, 64, 1]
    layers = []
    for left, right in zip(widths, widths[1:]):
        layers.append(torch.nn.Linear(left, right))
        if right != 1:
            layers.append(torch.nn.ELU())
    model = torch.nn.Sequential(*layers).eval()
    model.load_state_dict({key.removeprefix('mlp.'): value for key, value in actor.items()
                           if key.startswith('mlp.')}, strict=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'], sess_options=options)
    lower = np.array([-1.5, -8., -2., -.8], dtype=np.float32)
    upper = -lower
    inputs = np.random.default_rng(91411).uniform(lower, upper, (512, 4)).astype(np.float32)
    corners = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    inputs[:16] = np.where(corners, upper, lower)
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs)).numpy()
    actual = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0]
                             for row in inputs], axis=0)
    difference = float(np.abs(expected - actual).max())
    require(np.isfinite(actual).all() and difference < 1e-5,
            'Interrupted export differs from the saved actor')
    return {'nodes': len(inputs), 'maximum_torch_onnx_difference': difference,
            'normalization': False, 'boundary_nodes': len(corners)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-report', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--select', nargs='+', required=True, help='Policy:seed:completed_iterations identities')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-unevaluated-history', action='store_true',
                        help='Complete a saved actor with no historical evaluation; retain that missing status')
    args = parser.parse_args()
    history = json.loads(args.history_report.read_text())
    require(history['passed'] and len(history['policies']) == 24, 'Full24 ancestry was not verified')
    for path, value in history['evidence_sha256'].items():
        require(digest(path) == value, 'Bound checkpoint ancestry changed: ' + path)
    protocol = json.loads(args.protocol.read_text())
    protocol_sha = digest(args.protocol)
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, value in frozen.items():
        require(digest(path) == value, 'Frozen current source/input changed: ' + path)
    validator = Path(__file__).with_name('sc0090_validate_checkpoint.py').resolve()
    validator_sha = digest(validator)
    require(validator_sha == protocol['source_sha256'][str(validator)], 'Different validation executor')
    selected, identities = [], set()
    for selection in args.select:
        policy, seed_text, completed_text = selection.split(':')
        seed, completed = int(seed_text), int(completed_text)
        identity = (policy, seed, completed)
        require(identity not in identities, 'Duplicate checkpoint selection')
        identities.add(identity)
        policies = [p for p in history['policies'] if p['policy'] == policy and p['seed'] == seed]
        require(len(policies) == 1, 'Missing policy/seed ancestry')
        rows = [r for r in policies[0]['history'] if r['completed_iterations'] == completed]
        require(len(rows) == 1, 'Missing historical checkpoint')
        row = rows[0]
        unevaluated = (row['passed'] is None and row.get('validation_state') == 'not_completed'
                       and row.get('validation_report') is None)
        require(row['passed'] is True or (args.allow_unevaluated_history and unevaluated),
                'Historical checkpoint was not passed; only an explicit missing evaluation can be completed')
        checkpoint = Path(row['checkpoint'])
        require(digest(checkpoint) == row['checkpoint_sha256'], 'Selected actor changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(state['iter'] == completed - 1 and state['infos']['policy'] == policy
                and state['infos']['seed'] == seed
                and state['infos']['training_protocol_sha256'] == row['protocol_sha256'],
                'Original checkpoint training identity changed')
        parity = None
        if unevaluated:
            original_report = None
            onnx = checkpoint.parent / f'evaluations/iteration{completed:04}/policy.onnx'
            require(onnx.is_file(), 'Interrupted actor export is missing; do not substitute another checkpoint')
            parity = verify_unvalidated_export(state, onnx)
        else:
            original_report = json.loads(Path(row['validation_report']).read_text())
            onnx = Path(row['validation_report']).parent.parent / 'policy.onnx'
            require(digest(onnx) == original_report['onnx_sha256'], 'Selected actor export changed')
        selected.append({'policy': policy, 'seed': seed, 'completed_iterations': completed,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': row['checkpoint_sha256'],
            'onnx': str(onnx), 'onnx_sha256': digest(onnx),
            'training_protocol': row['protocol'], 'training_protocol_sha256': row['protocol_sha256'],
            'original_validation_report': row['validation_report'],
            'original_validation_report_sha256': digest(row['validation_report']) if original_report else None,
            'original_validation_passed': row['passed'],
            'first_completed_validation_of_saved_actor': unevaluated,
            'interrupted_export_parity': parity})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'source.py')
    shutil.copyfile(validator, output / 'validator-source.py')
    shutil.copyfile(args.protocol, output / 'evaluation-protocol.json')
    dump(output / 'pre-run.json', {'timestamp': datetime.now(timezone.utc).isoformat(),
        'selections': selected, 'source_sha256': digest(__file__),
        'history_report': str(args.history_report.resolve()), 'history_report_sha256': digest(args.history_report),
        'evaluation_protocol_sha256': protocol_sha, 'validator_sha256': validator_sha,
        'original_checkpoint_training_identities_preserved': True,
        'historical_validation_records_modified': False,
        'previously_unevaluated_actors_selected': sum(row['first_completed_validation_of_saved_actor'] for row in selected),
        'hardware_executed': False, 'NN_training_performed': False})
    rows = []
    for selected_row in selected:
        row = dict(selected_row)
        name = f"{row['policy']}-seed{row['seed']}-iteration{row['completed_iterations']:04}"
        folder = output / name
        folder.mkdir()
        shutil.copyfile(row['onnx'], folder / 'policy.onnx')
        command = [sys.executable, str(validator), '--policy', row['policy'],
            '--checkpoint', row['checkpoint'], '--onnx', str(folder / 'policy.onnx'),
            '--protocol', str(args.protocol.resolve()), '--output', str(folder / 'validation')]
        dump(folder / 'command.json', {'command': command})
        with (folder / 'validation.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, 'Refresh execution failed; original logs retained')
        path = folder / 'validation/report.json'
        report = json.loads(path.read_text())
        require(report['source_sha256'] == validator_sha and report['protocol_sha256'] == protocol_sha
                and report['checkpoint_sha256'] == row['checkpoint_sha256']
                and report['onnx_sha256'] == row['onnx_sha256'] and report['policy'] == row['policy'],
                'Refreshed actor or executor identity differs')
        row.update(validation_report=str(path), validation_report_sha256=digest(path),
            raw_score_recomputation_difference_deg=recompute(path, protocol),
            evaluation_protocol_sha256=protocol_sha,
            **{key: report[key] for key in ['passed', 'fault_count', 'mae_deg', 'p99_deg', 'zero_hold_std_deg']})
        rows.append(row)
        dump(output / 'evaluations.json', rows)
        print(json.dumps({key: row[key] for key in ['policy', 'seed', 'completed_iterations', 'passed', 'mae_deg', 'p99_deg']}), flush=True)
    for path, value in frozen.items():
        require(digest(path) == value, 'Frozen source/input changed during revalidation: ' + path)
    dump(output / 'report.json', {'passed': True, 'checkpoints': rows,
        'evaluation_protocol_sha256': protocol_sha, 'validator_sha256': validator_sha,
        'source_sha256': digest(__file__), 'frozen_contract_files_unchanged': len(frozen),
        'original_checkpoint_training_identities_preserved': True,
        'historical_validation_records_modified': False,
        'previously_unevaluated_actors_completed': sum(row['first_completed_validation_of_saved_actor'] for row in rows),
        'formal_convergence_claimed': False, 'hardware_executed': False,
        'scope': 'Current-executor revalidation of exact ancestry-bound saved actors. Original training identities remain distinct; no weights or old scores are modified.'})


if __name__ == '__main__':
    main()
