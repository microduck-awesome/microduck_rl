"""Measure actual validator repeatability with only the residual draw changed."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def resolve_actor(args, protocol):
    """Keep training identity distinct from the contract used for evaluation."""
    current_sha = digest(args.protocol)
    history = None
    if args.history_report:
        history = json.loads(args.history_report.read_text())
        expected = {(p, seed) for p in protocol['policy_groups'] for seed in protocol['seeds']}
        if not (history['passed'] and history['active_protocol_sha256'] == current_sha
                and len(history['policies']) == 24
                and {(p['policy'], p['seed']) for p in history['policies']} == expected):
            raise ValueError('A saved actor requires verified full24 ancestry under the current contract')
        for path, sha in history['evidence_sha256'].items():
            if digest(path) != sha:
                raise ValueError('Bound training ancestry changed: ' + path)
    manifest = None
    manifest_path = None
    if args.policy_dir:
        if args.onnx:
            raise ValueError('--onnx is only used with --checkpoint')
        folder = args.policy_dir.resolve()
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        checkpoint = folder / f"model_{manifest['iterations']-1}.pt"
        onnx = folder / 'policy.onnx'
        checkpoint_sha, onnx_sha = digest(checkpoint), digest(onnx)
        if checkpoint_sha != manifest['checkpoint_sha256'] or onnx_sha != manifest['onnx_sha256']:
            raise ValueError('Policy files differ from their original manifest')
    else:
        if not args.onnx or history is None:
            raise ValueError('--checkpoint requires --onnx and --history-report')
        checkpoint, onnx = args.checkpoint.resolve(), args.onnx.resolve()
        checkpoint_sha, onnx_sha = digest(checkpoint), digest(onnx)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    info = state['infos']
    policy, seed, iterations = info['policy'], info['seed'], state['iter'] + 1
    training_sha = info['training_protocol_sha256']
    if info['env_state']['common_step_counter'] != iterations * protocol['steps_per_iteration']:
        raise ValueError('Saved training counter differs from the checkpoint iteration')
    if manifest and (manifest['policy'] != policy or manifest['iterations'] != iterations
                     or manifest['training_protocol_sha256'] != training_sha
                     or manifest.get('seed', seed) != seed):
        raise ValueError('Original manifest differs from the saved training identity')
    original_report = None
    if history:
        selected = [row for p in history['policies'] for row in p['history']
                    if row['checkpoint_sha256'] == checkpoint_sha]
        if len(selected) != 1:
            raise ValueError('Checkpoint does not select one retained training ancestor')
        selected = selected[0]
        if not (selected['policy'] == policy and selected['seed'] == seed
                and selected['completed_iterations'] == iterations
                and selected['protocol_sha256'] == training_sha and selected['passed'] is True):
            raise ValueError('Saved ancestor has no matching completed original validation')
        original_report = Path(selected['validation_report'])
        original = json.loads(original_report.read_text())
        if original['checkpoint_sha256'] != checkpoint_sha or original['onnx_sha256'] != onnx_sha:
            raise ValueError('ONNX differs from the ancestry-bound saved export')
        old_path = Path(selected['protocol'])
        old = json.loads(old_path.read_text())
        if digest(old_path) != training_sha:
            raise ValueError('Original training protocol changed')
        for key in ['network', 'normalization', 'num_envs', 'steps_per_iteration',
                    'policy_groups', 'seeds', 'frozen_M6_sha256', 'frozen_O1_sha256']:
            if old[key] != protocol[key]:
                raise ValueError('Saved actor changed the frozen evaluation interface: ' + key)
        for key in ['validation_envs', 'validation_ticks', 'hold_scoring_start_tick',
                    'environment_seeds', 'trajectory_seeds']:
            if old['convergence'][key] != protocol['convergence'][key]:
                raise ValueError('Saved actor changed validation sampling: ' + key)
    elif training_sha != current_sha:
        raise ValueError('A historical training identity requires --history-report')
    return {'policy': policy, 'seed': seed, 'iterations': iterations,
            'checkpoint': checkpoint, 'onnx': onnx, 'checkpoint_sha256': checkpoint_sha,
            'onnx_sha256': onnx_sha, 'training_protocol_sha256': training_sha,
            'manifest_sha256': digest(manifest_path) if manifest_path else None,
            'original_validation_report': str(original_report) if original_report else None,
            'history_report_sha256': digest(args.history_report) if args.history_report else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actor_source = parser.add_mutually_exclusive_group(required=True)
    actor_source.add_argument('--policy-dir', type=Path)
    actor_source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--onnx', type=Path)
    parser.add_argument('--history-report', type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--donor-seeds', required=True, nargs='+', type=int)
    args = parser.parse_args()
    if len(args.donor_seeds) < 2 or len(set(args.donor_seeds)) != len(args.donor_seeds):
        raise ValueError('At least two distinct residual-draw seeds are required')
    protocol = json.loads(args.protocol.read_text())
    for path, sha in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if digest(path) != sha: raise RuntimeError('Frozen contract changed: '+path)
    actor = resolve_actor(args, protocol)
    policy, iterations = actor['policy'], actor['iterations']
    checkpoint, onnx = actor['checkpoint'], actor['onnx']
    source = Path(__file__).with_name('sc0090_validate_checkpoint.py')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output/'source.py')
    dump(output/'pre-run.json', {'policy': policy, 'seed': actor['seed'], 'iterations': iterations,
        'source_sha256': digest(__file__), 'validator_sha256': digest(source),
        'checkpoint_sha256': digest(checkpoint), 'onnx_sha256': digest(onnx),
        'manifest_sha256': actor['manifest_sha256'], 'protocol_sha256': digest(args.protocol),
        'training_protocol_sha256': actor['training_protocol_sha256'],
        'evaluation_protocol_sha256': digest(args.protocol),
        'history_report_sha256': actor['history_report_sha256'],
        'original_validation_report': actor['original_validation_report'],
        'original_training_identity_preserved': True,
        'donor_seeds': args.donor_seeds,
        'only_changed_input': 'Empirical residual generator seed; physical-condition and phase generators, targets and actor unchanged.',
        'fixed_degree_threshold_selected': False, 'hardware_executed': False})
    spec = importlib.util.spec_from_file_location('validator_for_repeatability', source)
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    factory = validator.ManagerBasedRlEnv
    repetitions = []; reference_profile = None; fault_count = 0
    for donor_seed in args.donor_seeds:
        physical_profiles = {}
        def seeded_factory(cfg, **kwargs):
            env = factory(cfg, **kwargs)
            pipeline = env.action_manager.get_term('position').pipeline
            pipeline.rng.manual_seed(donor_seed+cfg.seed)
            reset = env.reset
            def measured_reset(*positional, **keywords):
                result = reset(*positional, **keywords)
                prefix = 'environment'+str(cfg.seed)+'_'
                if pipeline.randomization is not None:
                    for key, value in {'phase_us': pipeline.episode_phase,
                        'conditions': pipeline.condition_index, 'voltage': pipeline.voltage,
                        'settings': pipeline.condition_settings[pipeline.condition_index],
                        'mass_kg': pipeline.condition_mass[pipeline.condition_index]}.items():
                        physical_profiles[prefix+key] = value.detach().cpu().numpy().copy()
                return result
            env.reset = measured_reset
            return env
        validator.ManagerBasedRlEnv = seeded_factory
        trial_dir = output/f'donor{donor_seed}'
        report = validator.validate(policy, checkpoint, onnx, args.protocol, trial_dir)
        fault_count += report['fault_count']; metrics = None
        if report['passed']:
            with np.load(trial_dir/'traces.npz', allow_pickle=False) as archive:
                signature = {key: archive[key].copy() for key in archive.files if not key.startswith('q_target')}
                for trial in range(2):
                    signature[f'target_trial{trial}'] = archive[f'q_target_trial{trial}'][:, :, 1].copy()
                signature.update(physical_profiles)
                if int(policy[1:]) >= 4:
                    assert len(physical_profiles) == 10
                if reference_profile is None: reference_profile = signature
                assert signature.keys() == reference_profile.keys()
                assert all(np.array_equal(value, reference_profile[key]) for key, value in signature.items())
                traces = np.concatenate([archive[f'q_target_trial{trial}'] for trial in range(2)], axis=1)
            np.savez_compressed(trial_dir/'physical-profile.npz', **physical_profiles)
            error = np.abs(traces[:, :, 0]-traces[:, :, 1])*180/np.pi
            assert np.allclose([error.mean(), np.quantile(error, .99)],
                [report['mae_deg'], report['p99_deg']], rtol=0, atol=1e-5)
            metrics = {key: report[key] for key in ['mae_deg', 'p99_deg', 'zero_hold_std_deg']}
            for family, values in report['families'].items():
                for key, value in values.items(): metrics[family+'_'+key] = value
        repetitions.append({'donor_seed': donor_seed, 'passed': report['passed'],
            'fault_count': report['fault_count'], 'metrics': metrics, 'trials': report['trials'],
            'report_sha256': digest(trial_dir/'report.json'), 'raw_sha256': digest(trial_dir/'traces.npz'),
            'physical_profile_sha256': digest(trial_dir/'physical-profile.npz') if report['passed'] else None})
        print(json.dumps(repetitions[-1]), flush=True)
    complete = all(r['passed'] for r in repetitions); observed = {}
    if complete:
        for key in repetitions[0]['metrics']:
            values = np.asarray([r['metrics'][key] for r in repetitions])
            pairs = np.asarray([abs(a-b) for i, a in enumerate(values) for b in values[i+1:]])
            observed[key] = {'values': values.tolist(), 'mean': float(values.mean()),
                'standard_deviation': float(values.std(ddof=1)), 'range': float(np.ptp(values)),
                'same_policy_abs_pair_differences': pairs.tolist(),
                'same_policy_abs_pair_difference_p95': float(np.quantile(pairs, .95))}
    dump(output/'report.json', {'passed': complete, 'policy': policy, 'seed': actor['seed'], 'iterations': iterations,
        'checkpoint_sha256': actor['checkpoint_sha256'], 'onnx_sha256': actor['onnx_sha256'],
        'training_protocol_sha256': actor['training_protocol_sha256'],
        'evaluation_protocol_sha256': digest(args.protocol),
        'validation_executor_sha256': digest(source), 'original_training_identity_preserved': True,
        'repetitions': repetitions, 'fault_count': fault_count,
        'condition_target_and_read_phase_profiles_identical': complete,
        'observed_repeatability': observed, 'source_sha256': digest(__file__),
        'pre_run_sha256': digest(output/'pre-run.json'),
        'fixed_degree_threshold_selected': False, 'automatic_stopping_enabled': False,
        'formal_convergence_claimed': False, 'hardware_executed': False, 'sim2real_benefit_verified': False,
        'scope': 'Descriptive initial residual-draw variation of the actual simulation validator at this checkpoint only; not a certified convergence margin or hardware uncertainty bound. Failed repetitions remain in the report.'})


if __name__ == '__main__':
    main()
