"""Record unchanged deterministic actor inputs/actions in quiet validation worlds.

Truth is diagnostic telemetry only. The original frozen validator and policy
execute the complete trajectories; no input, action, reset or timing is replaced.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items(): assert digest(path) == sha, path
    folder = args.policy_dir.resolve(); manifest = json.loads((folder / 'manifest.json').read_text())
    checkpoint = folder / f"model_{manifest['iterations'] - 1}.pt"
    onnx = folder / 'policy.onnx'
    assert digest(checkpoint) == manifest['checkpoint_sha256']
    assert digest(onnx) == manifest['onnx_sha256']
    assert manifest['training_protocol_sha256'] == digest(args.protocol)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('sc0090_validate_checkpoint.py')
    shutil.copyfile(__file__, output / 'source.py')
    shutil.copyfile(source, output / 'validator-source.py')
    pre = {'policy': manifest['policy'], 'seed': manifest['seed'], 'iterations': manifest['iterations'],
           'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
           'onnx': str(onnx), 'onnx_sha256': digest(onnx),
           'protocol_sha256': digest(args.protocol), 'source_sha256': digest(__file__),
           'validator_sha256': digest(source), 'manifest_sha256': digest(folder / 'manifest.json'),
           'truth_used_for_actions': False, 'action_inputs_changed': False,
           'query_delay_us': 20000, 'hardware_executed': False,
           'before_float_columns': ['q_hat', 'v_hat', 'last_raw_action', 'target', 'raw_action',
                                    'physical_q', 'physical_v', 'previous_goal_nominal'],
           'after_float_columns': ['physical_q', 'physical_v', 'goal_nominal', 'stored_raw_action'],
           'before_integer_columns': ['control_clock_us', 'goal_count', 'condition_index'],
           'after_integer_columns': ['control_clock_us', 'goal_count'], 'scope': __doc__}
    dump(output / 'pre-run.json', pre)
    spec = importlib.util.spec_from_file_location('quiet_trace_validator', source)
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    original_factory = validator.ManagerBasedRlEnv
    records = {}; physical_profiles = {}

    def traced_factory(cfg, **kwargs):
        env = original_factory(cfg, **kwargs)
        action = env.action_manager.get_term('position')
        pipeline = action.pipeline
        selected = (torch.arange(env.num_envs, device=env.device) % 5 == 0).nonzero().flatten()
        current = {'actor': None}
        fields = {key: [] for key in ['before_float', 'after_float', 'before_integer', 'after_integer']}
        records[str(cfg.seed)] = fields
        compute = env.observation_manager.compute
        def measured_compute(*positional, **keywords):
            result = compute(*positional, **keywords)
            if 'actor' in result:
                current['actor'] = result['actor'].detach().clone()
            return result
        env.observation_manager.compute = measured_compute
        reset = env.reset
        def measured_reset(*positional, **keywords):
            result = reset(*positional, **keywords)
            if pipeline.randomization is not None:
                prefix = 'environment' + str(cfg.seed) + '_'
                for key, value in {'phase_us': pipeline.episode_phase, 'conditions': pipeline.condition_index,
                    'voltage': pipeline.voltage, 'settings': pipeline.condition_settings[pipeline.condition_index],
                    'mass_kg': pipeline.condition_mass[pipeline.condition_index]}.items():
                    physical_profiles[prefix + key] = value.detach().cpu().numpy().copy()
            return result
        env.reset = measured_reset
        step = env.step
        def measured_step(actions, *positional, **keywords):
            obs = current['actor']
            if obs is None: raise RuntimeError('The actual actor input was not captured')
            bench = env.scene['bench']
            before_float = torch.cat([obs, actions, bench.data.joint_pos[:, :1],
                                      bench.data.joint_vel[:, :1], action.goal[:, None]], dim=1)[selected]
            before_integer = torch.stack([action.clock, action.goal_count, pipeline.condition_index], dim=1)[selected]
            # Copy before executing the original step; scene buffers are mutable.
            before_float = before_float.cpu().numpy().copy()
            before_integer = before_integer.cpu().numpy().copy()
            result = step(actions, *positional, **keywords)
            after_float = torch.stack([bench.data.joint_pos[:, 0], bench.data.joint_vel[:, 0],
                                       action.goal, action.raw_action[:, 0]], dim=1)[selected].cpu().numpy().copy()
            after_integer = torch.stack([action.clock, action.goal_count], dim=1)[selected].cpu().numpy().copy()
            for key, value in [('before_float', before_float), ('before_integer', before_integer),
                               ('after_float', after_float), ('after_integer', after_integer)]:
                fields[key].append(value)
            return result
        env.step = measured_step
        return env

    validator.ManagerBasedRlEnv = traced_factory
    report = validator.validate(manifest['policy'], checkpoint, onnx, args.protocol, output / 'validation')
    arrays = {f'environment{seed}_{key}': np.asarray(value) for seed, fields in records.items()
              for key, value in fields.items()}
    np.savez_compressed(output / 'actor-control-traces.npz', **arrays)
    np.savez_compressed(output / 'physical-profile.npz', **physical_profiles)
    nominal_path = folder / f"evaluations/iteration{manifest['iterations']:04}/validation/traces.npz"
    exact = True; maximum = 0.; profiles_equal = True
    with np.load(output / 'validation/traces.npz', allow_pickle=False) as actual, \
         np.load(nominal_path, allow_pickle=False) as nominal:
        assert actual.files == nominal.files
        for key in actual.files:
            exact &= np.array_equal(actual[key], nominal[key])
            if key.startswith('q_target'):
                maximum = max(maximum, float(np.max(np.abs(actual[key] - nominal[key]))))
                profiles_equal &= np.array_equal(actual[key][:, :, 1], nominal[key][:, :, 1])
            else:
                profiles_equal &= np.array_equal(actual[key], nominal[key])
    for path, sha in frozen.items(): assert digest(path) == sha, path
    result = {'passed': report['passed'] and profiles_equal, 'policy': manifest['policy'],
              'iterations': manifest['iterations'], 'fault_count': report['fault_count'],
              'nominal_full_trace_exactly_reproduced': bool(exact),
              'maximum_nominal_trace_difference_rad': maximum,
              'nominal_target_family_condition_profiles_identical': bool(profiles_equal),
              'pre_run_sha256': digest(output / 'pre-run.json'),
              'control_trace_sha256': digest(output / 'actor-control-traces.npz'),
              'physical_profile_sha256': digest(output / 'physical-profile.npz'),
              'validation_report_sha256': digest(output / 'validation/report.json'),
              'nominal_trace_sha256': digest(nominal_path), 'source_sha256': digest(__file__),
              'frozen_contract_files_unchanged': len(frozen), 'truth_used_for_actions': False,
              'formal_convergence_claimed': False, 'hardware_executed': False,
              'sim2real_benefit_verified': False, 'scope': __doc__}
    dump(output / 'report.json', result)
    print(json.dumps({key: result[key] for key in ['passed', 'fault_count',
          'nominal_full_trace_exactly_reproduced', 'maximum_nominal_trace_difference_rad']}), flush=True)
    if not result['passed']: raise SystemExit('Quiet trace execution incomplete; records retained')


if __name__ == '__main__':
    main()
