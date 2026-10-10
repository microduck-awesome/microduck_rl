"""Audit quiet-loop interfaces and local action sensitivity from real traces.

Counterfactual observations are evaluated offline only, with recorded action
history held fixed. They are not closed-loop trials or deployable truth inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(value):
    return {'mean': float(value.mean()), 'standard_deviation': float(value.std()),
            'mean_absolute': float(np.abs(value).mean()), 'abs_p99': float(np.quantile(np.abs(value), .99))}


def infer(session, inputs):
    return np.asarray([session.run(None, {session.get_inputs()[0].name: row[None]})[0][0, 0]
                       for row in inputs], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items(): assert digest(path) == sha, path
    trace = args.trace.resolve(); pre = json.loads((trace / 'pre-run.json').read_text())
    recorded = json.loads((trace / 'report.json').read_text())
    assert recorded['passed'] and recorded['fault_count'] == 0
    assert digest(trace / 'pre-run.json') == recorded['pre_run_sha256']
    assert digest(trace / 'actor-control-traces.npz') == recorded['control_trace_sha256']
    assert digest(trace / 'source.py') == recorded['source_sha256'] == pre['source_sha256']
    assert digest(trace / 'validator-source.py') == pre['validator_sha256']
    assert pre['protocol_sha256'] == digest(args.protocol)
    assert not pre['truth_used_for_actions'] and not pre['action_inputs_changed']
    checkpoint = Path(pre['checkpoint']); onnx = Path(pre['onnx'])
    assert digest(checkpoint) == pre['checkpoint_sha256'] and digest(onnx) == pre['onnx_sha256']
    manifest_path = checkpoint.parent / 'manifest.json'
    assert digest(manifest_path) == pre['manifest_sha256']
    manifest = json.loads(manifest_path.read_text()); c = manifest['action_contract']
    rule = protocol['convergence']; torch.set_num_threads(1)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'], sess_options=options)
    all_inputs = []; all_q_ref = []; all_v_ref = []; all_actions = []; rows = []; plotted = None
    with np.load(trace / 'actor-control-traces.npz', allow_pickle=False) as archive, \
         np.load(trace / 'validation/traces.npz', allow_pickle=False) as raw:
        for trial, seed in enumerate(rule['environment_seeds']):
            prefix = 'environment' + str(seed) + '_'
            before = archive[prefix + 'before_float']; after = archive[prefix + 'after_float']
            integers = archive[prefix + 'before_integer']; after_integer = archive[prefix + 'after_integer']
            worlds = rule['validation_envs'] // 5; ticks = rule['validation_ticks']
            assert before.shape == (ticks, worlds, 8) and after.shape == (ticks, worlds, 4)
            assert np.isfinite(before).all() and np.isfinite(after).all()
            assert np.all(before[:, :, 3] == 0)
            assert np.array_equal(after[:, :, 3], before[:, :, 4])
            assert np.array_equal(before[1:, :, 2], before[:-1, :, 4]) and np.all(before[0, :, 2] == 0)
            assert np.array_equal(before[1:, :, 5:7], after[:-1, :, :2])
            assert np.all(before[0, :, 5:7] == 0)
            assert np.array_equal(after_integer[:, :, 0] - integers[:, :, 0], np.full((ticks, worlds), 20000))
            assert np.array_equal(integers[1:, :, 0], after_integer[:-1, :, 0])
            assert np.array_equal(integers[1:, :, 1], after_integer[:-1, :, 1])
            selected = raw[f'family_trial{trial}'] == 0
            assert np.array_equal(raw[f'q_target_trial{trial}'][:, selected, 0], after[:, :, 0])
            assert np.array_equal(raw[f'condition_trial{trial}'][selected], integers[0, :, 2])
            q_ref = np.concatenate([np.zeros((1, worlds), np.float32), before[:-1, :, 5]])
            v_ref = np.concatenate([np.zeros((1, worlds), np.float32), before[:-1, :, 6]])
            # The frozen source queries control_clock-20ms. That is the previous
            # frame's pre-step state; current truth would introduce a timing error.
            assert pre['query_delay_us'] == 20000
            lower = max(c['minimum_count'], math.ceil((c['minimum_rad'] - c['calibration_bias']) /
                        c['calibration_scale'] / c['radians_per_count'] + c['zero_count']))
            upper = min(c['maximum_count'], math.floor((c['maximum_rad'] - c['calibration_bias']) /
                        c['calibration_scale'] / c['radians_per_count'] + c['zero_count']))
            delta = math.floor(c['maximum_rate_rad_s'] * c['policy_dt_s'] /
                               (c['calibration_scale'] * c['radians_per_count']))
            counts = torch.from_numpy(integers[:, :, 1])
            actions = torch.from_numpy(before[:, :, 4].copy())
            physical = (actions * c['action_scale_rad']).clamp(c['minimum_rad'], c['maximum_rad'])
            wanted = torch.round((physical - c['calibration_bias']) / c['calibration_scale'] /
                                 c['radians_per_count'] + c['zero_count']).clamp(lower, upper)
            expected = torch.minimum(counts + delta, torch.maximum(counts - delta, wanted)).numpy()
            mismatches = int(np.count_nonzero(expected != after_integer[:, :, 1]))
            assert np.all((after_integer[:, :, 1] >= lower) & (after_integer[:, :, 1] <= upper))
            assert np.max(np.abs(after_integer[:, :, 1] - integers[:, :, 1])) <= delta
            std = np.std(after[rule['hold_scoring_start_tick']:, :, 0] * 180 / np.pi, axis=0)
            rows.append({'environment_seed': seed, 'quiet_worlds': worlds, 'ticks': ticks,
                         'float32_action_mapping_mismatches': mismatches,
                         'mean_quiet_hold_std_deg': float(std.mean()),
                         'mean_count_changes_per_world': float(np.count_nonzero(
                             after_integer[:, :, 1] != integers[:, :, 1], axis=0).mean()),
                         'reference_q_estimation_error_deg': summary((before[:, :, 0] - q_ref) * 180 / np.pi),
                         'reference_v_estimation_error_rad_s': summary(before[:, :, 1] - v_ref)})
            all_inputs.append(before[:, :, :4].reshape(-1, 4))
            all_q_ref.append(q_ref.reshape(-1)); all_v_ref.append(v_ref.reshape(-1))
            all_actions.append(before[:, :, 4].reshape(-1))
            if plotted is None: plotted = (before.copy(), after.copy(), integers.copy(), q_ref.copy(), v_ref.copy())
    inputs = np.concatenate(all_inputs); q_ref = np.concatenate(all_q_ref); v_ref = np.concatenate(all_v_ref)
    actions = np.concatenate(all_actions); predicted = infer(session, inputs)
    maximum = float(np.max(np.abs(predicted - actions))); assert maximum < 1e-5
    variants = {}
    for name, use_q, use_v, reset_history in [('reference_q_only', True, False, False),
         ('reference_v_only', False, True, False), ('reference_q_and_v', True, True, False),
         ('reference_q_and_v_zero_history', True, True, True)]:
        changed = inputs.copy()
        if use_q: changed[:, 0] = q_ref
        if use_v: changed[:, 1] = v_ref
        if reset_history: changed[:, 2] = 0.
        response = infer(session, changed)
        variants[name] = {'requested_physical_action_change_deg': summary(
            (response - predicted) * c['action_scale_rad'] * 180 / np.pi),
            'requested_physical_action_deg': summary(response * c['action_scale_rad'] * 180 / np.pi)}
    zero_input = infer(session, np.zeros((1, 4), np.float32))[0]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'source.py')
    evidence = {str(path): digest(path) for path in [trace / 'pre-run.json', trace / 'report.json',
        trace / 'actor-control-traces.npz', trace / 'physical-profile.npz', trace / 'source.py',
        trace / 'validator-source.py', trace / 'validation/report.json', trace / 'validation/traces.npz',
        checkpoint, onnx, manifest_path, args.protocol]}
    result = {'passed': all(row['float32_action_mapping_mismatches'] == 0 for row in rows),
              'policy': pre['policy'], 'training_seed': pre['seed'], 'iterations': pre['iterations'],
              'quiet_worlds': sum(row['quiet_worlds'] for row in rows), 'control_nodes': len(inputs),
              'query_delay_us': 20000, 'raw_action_history_exact': True, 'timing_and_bounds_passed': True,
              'maximum_actual_actor_CPU_ONNX_difference': maximum,
              'nominal_full_trace_exactly_reproduced': recorded['nominal_full_trace_exactly_reproduced'],
              'actual_requested_physical_action_deg': summary(actions * c['action_scale_rad'] * 180 / np.pi),
              'zero_input_requested_physical_action_deg': float(zero_input * c['action_scale_rad'] * 180 / np.pi),
              'world_groups': rows, 'offline_one_step_input_probes': variants,
              'source_sha256': digest(__file__), 'evidence_sha256': evidence,
              'diagnostic_execution_role': pre.get('execution_role', 'frozen evaluator trace'),
              'evaluation_executor_matches_frozen_protocol': pre.get('evaluation_executor_matches_frozen_protocol', True),
              'frozen_contract_files_unchanged': len(frozen), 'truth_used_for_actions': False,
              'scope': '64 actual quiet validation worlds; original closed loop and all recorded input/action '
                       'history preserved. Reference truth aligned to control-20ms, not current time. '
                       'Counterfactual probes change individual recorded observations offline only, '
                       'with original action history held fixed; no altered closed-loop performance claim. '
                       'Passing these checks excludes the audited mapping/history/timing inconsistencies, '
                       'not every possible software defect or a sensor-error causal root-cause claim.',
              'fixed_degree_threshold_selected': False, 'formal_convergence_claimed': False,
              'hardware_executed': False, 'sim2real_benefit_verified': False}
    for path, sha in frozen.items(): assert digest(path) == sha, path
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    before, after, integers, qr, vr = plotted
    time = (integers[:, 0, 0] - integers[0, 0, 0]) / 1e6
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, before[:, 0, 5] * 180 / np.pi, label='Physical q at control time')
    axes[0].plot(time - .02, before[:, 0, 0] * 180 / np.pi, alpha=.75, label='Estimated q at query time')
    axes[0].step(time, (c['calibration_bias'] + c['calibration_scale'] * after[:, 0, 2]) * 180 / np.pi,
                 where='post', alpha=.7, label='Applied physical goal')
    axes[0].set_ylabel('Angle (deg)'); axes[0].legend(fontsize=8)
    axes[1].plot(time - .02, before[:, 0, 1], label='Estimated v at query time')
    axes[1].plot(time - .02, vr[:, 0], label='Physical v at query time')
    axes[1].set_ylabel('Velocity (rad/s)'); axes[1].legend(fontsize=8)
    probe = before[:, 0, :4].copy(); probe[:, 0] = qr[:, 0]; probe[:, 1] = vr[:, 0]
    axes[2].plot(time, before[:, 0, 4], label='Executed actor action')
    axes[2].plot(time, infer(session, probe), label='Offline reference-input probe (recorded history)')
    axes[2].set_ylabel('Raw action'); axes[2].set_xlabel('Control time since episode start (s)')
    axes[2].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.25)
    figure.suptitle(f"{pre['policy']}, seed{pre['seed']}, {pre['iterations']} iterations; validation seed{rule['environment_seeds'][0]}, world0")
    figure.tight_layout(); figure.savefig(output / 'quiet-world0.png', dpi=160); plt.close(figure)
    print(json.dumps({'passed': result['passed'], 'control_nodes': len(inputs),
          'mapping_mismatches': sum(row['float32_action_mapping_mismatches'] for row in rows),
          'maximum_actor_difference': maximum,
          'offline_probe_mean_absolute_changes_deg': {key: value['requested_physical_action_change_deg']['mean_absolute']
                                                       for key, value in variants.items()}}))
    if not result['passed']: raise SystemExit('Mapping consistency needs investigation; evidence retained')


if __name__ == '__main__':
    main()
