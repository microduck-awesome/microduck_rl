"""Evaluate direct-target B0 through the frozen real SC0090 GPU validator.

The anchor preserves the validator's initialization and saved environment
context. Its learned weights never determine B0's actions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class DirectTarget(torch.nn.Module):
    def __init__(self, action_scale):
        super().__init__()
        self.action_scale = action_scale

    def forward(self, observations):
        return observations[:, 3:4] / self.action_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--donor-seed', type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    frozen = {**protocol['source_sha256'], **protocol['input_sha256']}
    for path, sha in frozen.items():
        assert digest(path) == sha, path
    folder = args.anchor_dir.resolve()
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['training_protocol_sha256'] == digest(args.protocol)
    checkpoint = folder / f"model_{manifest['iterations'] - 1}.pt"
    assert digest(checkpoint) == manifest['checkpoint_sha256']
    scale = manifest['action_contract']['action_scale_rad']
    source = Path(__file__).with_name('sc0090_validate_checkpoint.py')
    spec = importlib.util.spec_from_file_location('frozen_direct_validator', source)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'source.py')
    shutil.copyfile(source, output / 'validator-source.py')
    graph = helper.make_graph([
        helper.make_node('Gather', ['observations', 'target_index'], ['target'], axis=1),
        helper.make_node('Div', ['target', 'action_scale'], ['action'])], 'sc0090-B0-direct',
        [helper.make_tensor_value_info('observations', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(np.array([3], np.int64), name='target_index'),
         numpy_helper.from_array(np.array(scale, np.float32), name='action_scale')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx_path = output / 'B0.onnx'; onnx.save(model, onnx_path)
    actor = DirectTarget(scale)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'],
                                  sess_options=validator._ort_options())
    rng = np.random.default_rng(91411)
    maximum = 0.
    for row in rng.uniform(-1, 1, (512, 4)).astype(np.float32):
        expected = actor(torch.from_numpy(row[None])).numpy()
        actual = session.run(None, {'observations': row[None]})[0]
        maximum = max(maximum, float(np.abs(actual - expected).max()))
        changed = row.copy(); changed[:3] = rng.uniform(-100, 100, 3)
        assert np.array_equal(actual, session.run(None, {'observations': changed[None]})[0])
    assert maximum < 1e-5
    pre = {'policy': 'B0', 'route_policy': manifest['policy'], 'S': manifest['S'],
           'O': manifest['O'], 'R': manifest['R'], 'anchor_iterations': manifest['iterations'],
           'anchor_checkpoint_sha256': digest(checkpoint),
           'anchor_manifest_sha256': digest(folder / 'manifest.json'),
           'source_sha256': digest(__file__), 'validator_sha256': digest(source),
           'protocol_sha256': digest(args.protocol), 'onnx_sha256': digest(onnx_path),
           'action_contract': manifest['action_contract'], 'donor_seed': args.donor_seed,
           'action_rule': 'current physical task target / frozen action_scale_rad',
           'learned_anchor_weights_determine_actions': False,
           'truth_used_for_actions': False, 'NN_training_performed': False,
           'parity_test_nodes': 512, 'maximum_torch_onnx_difference': maximum,
           'hardware_executed': False, 'scope': __doc__}
    dump(output / 'pre-run.json', pre)
    original_factory = validator.ManagerBasedRlEnv
    original_runner = validator.MjlabOnPolicyRunner
    physical_profiles = {}

    def seeded_factory(cfg, **kwargs):
        env = original_factory(cfg, **kwargs)
        pipeline = env.action_manager.get_term('position').pipeline
        pipeline.rng.manual_seed(args.donor_seed + cfg.seed)
        reset = env.reset
        def measured_reset(*positional, **keywords):
            result = reset(*positional, **keywords)
            if pipeline.randomization is not None:
                prefix = 'environment' + str(cfg.seed) + '_'
                for key, value in {'phase_us': pipeline.episode_phase,
                    'conditions': pipeline.condition_index, 'voltage': pipeline.voltage,
                    'settings': pipeline.condition_settings[pipeline.condition_index],
                    'mass_kg': pipeline.condition_mass[pipeline.condition_index]}.items():
                    physical_profiles[prefix + key] = value.detach().cpu().numpy().copy()
            return result
        env.reset = measured_reset
        return env

    class DirectExport:
        def as_onnx(self, verbose=False):
            return DirectTarget(scale)

    class DirectAlgorithm:
        def get_policy(self):
            return DirectExport()

    class ContextRunner:
        def __init__(self, *positional, **keywords):
            self.context = original_runner(*positional, **keywords)
            self.alg = DirectAlgorithm()
        def load(self, *positional, **keywords):
            return self.context.load(*positional, **keywords)

    validator.ManagerBasedRlEnv = seeded_factory
    validator.MjlabOnPolicyRunner = ContextRunner
    result = validator.validate(manifest['policy'], checkpoint, onnx_path,
                                args.protocol, output / 'validation')
    original_report = output / 'validation/report.json'
    shutil.copyfile(original_report, original_report.with_name('validator-route-report.json'))
    np.savez_compressed(output / 'physical-profile.npz', **physical_profiles)
    result.update(policy='B0', route_policy=manifest['policy'], action_rule=pre['action_rule'],
                  anchor_checkpoint_sha256=result.pop('checkpoint_sha256'),
                  learned_anchor_weights_determine_actions=False, truth_used_for_actions=False,
                  wrapper_source_sha256=digest(__file__),
                  scope='Direct-target B0 with the route anchor preserving initialization; '
                        'the same frozen GPU environment, task, causal stream and action boundary. '
                        'Loaded learned weights are ignored by the direct-target actor. Simulation only.')
    dump(original_report, result)
    report = {'passed': result['passed'], 'policy': 'B0', 'route_policy': manifest['policy'],
              'donor_seed': args.donor_seed, 'fault_count': result['fault_count'],
              'mae_deg': result['mae_deg'], 'p99_deg': result['p99_deg'],
              'zero_hold_std_deg': result['zero_hold_std_deg'], 'families': result['families'],
              'pre_run_sha256': digest(output / 'pre-run.json'),
              'validation_report_sha256': digest(original_report),
              'raw_sha256': result['raw_sha256'], 'onnx_sha256': digest(onnx_path),
              'physical_profile_sha256': digest(output / 'physical-profile.npz'),
              'source_sha256': digest(__file__), 'frozen_contract_files_unchanged': len(frozen),
              'formal_convergence_claimed': False, 'hardware_executed': False,
              'sim2real_benefit_verified': False, 'scope': result['scope']}
    for path, sha in frozen.items():
        assert digest(path) == sha, path
    dump(output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ['passed', 'policy', 'fault_count', 'mae_deg', 'p99_deg']}), flush=True)
    if not report['passed']:
        raise SystemExit('Direct baseline validation failed; raw data and failure retained')


if __name__ == '__main__':
    main()
