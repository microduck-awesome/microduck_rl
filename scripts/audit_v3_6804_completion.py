"""Recheck V3 A6804 source, recorded training lineage, and final artifacts.

Run with scripts/v2_python.sh.  The original experiment directory is required
to verify intermediate checkpoints and source archives; the resulting report
records their hashes and the distinct training environment configurations.
This checks simulation evidence, not physical robot qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401  # Registers the historical task name.
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from dataclasses import asdict
from evaluate_v2_head import cases_for


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/l40-vm01/data2/experiments/v3-rise6804-20261101')
ASSEMBLY = Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-assembly-v3-home-second')
V2 = '67120c2fc9c5f4be8323ce04588ab18c9a643e45'
ASSEMBLY_REV = '46a1b2c1d371def5775d33e67424755f253e9881'
MODEL_SHA = '45a9ce13c521ac07080381e9d5597b040e1988b98fb9f89c981008a7dac778dd'
MOTOR_SHA = '7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4'
TASK = 'Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'
BUCKETS = ('standing', 'sitting', 'prone', 'frontier', 'left_side', 'right_side', 'supine')
PHASES = (
    ('rollback001-recovery-to500', 1, 500, 0),
    ('recovery-0500-to1500', 500, 1500, 0),
    ('recovery-1500-to2500', 1500, 2500, 0),
    ('recovery-2500-to3500', 2500, 3500, 0),
    ('recovery-3500-to3762', 3500, 3762, 0),
    ('head1-3762-to4762', 3762, 4762, 1),
    ('head2-4762-to5012', 4762, 5012, 2),
    ('head4-5012-to5524', 5012, 5524, 4),
    ('head4-5524-to6036', 5524, 6036, 4),
    ('head2-6036-to6548', 6036, 6548, 2),
    ('head4-6548-to6804', 6548, 6804, 4),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def normalized(obj):
    # run.json prints callables with process-specific object addresses.
    return re.sub(r'0x[0-9a-fA-F]+', '<address>', json.dumps(obj, sort_keys=True))


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def verify_archive(path, manifest):
    seen = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in manifest and member.name not in seen
            file = archive.extractfile(member)
            assert file is not None
            digest = hashlib.sha256()
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
            assert digest.hexdigest() == manifest[member.name], member.name
            seen.add(member.name)
    assert seen == set(manifest)


def policy_parity(checkpoint, onnx_path, joint_names):
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    metadata = {x.key: x.value for x in model.metadata_props}
    assert tuple(metadata['joint_names'].split(',')) == tuple(joint_names)
    assert metadata['microduck_motor_sha256'] == MOTOR_SHA
    assert metadata['microduck_robot'] == 'microduck-v3-sc0090'
    config = asdict(load_rl_cfg(TASK))
    obs = TensorDict({'actor': torch.zeros(1, 61), 'critic': torch.zeros(1, 74)}, batch_size=[1])
    groups = resolve_obs_groups(obs, config['obs_groups'], ['actor', 'critic'])
    for key in ('cnn_cfg', 'distribution_cfg'):
        if config['actor'].get(key) is None:
            config['actor'].pop(key, None)
    if config['actor'].get('rnn_type') is None:
        for key in ('rnn_type', 'rnn_hidden_dim', 'rnn_num_layers'):
            config['actor'].pop(key, None)
    policy = resolve_callable(config['actor'].pop('class_name'))(obs, groups, 'actor', 14, **config['actor'])
    policy.load_state_dict(checkpoint['actor_state_dict'])
    policy.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    mean = policy.obs_normalizer.mean.detach().numpy()
    std = policy.obs_normalizer.std.detach().numpy()
    rng = np.random.default_rng(20261023)
    samples = [np.zeros((1, 61), np.float32), np.linspace(-.1, .1, 61, dtype=np.float32)[None]]
    samples += [(mean + rng.normal(size=(1, 61)) * std * .5).astype(np.float32) for _ in range(64)]
    error = 0.
    for sample in samples:
        result = session.run(None, {session.get_inputs()[0].name: sample})[0]
        with torch.inference_mode():
            expected = policy({'actor': torch.as_tensor(sample)}).numpy()
        assert result.shape == (1, 14) and np.isfinite(result).all()
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)
        error = max(error, float(np.abs(result - expected).max()))
    return {'samples': len(samples), 'max_abs_error': error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, default=BASE)
    parser.add_argument('--assembly', type=Path, default=ASSEMBLY)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, assembly = args.experiment.resolve(), args.assembly.resolve()
    result_root = ROOT / 'repro/sc0090-v3/rise6804'
    evidence = result_root / 'evidence'
    torch.set_num_threads(1)

    def git(*cmd, cwd=ROOT):
        return subprocess.check_output(['git', *cmd], cwd=cwd, text=True).strip()

    head = git('rev-parse', 'HEAD')
    assert git('merge-base', head, V2) == V2
    assert git('rev-parse', 'HEAD', cwd=assembly) == ASSEMBLY_REV
    assert not git('status', '--porcelain', cwd=assembly)
    assert not git('diff', '--name-only', V2, head, '--',
                   'src/mjlab_microduck/actuator',
                   'src/mjlab_microduck/custom_v2/error_proxy.py',
                   'src/mjlab_microduck/custom_v2/paired_randomization.py',
                   'src/mjlab_microduck/custom_v2/training-assets', 'pyproject.toml', 'uv.lock')

    assets = ROOT / 'src/mjlab_microduck/custom_v2/assets'
    source_files = sorted(p for p in (assembly / 'model').rglob('*') if p.is_file())
    assert len(source_files) == 78
    for source in source_files:
        dest = assets / source.relative_to(assembly / 'model')
        assert dest.is_file() and sha(source) == sha(dest), str(source)
    assert sha(assembly / 'reports/model_metadata.json') == sha(assets / 'model_metadata.json')
    assert sha(assets / 'robot.xml') == MODEL_SHA
    assert sha(ROOT / 'src/mjlab_microduck/actuator/params/sc0090_v5_m6.json') == MOTOR_SHA
    compiled = read(ROOT / 'docs/v3-compiled-model-audit.json')
    assert compiled['status'] == 'PASS' and not compiled['mismatches'] and compiled['model_sha256'] == MODEL_SHA
    metadata = read(assets / 'model_metadata.json')
    joints = metadata['policy_joints']
    assert len(joints) == 14 and len(metadata['home_joint_positions_rad']) == 15
    xml = ET.parse(assets / 'robot.xml').getroot()
    limits = {x.attrib['name']: tuple(map(float, x.attrib['range'].split()))
              for x in xml.iter('joint') if 'name' in x.attrib and 'range' in x.attrib}
    for name in joints:
        assert limits[name][0] < metadata['home_joint_positions_rad'][name] < limits[name][1]
    assert metadata['home_joint_positions_rad']['mouth'] == limits['mouth'][0] == 0
    print('PASS: assembly source, compiled model evidence, motor and HOME', flush=True)

    origin = base / 'recovery-home-corrected-native-0500'
    initial_run = read(origin / 'run.json')
    assert initial_run['resume'] is None and initial_run['git_head'] == '4312b3499191911429a2a92595dce40893398894'
    initial_cp_path = origin / 'model_0.pt'
    first = torch.load(initial_cp_path, map_location='cpu', weights_only=False)
    assert first['infos']['v2']['completed_updates'] == 1
    assert first['infos']['v2']['total_environment_steps'] == 8192 * 24
    assert first['infos']['env_state']['common_step_counter'] == 24
    assert first['infos']['v2']['source_sha256'] == initial_run['source_sha256']
    common_manifest = initial_run['source_sha256']
    assert len(common_manifest) >= 100
    for path, digest in common_manifest.items():
        assert (ROOT / path).is_file() and sha(ROOT / path) == digest, path
    verify_archive(origin / 'source.tar.gz', common_manifest)
    common_agent, common_dependencies = initial_run['agent'], initial_run['dependencies']
    recipes = {}
    phases = []
    previous = initial_cp_path
    for name, start, end, stage in PHASES:
        folder = base / name
        run = read(folder / 'run.json')
        status = read(folder / 'status.json')
        verified = read(folder / 'resume-verification.json')
        restored = read(folder / 'resume-environment.json')
        assert run['task'] == TASK and run['device'] == 'cuda:0' and run['num_envs'] == 8192
        assert run['seed'] == 20261015 and run['agent']['num_steps_per_env'] == 24
        assert run['allocation_updates'] == end - start
        assert run['agent'] == common_agent and run['dependencies'] == common_dependencies
        assert run['source_sha256'] == common_manifest
        subprocess.run(['git', 'merge-base', '--is-ancestor', run['git_head'], head], cwd=ROOT, check=True)
        assert Path(run['resume']).resolve() == previous.resolve()
        parent_sha = sha(previous)
        assert verified['status'] == 'PASS' and verified['checkpoint_sha256'] == parent_sha
        assert verified['completed_updates'] == start and verified['common_step_counter'] == start * 24
        assert verified['cumulative_environment_steps'] == start * 8192 * 24
        assert verified['new_num_envs'] == verified['parent_num_envs'] == 8192
        assert verified['actor_critic_normalizers_optimizer_exact'] and not verified['source_changed']
        assert restored['status'] == 'PASS' and restored['all_spawn_stages_match']
        assert status['status'] == 'ALLOCATION_COMPLETE' and status['completed_updates'] == end
        assert status['controls'] == end * 8192 * 24
        recipe = run['head_precision_recipe']
        assert (recipe['stage'] if recipe else 0) == stage
        preflight = read(run['preflight'])
        assert preflight['status'] == 'PASS' and preflight['envs'] == 8192
        assert preflight['cuda_visible_devices'] == '0' and preflight['device'] == 'cuda:0'
        assert preflight['robot_xml_sha256'] == MODEL_SHA and preflight['motor_sha256'] == MOTOR_SHA
        assert (preflight['head_precision_recipe']['stage'] if preflight['head_precision_recipe'] else 0) == stage
        assert all(sha(ROOT / path) == digest for path, digest in preflight['numerical_source_sha256'].items())
        assert preflight['throughput']['measured_updates'] >= 10
        env = normalized(run['environment'])
        if str(stage) not in recipes:
            recipes[str(stage)] = {'environment': run['environment'], 'head_recipe': recipe}
        else:
            assert env == normalized(recipes[str(stage)]['environment'])
            assert normalized(recipe) == normalized(recipes[str(stage)]['head_recipe'])
        verify_archive(folder / 'source.tar.gz', common_manifest)
        previous = Path(status['resume_checkpoint'])
        assert previous.is_file()
        phases.append({'name': name, 'start': start, 'end': end, 'head_stage': stage,
                       'run_sha256': sha(folder / 'run.json'), 'status_sha256': sha(folder / 'status.json'),
                       'resume_verification_sha256': sha(folder / 'resume-verification.json'),
                       'parent_checkpoint_sha256': parent_sha, 'checkpoint_sha256': sha(previous),
                       'preflight_sha256': sha(run['preflight']), 'recorded_git_head': run['git_head'],
                       'feedback_rng': verified['feedback_rng'], 'physics_resume': verified['physics_resume']})
        print(f'PASS: {name} ({start} -> {end})', flush=True)

    final_checkpoint = torch.load(base / PHASES[-1][0] / 'checkpoint_006804.pt', map_location='cpu', weights_only=False)
    published = torch.load(result_root / 'resume_006804.pt', map_location='cpu', weights_only=False)
    assert sha(result_root / 'resume_006804.pt') == sha(previous)
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        assert_equal(final_checkpoint[key], published[key])
    assert published['infos']['v2']['completed_updates'] == 6804
    assert published['infos']['v2']['total_environment_steps'] == 6804 * 8192 * 24
    assert published['infos']['v2']['head_precision_recipe']['stage'] == 4
    assert published['infos']['v2']['recovery']['polish']
    artifact = read(evidence / 'artifact-verification.json')
    assert artifact['status'] == 'PASS' and artifact['resume_checkpoint_sha256'] == sha(previous)
    assert artifact['checkpoint_sha256'] == sha(base / PHASES[-1][0] / 'checkpoint_006804.pt')
    assert artifact['onnx_sha256'] == sha(result_root / 'policy.onnx')
    parity = policy_parity(published, result_root / 'policy.onnx', joints)

    recovery = {}
    for filename in ('evaluation_006804.json', 'recovery-200hz.json', 'recovery-50hz-replay.json'):
        report = read(evidence / filename)
        assert report['checkpoint_sha256'] == artifact['checkpoint_sha256']
        assert {c['seed'] for c in report['cases']} == {2026101501, 2026101502}
        for case in report['cases']:
            assert case['finite'] and set(case['groups']) == set(BUCKETS)
            for entry in case['groups'].values():
                assert entry['trials'] == entry['wins'] == 128 and entry['faults'] == 0
        recovery[filename] = {'sha256': sha(evidence / filename), 'wins': 1792, 'trials': 1792}
    min_final_clearance = float('inf')
    max_final_home_error = 0.
    for seed in (2026101501, 2026101502):
        result = read(evidence / f'clearance-{seed}.json')
        assert result['physics_dt_s'] == .005 and result['non_sitting_initial_home_max_abs_error_rad'] == 0
        assert set(result['joint_results']) == set(BUCKETS)
        for bucket in BUCKETS:
            assert set(result['joint_results'][bucket]) == set(joints)
            for name in joints:
                home = metadata['home_joint_positions_rad'][name]
                margin = math.degrees(min(home - limits[name][0], limits[name][1] - home))
                max_error = result['final_half_second_home_results'][bucket][name]['final_half_second_max_abs_error_deg']
                min_final_clearance = min(min_final_clearance, margin - max_error)
                max_final_home_error = max(max_final_home_error, max_error)
    assert min_final_clearance > 5.
    head_report = read(evidence / 'head-response.json')
    assert head_report['checkpoint_sha256'] == artifact['checkpoint_sha256']
    assert head_report['robot_xml_sha256'] == MODEL_SHA
    assert len(head_report['records']) == 162
    assert {r['seed'] for r in head_report['records']} == {2026101503, 2026101504}
    assert all(r['trials'] == 8 and r['invalid_trials'] == 0 for r in head_report['records'])
    expected_cases = {c['name']: c for c in cases_for(head_report['current_ranges'], head_report['final_ranges'])}
    assert len(expected_cases) == 81
    for seed in (2026101503, 2026101504):
        rows = [r for r in head_report['records'] if r['seed'] == seed]
        assert len(rows) == len({r['name'] for r in rows}) == len(expected_cases)
        assert {r['name'] for r in rows} == set(expected_cases)
        for row in rows:
            expected = expected_cases[row['name']]
            assert row['range'] == expected['range'] and row['target'] == expected['target']
    for r in head_report['records']:
        assert max(map(abs, r['initial_head_error_deg'])) < 1e-6
        for name, offset in zip(head_report['joint_order'], r['target']):
            absolute = metadata['home_joint_positions_rad'][name] + offset
            assert limits[name][0] <= absolute <= limits[name][1]

    reset = read(evidence / 'reset-home-corrected-gpu.json')
    assert reset['device'] == 'physical GPU 0' and len(reset['results']) == 84
    assert all(x['samples'] == 128 and x['over_5mm'] == 0 for x in reset['results'])
    assert {(x['stage'], x['bucket'], x['kind']) for x in reset['results']} == {
        (stage, bucket, kind) for stage in range(6) for bucket in BUCKETS for kind in ('floor', 'self')}
    assert {x['stage'] for x in reset['initial_home_checks']} == set(range(6))
    assert all(x['non_sitting_15_joint_max_error_rad'] == x['all_buckets_mouth_max_error_rad'] == 0
               for x in reset['initial_home_checks'])
    assert all(x == 0 for x in reset['unreset_world_state_max_abs_diff'].values())
    audit = {'status': 'PASS', 'audited_git_head': head, 'v2_base': V2, 'assembly_revision': ASSEMBLY_REV,
             'assembly_file_count': len(source_files), 'model_sha256': MODEL_SHA, 'motor_sha256': MOTOR_SHA,
             'source_sha256': common_manifest, 'dependencies': common_dependencies, 'agent': common_agent,
             'training_environments_by_head_stage': recipes, 'initial_checkpoint_sha256': sha(initial_cp_path),
             'phases': phases, 'completed_updates': 6804, 'controls': 6804 * 8192 * 24,
             'published_resume_sha256': sha(previous), 'published_onnx_sha256': sha(result_root / 'policy.onnx'),
             'onnx_parity': parity, 'recovery': recovery,
             'final_half_second_clearance_lower_bound_deg': min_final_clearance,
             'final_half_second_max_policy_home_error_deg': max_final_home_error,
             'head_records': 162, 'head_repetitions_per_record': 8,
             'qualification': 'Development-seed simulation evidence only; no real robot or servo-zero qualification.',
             'resume_limit': 'Native new episodes; historical per-environment feedback RNG was not checkpointed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS: final checkpoint, ONNX, recovery, HOME, head, and all source archives; {args.output}')


if __name__ == '__main__':
    main()
