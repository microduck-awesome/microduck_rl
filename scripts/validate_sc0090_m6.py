"""Audit the actual frozen M6 mjlab actuator against saved qualified CPU traces."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np
import torch

from mjlab_microduck.sc0090.backend import GpuM6Batch
from mjlab_microduck.sc0090.physics import load_frozen


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def errors(actual, target):
    error = np.abs(actual[:, :2]-target[:, :2])
    return {'q_mae_deg': float(np.rad2deg(error[:, 0].mean())),
            'q_p99_deg': float(np.rad2deg(np.quantile(error[:, 0], .99))),
            'q_max_deg': float(np.rad2deg(error[:, 0].max())),
            'v_rmse_rad_s': float(np.sqrt((error[:, 1]**2).mean()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'development'], required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    study = args.study.resolve()
    ws = study.parents[1]
    reference_root = study/'integration'/('smoke-attempt2' if args.stage == 'smoke' else 'development')
    cpu_report = json.loads((reference_root/'report.json').read_text())
    cpu_protocol = json.loads((reference_root/'pre-run.json').read_text())
    if not cpu_report['passed']:
        raise ValueError('The frozen CPU integration must be qualified first')
    model_path = ws/'sc0090-datasets/v5-model-study/models/m6-seed1/model.json'
    load_frozen(model_path)
    role = 'train' if args.stage == 'smoke' else 'validation'
    input_path = ws/f'sc0090-datasets/v5-identification/inputs/{role}.npz'
    if digest(input_path) != cpu_protocol['input_sha256']:
        raise ValueError('CPU reference input identity changed')
    with np.load(reference_root/'three-way.npz', allow_pickle=False) as f:
        offsets, entries, source, cpu = [f[k] for k in ['offsets', 'entries', 'source', 'mujoco']]
        metadata = json.loads(str(f['metadata']))
    with np.load(input_path, allow_pickle=False) as f:
        all_metadata = json.loads(str(f['metadata']))
        by_id = {m['source_capture_id']: i for i, m in enumerate(all_metadata)}
        indices = [by_id[m['source_capture_id']] for m in metadata]
        settings = f['settings'][indices]
        # Bind the saved reference to the exact initial states and commands.
        expected = np.concatenate([f['entries'][f['offsets'][i]:f['offsets'][i+1]] for i in indices])
        np.testing.assert_array_equal(entries, expected)
    label = f'warp-{args.stage}' + ('' if args.attempt == 1 else f'-attempt{args.attempt}')
    out = study/'integration'/label
    out.mkdir(exist_ok=False)
    gates = json.loads((study/'protocol.json').read_text())['integration_gates']
    code_paths = [Path(__file__).resolve(), *sorted((ws/'microduck_rl/src/mjlab_microduck/sc0090').glob('*.py'))]
    dump(out/'pre-run.json', {'timestamp': datetime.now(timezone.utc).isoformat(), 'records': len(metadata),
        'gates': gates, 'model_sha256': digest(model_path), 'input_sha256': digest(input_path),
        'cpu_reference_sha256': digest(reference_root/'three-way.npz'),
        'source_sha256': {str(p.relative_to(ws)): digest(p) for p in code_paths},
        'environment': {'python': sys.version, 'torch': torch.__version__, 'mujoco': mujoco.__version__,
                        'cuda': torch.version.cuda, 'device': args.device}})
    start = time.monotonic()
    try:
        sim = GpuM6Batch(model_path, settings, [m['mass']+m['arm_mass'] for m in metadata], device=args.device)
        actual = sim.rollout(offsets, entries)
        rows = []
        for i, meta in enumerate(metadata):
            a, b = offsets[i:i+2]
            record = {'run_id': meta['source_capture_id'], 'kp': meta['kp'], 'vin': meta['vin'],
                'group': meta['calibration']['experiment']['group_id'], 'trajectory': meta['trajectory'],
                'warp_vs_source': errors(actual[a:b], source[a:b]),
                'warp_vs_cpu': errors(actual[a:b], cpu[a:b]),
                'warp_vs_truth': errors(actual[a:b], entries[a:b, 1:3]),
                'source_vs_truth': errors(source[a:b], entries[a:b, 1:3]),
                'final_q_source_truth_warp_rad': [float(source[b-1, 0]), float(entries[b-1, 1]), float(actual[b-1, 0])]}
            masks = {'disabled': entries[a:b, 3] == 0, 'slow': abs(entries[a:b, 2]) < .1,
                     'reversal': np.zeros(b-a, dtype=bool)}
            for k in np.flatnonzero(np.diff(np.sign(entries[a:b, 2])) != 0):
                masks['reversal'][max(0, k-10):k+11] = True
            record['segments'] = {name: {'samples': int(mask.sum()),
                'warp_vs_source': errors(actual[a:b][mask], source[a:b][mask]),
                'warp_vs_truth': errors(actual[a:b][mask], entries[a:b, 1:3][mask])}
                for name, mask in masks.items() if mask.any()}
            rows.append(record)
        mean = lambda key, metric: float(np.mean([r[key][metric] for r in rows]))
        checks = {
            'paired_q_mean': mean('warp_vs_source', 'q_mae_deg') <= gates['warp_mean_paired_q_mae_deg'],
            'paired_q_worst_record': max(r['warp_vs_source']['q_mae_deg'] for r in rows) <= gates['warp_max_record_paired_q_mae_deg'],
            'paired_q_global_max': max(r['warp_vs_source']['q_max_deg'] for r in rows) <= gates['warp_global_q_difference_deg'],
            'paired_v_mean': mean('warp_vs_source', 'v_rmse_rad_s') <= gates['warp_mean_paired_v_rmse_rad_s'],
            'physical_MAE_preserved': mean('warp_vs_truth', 'q_mae_deg') <= mean('source_vs_truth', 'q_mae_deg')+gates['physical_mean_MAE_increase_deg'],
        }
        np.savez_compressed(out/'four-way.npz', offsets=offsets, entries=entries, source=source,
                            cpu_mujoco=cpu, warp=actual, metadata=json.dumps(metadata))
        report = {'passed': all(checks.values()), 'full_development_audit': args.stage == 'development' and len(rows) == 844,
            'checks': checks, 'records': len(rows), 'samples': len(actual), 'elapsed_s': time.monotonic()-start,
            'warp_mean_MAE_deg': mean('warp_vs_truth', 'q_mae_deg'), 'source_mean_MAE_deg': mean('source_vs_truth', 'q_mae_deg'),
            'per_record': rows,
            'by_condition': {key: {str(value): {'records': sum(r[key] == value for r in rows),
                'warp_mean_MAE_deg': float(np.mean([r['warp_vs_truth']['q_mae_deg'] for r in rows if r[key] == value])),
                'paired_mean_MAE_deg': float(np.mean([r['warp_vs_source']['q_mae_deg'] for r in rows if r[key] == value]))}
                for value in sorted({r[key] for r in rows})} for key in ['group', 'kp', 'vin', 'trajectory']},
            'scope': 'Actual mjlab custom actuator and Warp body integration; S/O and RL-task contracts separate'}
        dump(out/'report.json', report)
        snapshot = out/'source'
        snapshot.mkdir()
        for path in code_paths:
            shutil.copyfile(path, snapshot/path.name)
        print(json.dumps({k: v for k, v in report.items() if k not in ['per_record', 'by_condition']}), flush=True)
        if not report['passed']:
            raise SystemExit('GPU backend failed; preserve frozen parameters and fix integration')
    except Exception as error:
        dump(out/'error.json', {'type': type(error).__name__, 'message': str(error), 'elapsed_s': time.monotonic()-start})
        raise


if __name__ == '__main__':
    main()
