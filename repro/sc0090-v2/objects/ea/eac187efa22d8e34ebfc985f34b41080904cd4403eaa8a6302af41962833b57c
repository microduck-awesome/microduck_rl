"""Audit simultaneous multi-axis arrival using saved, unmodified head rollouts."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def arrival(position, target, amplitude, dt=.02, hold_s=.1):
    active = amplitude.abs() > 1e-9
    tolerance = torch.maximum(.1 * amplitude.abs(), amplitude.new_full((), .002))
    band = (position - target[None]).abs() <= tolerance[None]
    sustained = band.unfold(0, math.ceil(hold_s / dt), 1).all(-1)
    times = torch.arange(1, len(sustained) + 1, device=position.device) * dt
    independent = torch.where(sustained, times[:, None, None], float('inf')).amin(0)
    simultaneous = (sustained | ~active[None]).all(-1)
    joint_time = torch.where(simultaneous, times[:, None], float('inf')).amin(0)
    return independent, joint_time, active


def guard():
    # Each axis arrives, but their qualifying intervals never overlap.
    x = torch.zeros(10, 1, 4)
    x[:5, 0, 0] = 1.
    x[5:, 0, 1] = 1.
    goal = torch.tensor([[1., 1., 0., 0.]])
    separate, joint, _ = arrival(x, goal, goal)
    assert torch.isfinite(separate[0, :2]).all() and torch.isinf(joint[0])
    x[:, 0, :2] = 1.
    _, joint, _ = arrival(x, goal, goal)
    torch.testing.assert_close(joint, torch.tensor([.02]))


def summarize(report_path, device):
    report = json.loads(report_path.read_text())
    records = report['records']
    result = dict(
        checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256'],
        source_report=str(report_path.resolve()),
        source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        metric='All commanded axes simultaneously within max(10% step, .002 rad) for .1 s',
        inactive_axes='Excluded from arrival gate; their steady absolute errors retained separately',
        scope='Saved standing development rollouts; not hardware, disturbance, or held-out qualification',
        summary={}, records=[], traces_sha256={}, per_axis_existing_report_parity=True)
    groups = defaultdict(lambda: dict(trials=0, simultaneous_pass=0, independent_all_pass=0,
                                     final_half_s_hold_pass=0, invalid_trials=0))
    for seed in sorted({r['seed'] for r in records}):
        rows = [r for r in records if r['seed'] == seed]
        trace_path = report_path.parent / f'traces-{seed}.npz'
        result['traces_sha256'][str(trace_path.resolve())] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        with np.load(trace_path) as raw:
            pos = torch.as_tensor(raw['physical_head_rad'], device=device)
            commands = torch.as_tensor(raw['command_rad'], device=device)
            invalid = torch.as_tensor(raw['invalid'], device=device).bool()
        target = torch.cat([torch.tensor(r['target'], device=device).repeat(r['trials'], 1) for r in rows])
        assert pos.shape == (300, len(target), 4)
        torch.testing.assert_close(commands[100:200], target[None].expand(100, -1, -1), rtol=0, atol=0)
        for phase, start, end, desired in [('step', 100, 200, target),
                                           ('return', 200, 300, torch.zeros_like(target))]:
            independent, joint, active = arrival(pos[start:end], desired, target)
            bad = invalid[start:end].any(0)
            assert bad.shape == joint.shape
            error = (pos[end-25:end] - desired[None]).abs().mean(0) * (180 / math.pi)
            tolerance = torch.maximum(.1 * target.abs(), target.new_full((), .002))
            tail_band = ((pos[end-25:end] - desired[None]).abs() <= tolerance[None]) | ~active[None]
            tail_joint = tail_band.all(-1) & ~bad[None]
            tail_hold = tail_joint.all(0)
            offset = 0
            for r in rows:
                n = r['trials']; sl = slice(offset, offset+n); offset += n
                count = sum(abs(x) > 1e-9 for x in r['target'])
                if not count:
                    continue
                independent_pass = torch.isfinite(independent[sl])
                for axis, on in enumerate(active[sl][0]):
                    if on:
                        expected = round(r[phase]['time90_reached_fraction'][axis] * n)
                        assert int(independent_pass[:, axis].sum()) == expected, (r['name'], seed, phase, axis)
                all_independent = (independent_pass | ~active[sl]).all(-1) & ~bad[sl]
                passed = torch.isfinite(joint[sl]) & ~bad[sl]
                assert bool((~passed | all_independent).all())
                key = f"{phase}/{r['range']}/{count}-axis"
                g = groups[key]; g['trials'] += n
                g['simultaneous_pass'] += int(passed.sum())
                g['independent_all_pass'] += int(all_independent.sum())
                g['final_half_s_hold_pass'] += int(tail_hold[sl].sum())
                g['invalid_trials'] += int(bad[sl].sum())
                times = torch.where(bad[sl], float('inf'), joint[sl]).sort().values
                median = float(times[(n-1)//2]); p95 = float(times[math.ceil(.95*n)-1])
                result['records'].append(dict(
                    name=r['name'], seed=seed, phase=phase, range=r['range'], active_axes=count,
                    trials=n, simultaneous_pass=int(passed.sum()),
                    independent_all_pass=int(all_independent.sum()),
                    final_half_s_hold_pass=int(tail_hold[sl].sum()),
                    final_half_s_band_fraction=float(tail_joint[:, sl].float().mean()),
                    joint_t90_median_all_trials_s=median if math.isfinite(median) else None,
                    joint_t90_p95_all_trials_s=p95 if math.isfinite(p95) else None,
                    all_axis_steady_mae_deg=error[sl].mean(0).tolist()))
    result['summary'] = dict(groups)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    guard()
    args.output.mkdir(exist_ok=False, parents=True)
    results = [summarize(p, args.device) for p in args.reports]
    out = dict(schema=1, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               synthetic_nonoverlap_guard_pass=True, reports=results)
    (args.output/'report.json').write_text(json.dumps(out, indent=2))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    for r in results:
        print(Path(r['source_report']).parent.name, json.dumps(r['summary']), flush=True)
