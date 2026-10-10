"""Deterministic GPU validation on training-distribution simulation trajectories.

Uses independent seeds and true simulated state ONLY for scoring. The actor
keeps the frozen causal4D input; no Sunday or cross-model test data are read.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from mjlab_microduck.tasks.sc0090_env_cfg import make_sc0090_env_cfg, make_sc0090_rl_cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(policy, checkpoint, onnx, protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    rule = protocol['convergence']
    number = int(policy[1:]); S = f'S{number%2}'; O = f'O{number//2%2}'; R = f'R{number//4}'
    output.mkdir(parents=True, exist_ok=False)
    session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'],
                                  sess_options=_ort_options())
    arrays = {}; trials = []; errors = []; families = []; holds = []; faults = 0
    maximum_difference = 0.
    torch.set_num_threads(1)
    for trial, (seed, target_seed) in enumerate(zip(rule['environment_seeds'], rule['trajectory_seeds'], strict=True)):
        cfg = make_sc0090_env_cfg(S, O, R=R)
        cfg.scene.num_envs = rule['validation_envs']; cfg.seed = seed
        cfg.auto_reset = False; cfg.episode_length_s = (rule['validation_ticks']+1)*.02
        cfg.commands['target'].trajectory_seed = target_seed
        cfg.commands['target'].resampling_time_range = (1000., 1000.)
        env = None; frames = []; trial_faults = 0; failure = None
        try:
            with torch.inference_mode():
                env = ManagerBasedRlEnv(cfg, device='cuda:0')
                wrapped = RslRlVecEnvWrapper(env)
                runner = MjlabOnPolicyRunner(wrapped, asdict(make_sc0090_rl_cfg()), None, 'cuda:0')
                runner.load(str(checkpoint), map_location='cuda:0')
                actor = runner.alg.get_policy().as_onnx(verbose=False).to('cuda:0').eval()
                env.reset()
                target = env.command_manager.get_term('target')
                ids = torch.arange(env.num_envs, device='cuda:0')
                # Exact20% hold; each of the four nonzero trajectory families20%.
                family = ids % 5
                target.kind.copy_((family-1).clamp(min=0))
                target.amplitude[family == 0] = 0.
                # Avoid inheriting randomly sampled zero amplitude in other families.
                generator = torch.Generator(device='cuda:0').manual_seed(target_seed)
                random = torch.rand((env.num_envs, 2), generator=generator, device='cuda:0')
                amplitude = (.2+.5*random[:, 0])*torch.where(random[:, 1] < .5, -1., 1.)
                target.amplitude[family != 0] = amplitude[family != 0]
                target._update_command()
                obs = env.observation_manager.compute(update_history=True)['actor']
                if not torch.equal(obs[:, 3], target.command[:, 0]):
                    raise RuntimeError('Initial actor target differs from the current validation command')
                expected = actor(obs).cpu().numpy()
                # Deployed ONNX deliberately has a fixed batch1 interface.
                # Check every validation world through that actual contract.
                inputs = obs.cpu().numpy()
                actual = np.concatenate([session.run(None, {session.get_inputs()[0].name: x[None]})[0]
                                         for x in inputs], axis=0)
                maximum_difference = max(maximum_difference, float(np.max(np.abs(expected-actual))))
                if maximum_difference >= 1e-5:
                    raise RuntimeError('Validation actor does not match exported ONNX')
                condition = env.action_manager.get_term('position').pipeline
                condition_ids = condition.condition_index.cpu().numpy() if R == 'R1' else np.zeros(env.num_envs, dtype=np.int64)
                for tick in range(rule['validation_ticks']):
                    observations, reward, done, timeout, extras = env.step(actor(obs))
                    q = env.scene['bench'].data.joint_pos[:, 0]
                    reference = target.value_at(env.episode_length_buf*env.step_dt)
                    frames.append(torch.stack([q, reference], -1).cpu().numpy())
                    trial_faults += int(done.sum())
                    if bool(done.any()) or bool(timeout.any()):
                        failure = 'terminal_before_validation_end'; break
                    obs = observations['actor']
                traces = np.asarray(frames, dtype=np.float32)
                arrays[f'q_target_trial{trial}'] = traces
                arrays[f'family_trial{trial}'] = family.cpu().numpy()
                arrays[f'condition_trial{trial}'] = condition_ids
                if len(frames) == rule['validation_ticks'] and not failure:
                    error = np.abs(traces[:, :, 0]-traces[:, :, 1])*180/np.pi
                    errors.append(error.reshape(-1)); families.append((error, family.cpu().numpy()))
                    hold = traces[rule['hold_scoring_start_tick']:, family.cpu().numpy() == 0, 0]*180/np.pi
                    holds.extend(np.std(hold, axis=0).tolist())
        except Exception as exc:
            failure = type(exc).__name__+': '+str(exc)
            trial_faults += 1
            if frames: arrays[f'q_target_trial{trial}'] = np.asarray(frames, dtype=np.float32)
        finally:
            if env is not None: env.close()
        faults += trial_faults
        trials.append({'seed': seed, 'trajectory_seed': target_seed, 'ticks': len(frames),
                       'fault_count': trial_faults, 'failure': failure})
        if failure: break
    raw = output/'traces.npz'; np.savez_compressed(raw, **arrays)
    full = len(trials) == len(rule['environment_seeds']) and all(t['failure'] is None for t in trials)
    metric = np.concatenate(errors) if errors else np.array([])
    by_family = {}
    for i, name in enumerate(['zero', 'step', 'reversal', 'sine', 'chirp']):
        selected = [e[:, f == i].reshape(-1) for e, f in families]
        values = np.concatenate(selected) if selected else np.array([])
        by_family[name] = {'mae_deg': float(np.mean(values)) if len(values) else None,
                           'p99_deg': float(np.quantile(values, .99)) if len(values) else None}
    report = {'passed': full and faults == 0, 'fault_count': faults,
              'mae_deg': float(np.mean(metric)) if len(metric) else None,
              'p99_deg': float(np.quantile(metric, .99)) if len(metric) else None,
              'zero_hold_std_deg': float(np.mean(holds)) if holds else None,
              'families': by_family, 'trials': trials, 'policy': policy,
              'protocol_sha256': digest(protocol_path), 'checkpoint_sha256': digest(checkpoint),
              'onnx_sha256': digest(onnx), 'raw_sha256': digest(raw),
              'source_sha256': digest(__file__), 'maximum_torch_onnx_difference': maximum_difference,
              'hardware_executed': False, 'sim2real_benefit_verified': False,
              'scope': 'Independent seeded simulation validation in each training route; fixed training-side model/conditions. No cross-model, recorded development or Sunday test selection.'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def _ort_options():
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True); parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True); parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = validate(args.policy, args.checkpoint, args.onnx, args.protocol, args.output)
    print(json.dumps({k: report[k] for k in ['passed', 'fault_count', 'mae_deg', 'p99_deg', 'zero_hold_std_deg']}), flush=True)


if __name__ == '__main__': main()
