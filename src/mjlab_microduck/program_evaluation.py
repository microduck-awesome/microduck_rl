"""Frozen V4 policy assessment with separate retention and forced-challenge groups."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks import SC0090FineTuneRunner, mdp


def evaluate_seed(request, configs, seed):
    cfg = deepcopy(configs["env"])
    recovery = request["task"] == "recovery"
    names = mdp.SC0090_RECOVERY_BUCKETS if recovery else tuple(k for k, _ in mdp.SC0090_PROGRAM_WALK_COMMANDS)
    samples = request["samples_per_group"]
    cfg.scene.num_envs = len(names)*2*samples
    cfg.seed, cfg.auto_reset, cfg.episode_length_s = seed, False, 10.
    for command in cfg.commands.values():
        command.resampling_time_range = (1000., 1000.)
    if recovery:
        cfg.events["set_ground_state"].params["evaluation_buckets"] = True
    env = ManagerBasedRlEnv(cfg, device=request["device"])
    try:
        env._sc0090_program_eval_samples = samples
        vec = RslRlVecEnvWrapper(env, clip_actions=configs["agent"]["clip_actions"])
        runner = SC0090FineTuneRunner(vec, deepcopy(configs["agent"]), device=request["device"])
        infos = runner.load(request["checkpoint"], load_cfg={"actor": True}, map_location=request["device"])
        mdp.sc0090_program_restore(env, infos["sc0090_program"])
        if mdp.sc0090_program_state(env)["phase_index"] != request["phase_index"]:
            raise ValueError("Checkpoint phase mismatch")
        if recovery:
            mdp._sc0090_recovery_state(env).load_state_dict(infos["sc0090_recovery"])
        policy = runner.get_inference_policy(device=request["device"])
        with torch.inference_mode():
            env.reset(seed=seed)
            mdp.sc0090_program_evaluation_commands(env, samples)
            obs = vec.get_observations()
            if obs["actor"].shape != (env.num_envs, 61):
                raise ValueError("Actor observation contract changed")
            n, device = env.num_envs, env.device
            failed = torch.zeros(n, device=device, dtype=torch.bool)
            first_rise = torch.full((n,), 8., device=device)
            seen_rise = torch.zeros_like(failed)
            action_ss = torch.zeros(n, device=device)
            torque_ss = torch.zeros(n, device=device)
            settled_ss = torch.zeros(n, device=device)
            xy_error = torch.zeros(n, device=device)
            yaw_error = torch.zeros(n, device=device)
            pose_error = torch.zeros(n, 3, device=device)
            previous_action = previous_torque = None
            steps = round(8./env.step_dt)
            settled_steps = 0
            push_times = {round(3./env.step_dt): 0, round(5./env.step_dt): 1}
            for step in range(steps):
                if step in push_times:
                    mdp.sc0090_program_evaluation_push(env, push_times[step])
                    obs = vec.get_observations()
                actions = policy(obs, stochastic_output=False)
                if actions.shape != (n, 14) or not torch.isfinite(actions).all().item():
                    raise ValueError("Invalid policy actions")
                obs, reward, done, _ = vec.step(actions)
                values = [reward, env.sim.data.qpos, env.sim.data.qvel, *obs.values()]
                if not all(torch.isfinite(v).all().item() for v in values):
                    raise ValueError("Nonfinite assessment rollout")
                data = env.scene["robot"].data
                torque = data.actuator_force
                if previous_action is not None:
                    action_ss += (actions-previous_action).square().mean(-1)
                    torque_ss += (torque-previous_torque).square().mean(-1)
                previous_action, previous_torque = actions.clone(), torque.clone()
                if recovery:
                    success = mdp._sc0090_recovery_state(env).success
                    first_rise = torch.where(success & ~seen_rise & ~failed,
                                              torch.full_like(first_rise, (step+1)*env.step_dt), first_rise)
                    seen_rise |= success
                if step >= round(6./env.step_dt):
                    settled_steps += 1
                    settled_ss += data.root_link_ang_vel_b.square().mean(-1)
                    cmd = env.command_manager.get_command("twist")
                    xy_error += (data.root_link_lin_vel_b[:, :2]-cmd[:, :2]).norm(dim=-1)
                    yaw_error += (data.root_link_ang_vel_b[:, 2]-cmd[:, 2]).abs()
                    pose_error += torch.stack(mdp.sc0090_program_pose_errors(env), -1).abs()
                failed |= done.bool()
                if done.any().item():
                    # Terminated worlds are permanent failures for this trial.
                    # Reset only to satisfy mjlab's manual-reset API; do not
                    # count a second episode or discard a difficult start.
                    env.reset(env_ids=done.nonzero().squeeze(-1))
                    mdp.sc0090_program_evaluation_commands(env, samples)
                    obs = vec.get_observations()
            phase = mdp.sc0090_program_phase(env)
            rehearsal = mdp.sc0090_program_buffers(env)["rehearsal"]
            if recovery:
                nominal = mdp._sc0090_recovery_state(env).hold >= math.ceil(.5/env.step_dt)
                target = mdp.sc0090_program_buffers(env)["pose_hold"] >= math.ceil(.5/env.step_dt)
                success = seen_rise & torch.where((~rehearsal) & (phase["body_weight"] > 0), target, nominal)
            else:
                cmd = env.command_manager.get_command("twist")
                success = ((xy_error/settled_steps <= .035+.25*cmd[:, :2].norm(dim=-1))
                           & (yaw_error/settled_steps <= .15+.25*cmd[:, 2].abs()))
                if phase["body_weight"] > 0:
                    pose_ok = ((pose_error[:, 0]/settled_steps <= .01)
                               & (pose_error[:, 1:].amax(-1)/settled_steps <= math.radians(5)))
                    success &= rehearsal | pose_ok
            success &= ~failed
            ids = torch.arange(n, device=device)
            groups = ids // (2*samples)
            result = {"seed": seed, "finite": True}
            for profile, mask in (("retention", rehearsal), ("challenge", ~rehearsal)):
                result[profile] = {}
                for i, name in enumerate(names):
                    selected = mask & (groups == i)
                    result[profile][name] = {"trials": int(selected.sum()), "wins": int(success[selected].sum())}
            challenge = ~rehearsal
            result["metrics"] = {
                "action_delta_rms": float((action_ss[challenge].mean()/max(steps-1, 1)).sqrt()),
                "torque_delta_rms": float((torque_ss[challenge].mean()/max(steps-1, 1)).sqrt()),
                "settled_ang_rms": float((settled_ss[challenge].mean()/settled_steps).sqrt()),
                "rise_time_p95": float(torch.quantile(first_rise[challenge], .95)) if recovery else 0.,
            }
            print(f"[Program evaluation seed {seed}] {result}", flush=True)
            return result
    finally:
        torch.cuda.synchronize()
        wp.synchronize()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request["schema"] != 2 or request["duration_s"] != 8.:
        raise ValueError("Unsupported assessment protocol")
    for key in ("config", "checkpoint"):
        if hashlib.sha256(Path(request[key]).read_bytes()).hexdigest() != request[key+"_sha256"]:
            raise ValueError(f"Assessment input changed: {key}")
    # Input is written by the local runner; never deserialize untrusted inputs.
    with Path(request["config"]).open("rb") as stream:
        configs = pickle.load(stream)
    configure_torch_backends()
    torch.set_num_threads(2)
    torch.cuda.set_device(request["device"])
    started = time.monotonic()
    cases = []
    for seed in request["seeds"]:
        cases.append(evaluate_seed(request, configs, seed))
        gc.collect()
        torch.cuda.empty_cache()
    report = {**request, "cases": cases, "wall_seconds": time.monotonic()-started,
              "protocol": "retention: no push/body command; challenge: phase body commands, forced pushes at 3/5 s; full DR/noise"}
    tmp = args.request.parent / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    tmp.replace(args.request.parent / "report.json")


if __name__ == "__main__":
    main()
