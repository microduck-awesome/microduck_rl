"""Bounded, isolated recovery assessment. Input is a trusted local runner request."""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
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
    cfg.scene.num_envs = len(mdp.SC0090_RECOVERY_BUCKETS) * request["samples_per_group"]
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = request["duration_s"] + 2.0
    cfg.curriculum = {}
    cfg.events["set_ground_state"].params["evaluation_buckets"] = True
    # Retain training sensor noise, DR, pushes and sampled commands. Only the
    # initial pose mixture and deterministic policy differ from training.
    env = ManagerBasedRlEnv(cfg, device=request["device"])
    try:
        vec = RslRlVecEnvWrapper(env, clip_actions=configs["agent"]["clip_actions"])
        runner = SC0090FineTuneRunner(vec, deepcopy(configs["agent"]), device=request["device"])
        infos = runner.load(request["checkpoint"], load_cfg={"actor": True},
                            strict=True, map_location=request["device"])
        state = mdp._sc0090_recovery_state(env)
        state.load_state_dict(infos["sc0090_recovery"])
        if state.stage != request["stage"]:
            raise ValueError("Checkpoint frontier disagrees with evaluation request")
        policy = runner.get_inference_policy(device=request["device"])
        with torch.inference_mode():
            env.reset(seed=seed)
            obs = vec.get_observations()
            if obs["actor"].shape != (env.num_envs, 61):
                raise ValueError("Recovery actor observation contract changed")
            steps = round(request["duration_s"] / env.step_dt)
            for step in range(steps):
                actions = policy(obs, stochastic_output=False)
                if actions.shape != (env.num_envs, 14):
                    raise ValueError("Recovery action contract changed")
                obs, reward, done, _ = vec.step(actions)
                if done.any().item():
                    raise ValueError(f"Unexpected termination in bounded evaluation at step {step}")
                values = [actions, reward, env.sim.data.qpos, env.sim.data.qvel, *obs.values()]
                if not all(torch.isfinite(value).all().item() for value in values):
                    raise ValueError(f"Nonfinite recovery evaluation at step {step}")
            groups = {}
            for i, name in enumerate(mdp.SC0090_RECOVERY_BUCKETS):
                selected = state.bucket == i
                groups[name] = {"trials": int(selected.sum().item()),
                                "wins": int(state.success[selected].sum().item())}
            print(f"[Evaluation seed {seed}] {groups}", flush=True)
            return {"seed": seed, "finite": True, "groups": groups}
    finally:
        torch.cuda.synchronize()
        wp.synchronize()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request["schema"] != 1 or request["deterministic"] is not True or request["duration_s"] != 8.0:
        raise ValueError("Unsupported recovery assessment protocol")
    for key in ("checkpoint", "config"):
        if hashlib.sha256(Path(request[key]).read_bytes()).hexdigest() != request[f"{key}_sha256"]:
            raise ValueError(f"Changed evaluation input: {key}")
    # The runner creates this pickle locally; never load externally supplied inputs.
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
    report = {**request, "cases": cases, "wall_seconds": time.monotonic() - started,
              "protocol": "training DR/noise/commands/pushes; fixed pose groups; deterministic actor"}
    temporary = args.request.parent / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.request.parent / "report.json")


if __name__ == "__main__":
    main()
