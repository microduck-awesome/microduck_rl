"""Bounded SC0090 execution A/B using the actual PPO runner.

Run each variant in a fresh process with the same checkpoint, seed and env count.
The output directory must be separate from the source training run. This script
does not manage, resume or modify production training processes.
All variants include the per-world derived-storage correctness fix.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("walk", "recovery"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("reference", "writes", "graphs", "sites", "optimized"),
                        default="optimized")
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.iterations < 2 or args.warmup < 1 or args.envs < 1:
        parser.error("need at least 2 measured iterations, 1 warmup and 1 environment")
    if args.output.resolve() == args.checkpoint.resolve().parent:
        parser.error("output must not be the source checkpoint directory")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.pop("MICRODUCK_WARM_START", None)
    configure_torch_backends()
    torch.set_num_threads(args.threads)
    task = ("Mjlab-Velocity-Flat-MicroDuck-SC0090-V2" if args.task == "walk"
            else "Mjlab-StandUp-Flat-MicroDuck-SC0090-V2")
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = args.envs
    cfg.seed = 42
    if args.variant not in ("sites", "optimized"):
        cfg.scene.spec_fn = None
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.fast_friction_writes = args.variant in ("writes", "optimized")
    cfg.events["cache_reset_constants"].params["enabled"] = args.variant in ("graphs", "optimized")
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        vec = RslRlVecEnvWrapper(env)
        agent = asdict(load_rl_cfg(task))
        agent["logger"] = "tensorboard"
        runner = load_runner_cls(task)(vec, agent, str(args.output), device="cuda:0")
        runner.load(str(args.checkpoint), map_location="cuda:0")
        samples = []
        original_log = runner.logger.log
        previous_finish = None

        def log(**kwargs):
            nonlocal previous_finish
            original_log(**kwargs, print_minimal=True)
            # Include logging and all asynchronous GPU work in the end-to-end
            # interval. Checkpoint writes appear in the following interval.
            torch.cuda.synchronize()
            finish = time.perf_counter()
            samples.append({"iteration": kwargs["it"],
                            "wall_s": None if previous_finish is None else finish - previous_finish,
                            "end_unix": time.time(),
                            "collection_s": kwargs["collect_time"],
                            "learning_s": kwargs["learn_time"]})
            previous_finish = finish

        runner.logger.log = log
        runner.learn(args.warmup + args.iterations, init_at_random_ep_len=True)
        measured = samples[args.warmup:]
        median = statistics.median(s["wall_s"] for s in measured)
        cache = getattr(env.sim, "_microduck_recompute_cache", None)
        result = {
            "task": task, "variant": args.variant, "checkpoint": str(args.checkpoint.resolve()),
            "envs": args.envs, "threads": args.threads, "warmup": args.warmup,
            "samples": samples, "median_wall_s": median,
            "steps_per_second": args.envs * agent["num_steps_per_env"] / median,
            "cuda_graphs": env.sim.use_cuda_graph,
            "model_site_count": env.sim.mj_model.nsite,
            "per_world_recompute_storage": True,
            "recompute_graphs": [] if cache is None else [level.name for level in cache.graphs],
        }
        if args.export:
            import onnxruntime as ort
            from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx

            path = args.output / "policy.onnx"
            runner.export_policy_to_onnx(str(args.output), path.name)
            saved_checkpoint = args.output / f"model_{runner.current_learning_iteration}.pt"
            attach_metadata_to_onnx(str(path), get_base_metadata(env, run_path=str(saved_checkpoint)))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            # Training enables TF32. Compare export against full FP32 inference
            # so the validation does not measure TF32-vs-CPU rounding instead.
            previous_precision = torch.backends.cuda.matmul.fp32_precision
            try:
                torch.backends.cuda.matmul.fp32_precision = "ieee"
                with torch.inference_mode():
                    obs = vec.get_observations()
                    expected = runner.get_inference_policy()(obs).cpu().numpy()
                    actor_obs = obs["actor"].cpu().numpy()
            finally:
                torch.backends.cuda.matmul.fp32_precision = previous_precision
            actual = np.concatenate([session.run(None, {session.get_inputs()[0].name: x[None]})[0]
                                     for x in actor_obs])
            assert actor_obs.shape[1] == 61 and actual.shape == (args.envs, 14)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
            result["onnx"] = {"shape": list(actual.shape), "finite": True,
                              "max_abs_error": float(np.max(np.abs(actual - expected)))}
        (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        print("BENCHMARK_RESULT", json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
