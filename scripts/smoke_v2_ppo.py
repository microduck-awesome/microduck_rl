"""Local PPO smoke with the stock Microduck runner and V2 official baseline task.

Uses 64 environments, five updates and local TensorBoard by default. Artifacts are temporary
validation products, not a trained policy for deployment.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["WANDB_MODE"] = "disabled"
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.custom_v2.microduck_v2 import (
    TASK_ID,
    POLICY_JOINTS,
    ASSETS,
    RUNTIME,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / (output.name + ".onnx")
    onnx_path.unlink(missing_ok=True)  # Do not let a stale validation export pass.
    torch.set_num_threads(2)
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("Positive env/iteration counts required")
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    agent = load_rl_cfg(args.task)
    agent.logger = "tensorboard"
    agent.upload_model = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    model_hash = hashlib.sha256((ASSETS / "robot_walk.xml").read_bytes()).hexdigest()
    contact_capacity = env.sim.wp_data.naconmax
    contact_peak = 0
    physics_step = env.sim.step

    def checked_physics_step():
        nonlocal contact_peak
        physics_step()
        count = int(env.sim.wp_data.nacon.numpy()[0])
        contact_peak = max(contact_peak, count)
        assert count <= contact_capacity, (
            "Warp contact buffer overflow",
            count,
            contact_capacity,
        )

    env.sim.step = checked_physics_step
    started = time.monotonic()
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(args.task)(
            wrapped, asdict(agent), str(output), args.device
        )
        actor = runner.alg.get_policy()
        before = [p.detach().clone() for p in actor.parameters()]
        updates = []
        original_update = runner.alg.update

        def check_update():
            result = original_update()
            values = {k: float(v) for k, v in result.items()}
            assert all(math.isfinite(v) for v in values.values()), values
            assert all(torch.isfinite(p).all() for p in actor.parameters())
            updates.append(values)
            return result

        runner.alg.update = check_update
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        changes = [
            float((p.detach() - old).abs().max())
            for p, old in zip(actor.parameters(), before)
        ]
        assert len(updates) == args.iterations and max(changes) > 0
        report = dict(
            status="PASS",
            task=args.task,
            device=args.device,
            envs=args.num_envs,
            iterations=args.iterations,
            steps_per_env=agent.num_steps_per_env,
            learning_epochs=agent.algorithm.num_learning_epochs,
            minibatches=agent.algorithm.num_mini_batches,
            losses=updates,
            policy_parameter_max_change=max(changes),
            finite_policy_parameters=True,
            elapsed_seconds=time.monotonic() - started,
            robot_walk_sha256=model_hash,
            action_scale=cfg.actions["joint_pos"].scale,
            peak_contacts_across_worlds=contact_peak,
            contact_capacity_across_worlds=contact_capacity,
            contact_capacity_checked_each_physics_step=True,
            logging="local TensorBoard",
            upload_model=False,
            limitation="A short PPO smoke validates the training path; it does not demonstrate learned walking or SC0090 sim2real.",
        )
        report["onnx_exported"] = onnx_path.is_file()
        assert (
            onnx_path.is_file()
        ), "The runner failed to export ONNX; inspect its warning."
        if onnx_path.is_file():
            import onnx
            import onnxruntime as ort

            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            metadata = {p.key: p.value for p in model.metadata_props}
            assert metadata["microduck_runtime"] == RUNTIME
            assert metadata["microduck_deployment"] == "simulation-baseline-only"
            assert "microduck_training_contract" not in metadata
            report["runtime"] = metadata["microduck_runtime"]
            report["feedback"] = metadata["microduck_feedback"]
            report["motor_sha256"] = metadata["microduck_motor_sha256"]
            assert not hasattr(env.unwrapped, "_v2_joint_feedback")
            assert wrapped.get_observations()["actor"].shape == (args.num_envs, 61)
            assert wrapped.get_observations()["critic"].shape == (args.num_envs, 76)
            joint_names = metadata.get("joint_names", "").split(",")
            assert tuple(joint_names) == POLICY_JOINTS, joint_names
            for name in ("default_joint_pos", "joint_stiffness", "joint_damping"):
                assert len(metadata[name].split(",")) == 14, name
            assert math.isclose(
                float(metadata["action_scale"]), cfg.actions["joint_pos"].scale
            )
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            actor.eval()
            error = 0.0
            samples = (
                np.zeros((1, 61), dtype=np.float32),
                np.linspace(-0.1, 0.1, 61, dtype=np.float32)[None],
                wrapped.get_observations()["actor"][:1].detach().cpu().numpy(),
            )
            for x in samples:
                result = session.run(None, {session.get_inputs()[0].name: x})[0]
                assert result.shape == (1, 14) and np.isfinite(result).all()
                with torch.inference_mode():
                    expected = (
                        actor({"actor": torch.as_tensor(x, device=args.device)})
                        .cpu()
                        .numpy()
                    )
                np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)
                error = max(error, float(np.abs(result - expected).max()))
            report["onnx_cpu_inference"] = {
                "input": [1, 61],
                "output": [1, 14],
                "metadata_joint_count": len(joint_names),
                "normalizer_parity_max_error": error,
            }
        destination = args.report or output / "report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
