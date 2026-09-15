"""Full-state continuation and isolated assessments for both SC0090 V4 tasks."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shutil

import cloudpickle
import torch
import yaml
from mjlab.rl import MjlabOnPolicyRunner

from . import SC0090FineTuneRunner, mdp
from mjlab_microduck.actuator.sc0090 import SC0090_MODEL_PATH


def identify_checkpoint_task(path, infos, explicit="auto"):
    tagged = (infos.get("sc0090_program") or {}).get("task")
    if tagged is None and "sc0090_recovery" in infos:
        tagged = "recovery"
    sidecar = Path(path).parent / "params" / "agent.yaml"
    if tagged is None and sidecar.exists():
        name = yaml.load(sidecar.read_text(), Loader=yaml.BaseLoader).get("experiment_name", "")
        for kind in ("walk", "recovery"):
            if name == f"sc0090_{kind}" or name.startswith(f"sc0090_{kind}_"):
                tagged = kind
    if explicit != "auto" and tagged is not None and explicit != tagged:
        raise ValueError("Explicit legacy task disagrees with checkpoint provenance")
    if tagged is None:
        tagged = explicit if explicit != "auto" else None
    if tagged not in ("walk", "recovery"):
        raise ValueError("Legacy checkpoint task unknown: retain params/agent.yaml or set training_program.legacy_task")
    return tagged


class SC0090ProgramRunner(SC0090FineTuneRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        if os.environ.get("MICRODUCK_WARM_START", "0") not in ("", "0"):
            raise ValueError("V4 preserves global counters; unset MICRODUCK_WARM_START. Omit checkpoint for fresh training.")
        self._evaluation_train_cfg = deepcopy(train_cfg)
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        cfg = self.cfg["training_program"]
        if self.is_distributed or torch.device(device).type != "cuda":
            raise ValueError("V4's isolated evaluation currently requires a single CUDA training GPU")
        if (cfg["interval_iterations"] < 1 or cfg["samples_per_group"] < 128
                or len(cfg["seeds"]) < 2 or len(set(cfg["seeds"])) != len(cfg["seeds"])
                or not math.isfinite(cfg["timeout_seconds"]) or cfg["timeout_seconds"] <= 0):
            raise ValueError("Invalid program evaluation configuration")
        self._abi = {"actor_obs": 61, "actions": 14,
                     "motor_sha256": hashlib.sha256(SC0090_MODEL_PATH.read_bytes()).hexdigest()}
        if cfg["checkpoint"]:
            if self.cfg.get("resume"):
                raise ValueError("Use either training_program.checkpoint or agent.resume, not both")
            self.load(cfg["checkpoint"])

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        task = mdp.sc0090_program_spec(self.env.unwrapped)["task"]
        if identify_checkpoint_task(path, infos, self.cfg["training_program"]["legacy_task"]) != task:
            raise ValueError("Cannot transfer walking/recovery normalizers across task families")
        metadata = infos.get("sc0090_program_runner", {})
        if metadata and metadata["abi"] != self._abi:
            raise ValueError("Checkpoint motor/observation/action contract changed")
        # The parent restores actor, critic, optimizer and global environment step.
        restored = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location or self.device)
        if load_cfg is None:
            learning_rate = metadata.get("learning_rate", self.alg.optimizer.param_groups[0]["lr"])
            if (not math.isfinite(learning_rate) or learning_rate <= 0
                    or any(group["lr"] != learning_rate for group in self.alg.optimizer.param_groups)):
                raise ValueError("Checkpoint adaptive/optimizer learning rates disagree")
            self.alg.learning_rate = learning_rate
            # RSL's checkpoint iteration names the completed update, while learn()
            # begins at current_learning_iteration. Resume at the NEXT update label.
            self.current_learning_iteration = metadata.get("next_iteration", checkpoint["iter"]+1)
            if self.current_learning_iteration != checkpoint["iter"]+1:
                raise ValueError("Invalid next-iteration checkpoint state")
        mdp.sc0090_program_restore(self.env.unwrapped, infos.get("sc0090_program"),
                                   infos.get("sc0090_recovery"))
        with torch.inference_mode():
            # mjlab does not serialize physics episodes. Reapply the restored
            # live settings before resetting them; keep global training counters.
            self.env.unwrapped.reset()
        print(f"[Training program restored] task={task} next_iteration={self.current_learning_iteration} "
              f"phase={mdp.sc0090_program_phase(self.env.unwrapped)['name']} lr={self.alg.learning_rate}", flush=True)
        return restored

    def _infos(self, infos):
        recovery = getattr(self.env.unwrapped, "_sc0090_recovery", None)
        if recovery is not None:
            infos = {**(infos or {}), "sc0090_recovery": recovery.state_dict()}
        return {**(infos or {}), "sc0090_program": deepcopy(mdp.sc0090_program_state(self.env.unwrapped)),
                "sc0090_program_runner": {"abi": self._abi,
                    "learning_rate": self.alg.learning_rate,
                    "next_iteration": self.current_learning_iteration+1}}

    def save(self, path, infos=None):
        super().save(path, self._infos(infos))
        env = self.env.unwrapped
        cfg = self.cfg["training_program"]
        state = mdp.sc0090_program_state(env)
        step = env.common_step_counter
        interval = cfg["interval_iterations"] * self.cfg["num_steps_per_env"]
        if state["last_attempt_step"] >= 0 and step-state["last_attempt_step"] < interval:
            return
        state["last_attempt_step"] = step
        folder = Path(path).parent / "program_evaluations" / f"step_{step:09d}"
        try:
            folder.mkdir(parents=True, exist_ok=False)
            frozen = folder / "model.pt"
            shutil.copyfile(path, frozen)
            config = folder / "config.pkl"
            with config.open("wb") as stream:
                cloudpickle.dump({"env": env.cfg, "agent": self._evaluation_train_cfg}, stream)
            request = dict(schema=2, task=state["task"], program_hash=state["program_hash"],
                phase_index=state["phase_index"], global_step=step,
                iteration=self.current_learning_iteration, checkpoint=str(frozen.resolve()),
                checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                config=str(config.resolve()), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                seeds=list(cfg["seeds"]), samples_per_group=cfg["samples_per_group"], duration_s=8., device=str(self.device))
            request_path = folder / "request.json"
            request_path.write_text(json.dumps(request, indent=2)+"\n")
            child_env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="", OMP_NUM_THREADS="2",
                             MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
            child_env.pop("MICRODUCK_WARM_START", None)
            with (folder / "evaluation.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "mjlab_microduck.program_evaluation", str(request_path.resolve())],
                    env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=cfg["timeout_seconds"])
            report = json.loads((folder / "report.json").read_text())
            updated = mdp.sc0090_program_apply_evaluation(env, report, request)
            if updated["last_result"]["passed"]:
                shutil.copyfile(frozen, Path(path).parent / "last_qualified.pt")
            # Apply changed weights/commands now; existing episodes naturally
            # adopt new spawn conditions on their next reset.
            mdp.sc0090_program_curriculum(env, None, mdp.sc0090_program_spec(env))
            print(f"[Training program evaluation] {updated['last_result']} next_phase="
                  f"{mdp.sc0090_program_phase(env)['name']}", flush=True)
            if self.logger.writer is not None:
                self.logger.writer.add_scalar("Program/phase", updated["phase_index"], self.current_learning_iteration)
                self.logger.writer.add_scalar("Program/passed", int(updated["last_result"]["passed"]), self.current_learning_iteration)
        except Exception as exc:
            error = dict(step=step, error=repr(exc))
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error.json").write_text(json.dumps(error, indent=2)+"\n")
            mdp.sc0090_program_state(env)["last_error"] = error
            print(f"[Training program evaluation failed] {exc!r}", flush=True)
        MjlabOnPolicyRunner.save(self, path, self._infos(infos))
