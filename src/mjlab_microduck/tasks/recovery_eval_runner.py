"""Run frozen-policy assessments in a process isolated from PPO state/RNG."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import cloudpickle
import shutil
import subprocess
import sys
import time

from mjlab.rl import MjlabOnPolicyRunner

from . import SC0090FineTuneRunner, mdp


class SC0090RecoveryEvalRunner(SC0090FineTuneRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        # RSL consumes class_name keys while building PPO/models. Preserve the
        # complete constructor config before that in-place mutation.
        self._evaluation_train_cfg = deepcopy(train_cfg)
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if self.is_distributed:
            raise ValueError("Recovery evaluation curriculum currently requires one training GPU")
        cfg = self.cfg["recovery_evaluation"]
        if (cfg["interval_iterations"] < 1 or cfg["samples_per_group"] < 1
                or len(set(cfg["seeds"])) != len(cfg["seeds"]) or len(cfg["seeds"]) < 2
                or cfg["timeout_seconds"] <= 0):
            raise ValueError("Invalid recovery evaluation configuration")
        self.env.unwrapped._sc0090_eval_history = {}

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        if load_cfg is None and os.environ.get("MICRODUCK_WARM_START", "0") in ("", "0"):
            self.env.unwrapped._sc0090_eval_history = deepcopy(
                (infos or {}).get("sc0090_evaluation", {}))
        return infos

    def _infos(self, infos):
        env = self.env.unwrapped
        return {**(infos or {}),
                "sc0090_recovery": mdp._sc0090_recovery_state(env).state_dict(),
                "sc0090_evaluation": deepcopy(env._sc0090_eval_history)}

    def save(self, path, infos=None):
        # The official save/export path owns policy serialization/normalization.
        super().save(path, self._infos(infos))
        env = self.env.unwrapped
        cfg = self.cfg["recovery_evaluation"]
        it = self.current_learning_iteration
        last = env._sc0090_eval_history.get("last_attempt_iteration", -1)
        if last >= 0 and it - last < cfg["interval_iterations"]:
            return
        env._sc0090_eval_history["last_attempt_iteration"] = it
        started = time.monotonic()
        folder = Path(path).parent / "evaluations" / f"iteration_{it:06d}"
        try:
            folder.mkdir(parents=True, exist_ok=False)
            checkpoint = folder / "model.pt"
            shutil.copyfile(path, checkpoint)
            config = folder / "config.pkl"
            # These are this runner's trusted local configuration objects.
            # Passing them preserves CLI overrides and the training DR/commands.
            with config.open("wb") as stream:
                cloudpickle.dump({"env": env.cfg, "agent": self._evaluation_train_cfg}, stream)
            request = {
                "schema": 1, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "config": str(config.resolve()),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "iteration": it, "stage": mdp._sc0090_recovery_state(env).stage,
                "seeds": list(cfg["seeds"]), "samples_per_group": cfg["samples_per_group"],
                "duration_s": 8.0, "deterministic": True, "device": self.device,
            }
            request_path = folder / "request.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            child_env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="", OMP_NUM_THREADS="2",
                             MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
            child_env.pop("MICRODUCK_WARM_START", None)
            # No evaluation kernels, RNG sampling, resets or normalizer updates
            # occur in the training process. The child bypasses shared MPS, so
            # its timeout cleanup does not terminate a shared MPS client.
            with (folder / "evaluation.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "mjlab_microduck.recovery_evaluation",
                                str(request_path.resolve())], env=child_env,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=cfg["timeout_seconds"])
            report = json.loads((folder / "report.json").read_text())
            history = mdp.sc0090_apply_recovery_evaluation(env, report, request)
            history["report"] = str((folder / "report.json").resolve())
            print(f"[Recovery evaluation] iteration={it} stage={request['stage']}->{history['stage_after']} "
                  f"rates={history['rates']}", flush=True)
            if self.logger.writer is not None:
                for name, value in history["rates"].items():
                    self.logger.writer.add_scalar(f"Evaluation/success_{name}", value, it)
                self.logger.writer.add_scalar("Evaluation/stage", history["stage_after"], it)
                self.logger.writer.add_scalar("Evaluation/wall_seconds", time.monotonic() - started, it)
        except Exception as exc:
            # An incomplete assessment never advances the curriculum. Retain
            # training progress and make the failure visible for investigation.
            error = {"iteration": it, "error": repr(exc)}
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error.json").write_text(json.dumps(error, indent=2) + "\n")
            env._sc0090_eval_history["last_error"] = error
            print(f"[Recovery evaluation failed; stage unchanged] {exc!r}", flush=True)
        # Persist the new assessment/promotion immediately without re-exporting
        # the identical policy or repeating the official learning loop.
        MjlabOnPolicyRunner.save(self, path, self._infos(infos))
