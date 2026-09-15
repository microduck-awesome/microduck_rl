"""Full-state continuation and isolated assessments for both SC0090 V4 tasks."""
from copy import deepcopy
import hashlib
import io
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shutil
import signal
import tempfile

import cloudpickle
import torch
import yaml

from . import SC0090FineTuneRunner, mdp
from mjlab_microduck.actuator.sc0090 import SC0090_MODEL_PATH, SC0090_DYNAMICS_REVISION


def atomic_write(path, write):
    """Publish a complete file with a same-filesystem rename; retain the old file on failure."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_run_lock(directory):
    """One writer per run directory; never wait on another training process."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".program_writer.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


class EvaluationCleanupError(RuntimeError):
    """Do not resume GPU training if assessment process cleanup could not finish."""


def run_evaluation_process(command, child_env, log, timeout):
    # A new session lets timeout/interrupt cleanup also reach descendants. Avoid
    # subprocess.run's unbounded wait after SIGKILL on a stuck driver process.
    process = subprocess.Popen(command, env=child_env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        status = process.wait(timeout=timeout)
        if status:
            raise subprocess.CalledProcessError(status, command)
    finally:
        # Descendants may remain even if their leader already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=10.)
            except subprocess.TimeoutExpired as exc:
                raise EvaluationCleanupError(
                    f"Evaluator PID {process.pid} did not exit; checkpoint saved, stopping training") from exc


def validate_runner_config(train_cfg, device, training):
    # Must precede the parent constructor: it initializes NCCL and can wait for
    # absent ranks before the old post-constructor single-GPU check ever runs.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.device(device).type != "cuda":
        raise ValueError("V4's isolated evaluation requires a single CUDA training GPU")
    if os.environ.get("MICRODUCK_WARM_START", "0") not in ("", "0"):
        raise ValueError("V4 preserves global counters; unset MICRODUCK_WARM_START. Omit checkpoint for fresh training.")
    cfg = train_cfg["training_program"]
    for name, value, minimum in (
        ("interval_iterations", cfg["interval_iterations"], 1),
        ("samples_per_group", cfg["samples_per_group"], 128),
        ("save_interval", train_cfg["save_interval"], 1),
        ("num_steps_per_env", train_cfg["num_steps_per_env"], 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}: expected integer >= {minimum}")
    if (len(cfg["seeds"]) < 2 or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in cfg["seeds"])
            or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or not math.isfinite(cfg["timeout_seconds"]) or cfg["timeout_seconds"] <= 0):
        raise ValueError("Invalid program evaluation seeds/timeout")
    if cfg["checkpoint"] and train_cfg.get("resume"):
        raise ValueError("Use either training_program.checkpoint or agent.resume, not both")
    if training and (type(train_cfg["max_iterations"]) is not int or train_cfg["max_iterations"] < 1):
        raise ValueError("V4 requires --agent.max-iterations: a positive budget of additional PPO updates")


def validate_finite_checkpoint(value):
    """Reject nonfinite source state before any model or optimizer is overwritten."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise ValueError("Checkpoint contains nonfinite tensors")
    elif isinstance(value, dict):
        for child in value.values():
            validate_finite_checkpoint(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            validate_finite_checkpoint(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Checkpoint contains nonfinite scalars")


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
        validate_runner_config(train_cfg, device, training=log_dir is not None)
        self._writer_lock = acquire_run_lock(log_dir) if log_dir is not None else None
        self._evaluation_train_cfg = deepcopy(train_cfg)
        try:
            super().__init__(env, train_cfg, log_dir, device, **kwargs)
        except BaseException:
            if self._writer_lock is not None:
                self._writer_lock.close()
            raise
        cfg = self.cfg["training_program"]
        self._abi = {"actor_obs": 61, "actions": 14,
                     "dynamics_revision": SC0090_DYNAMICS_REVISION,
                     "motor_sha256": hashlib.sha256(SC0090_MODEL_PATH.read_bytes()).hexdigest()}
        if cfg["checkpoint"]:
            self.load(cfg["checkpoint"])
        if log_dir is not None:
            steps = cfg['interval_iterations'] * self.cfg['num_steps_per_env']
            print(f"[Training program schedule] budget={self.cfg['max_iterations']} additional PPO updates; "
                  f"assessment interval={cfg['interval_iterations']} updates = {steps} steps per environment, "
                  f"aligned to saves every {self.cfg['save_interval']} updates; phase changes require assessment passes",
                  flush=True)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Validate and load the SAME bytes even if another writer replaces path.
        payload = Path(path).read_bytes()
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        validate_finite_checkpoint(checkpoint)
        infos = checkpoint.get("infos") or {}
        task = mdp.sc0090_program_spec(self.env.unwrapped)["task"]
        if identify_checkpoint_task(path, infos, self.cfg["training_program"]["legacy_task"]) != task:
            raise ValueError("Cannot transfer walking/recovery normalizers across task families")
        metadata = infos.get("sc0090_program_runner", {})
        if metadata and metadata["abi"] != self._abi:
            raise ValueError("Checkpoint motor/dynamics/observation/action contract changed; import the original legacy expert and reassess")
        restore_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        restore_iteration = load_cfg is None or load_cfg.get("iteration", False)
        if restore_optimizer:
            groups = checkpoint["optimizer_state_dict"]["param_groups"]
            learning_rate = metadata.get("learning_rate", groups[0]["lr"])
            if (not math.isfinite(learning_rate) or learning_rate <= 0
                    or any(group["lr"] != learning_rate for group in groups)):
                raise ValueError("Checkpoint adaptive/optimizer learning rates disagree")
        if restore_iteration:
            # RSL's checkpoint iteration names the completed update, while learn()
            # begins at current_learning_iteration. Resume at the NEXT update label.
            next_iteration = metadata.get("next_iteration", checkpoint["iter"]+1)
            if (type(checkpoint["iter"]) is not int or checkpoint["iter"] < 0
                    or type(next_iteration) is not int or next_iteration != checkpoint["iter"]+1):
                raise ValueError("Invalid next-iteration checkpoint state")
        counter = infos.get("env_state", {}).get("common_step_counter")
        if type(counter) is not int or counter < 0:
            raise ValueError("Checkpoint must contain a valid global environment step counter")
        # Validate curriculum metadata before mutating any model/optimizer state.
        from types import SimpleNamespace
        probe = SimpleNamespace(cfg=self.env.unwrapped.cfg, common_step_counter=counter)
        mdp.sc0090_program_restore(probe, infos.get("sc0090_program"), infos.get("sc0090_recovery"))
        if infos.get("sc0090_program") and metadata.get("evaluation_protocol") != 3:
            # Protocol 2 advanced delay buffers on manual resets and measured
            # retry episodes. Its smoothing baseline is not comparable to v3.
            if probe._sc0090_program["reference_metrics"] and (restore_optimizer or restore_iteration):
                raise ValueError("Old V4 assessment metrics use protocol 2; retain the old recipe or import a V2/V3 checkpoint")
            probe._sc0090_program.update(passes=0, last_attempt_step=-1, last_result={}, reference_metrics={})
        restored = super().load(io.BytesIO(payload), load_cfg=load_cfg, strict=strict,
                                map_location=map_location or self.device)
        if restore_optimizer:
            self.alg.learning_rate = learning_rate
        if restore_iteration:
            self.current_learning_iteration = next_iteration
        self.env.unwrapped._sc0090_program = deepcopy(probe._sc0090_program)
        with torch.inference_mode():
            # mjlab does not serialize physics episodes. Reapply the restored
            # live settings before resetting them; keep global training counters.
            self.env.unwrapped.reset()
        print(f"[Training program restored] task={task} next_iteration={self.current_learning_iteration} "
              f"phase={mdp.sc0090_program_phase(self.env.unwrapped)['name']} lr={self.alg.learning_rate}", flush=True)
        return restored

    def _save_checkpoint(self, path, infos, program_state=None):
        saved = self.alg.save()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = {**self._infos(infos),
                          "env_state": {"common_step_counter": self.env.unwrapped.common_step_counter}}
        if program_state is not None:
            saved["infos"]["sc0090_program"] = deepcopy(program_state)
        atomic_write(path, lambda stream: torch.save(saved, stream))

    def _export_checkpoint(self, path):
        # Keep the official normalizer-aware exporter and metadata contract.
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
        import wandb
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        temporary = policy_dir / f".{filename}.tmp.onnx"
        try:
            self.export_policy_to_onnx(str(policy_dir), temporary.name)
            run_name = wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
            attach_metadata_to_onnx(str(temporary), get_base_metadata(self.env.unwrapped, run_name))
            temporary.replace(onnx_path)
            if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as exc:
            print(f"[WARN] ONNX export failed (training continues): {exc}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)

    def _infos(self, infos):
        recovery = getattr(self.env.unwrapped, "_sc0090_recovery", None)
        if recovery is not None:
            infos = {**(infos or {}), "sc0090_recovery": recovery.state_dict()}
        return {**(infos or {}), "sc0090_program": deepcopy(mdp.sc0090_program_state(self.env.unwrapped)),
                "sc0090_program_runner": {"abi": self._abi,
                    "evaluation_protocol": 3,
                    "learning_rate": self.alg.learning_rate,
                    "next_iteration": self.current_learning_iteration+1}}

    def save(self, path, infos=None):
        self._save_checkpoint(path, infos)
        self._export_checkpoint(path)
        env = self.env.unwrapped
        cfg = self.cfg["training_program"]
        state = mdp.sc0090_program_state(env)
        step = env.common_step_counter
        interval = cfg["interval_iterations"] * self.cfg["num_steps_per_env"]
        if state["last_attempt_step"] >= 0 and step-state["last_attempt_step"] < interval:
            if self.cfg["upload_model"]:
                self.logger.save_model(path, self.current_learning_iteration)
            return
        state["last_attempt_step"] = step
        root = Path(path).parent / "program_evaluations"
        root.mkdir(parents=True, exist_ok=True)
        # Crash/retry at the same counter must not reuse a partial/stale report.
        folder = Path(tempfile.mkdtemp(dir=root, prefix=f"step_{step:09d}_"))
        try:
            frozen = folder / "model.pt"
            shutil.copyfile(path, frozen)
            config = folder / "config.pkl"
            with config.open("wb") as stream:
                cloudpickle.dump({"env": env.cfg, "agent": self._evaluation_train_cfg}, stream)
            request = dict(schema=3, task=state["task"], program_hash=state["program_hash"],
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
                run_evaluation_process(
                    [sys.executable, "-m", "mjlab_microduck.program_evaluation", str(request_path.resolve())],
                    child_env, log, cfg["timeout_seconds"])
            report = json.loads((folder / "report.json").read_text())
            updated = mdp.sc0090_program_apply_evaluation(env, report, request, commit=False)
            if updated["last_result"]["passed"]:
                # Save the validated metadata too, so rollback restores the pass
                # streak/cadence of the qualified model, not the pre-eval state.
                self._save_checkpoint(Path(path).parent / "last_qualified.pt", infos, program_state=updated)
            # The next reset applies the phase inside env.step(), before its
            # returned observations. Applying here invalidates the learn loop's
            # already-cached observations for the very next action.
        except BaseException as exc:
            state["passes"] = 0
            error = dict(step=step, error=repr(exc))
            state["last_error"] = error
            state["last_result"] = dict(passed=False, advanced=False,
                                       assessed_phase=mdp.sc0090_program_phase(env)["name"], **error)
            if not isinstance(exc, Exception) or isinstance(exc, EvaluationCleanupError):
                raise
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error.json").write_text(json.dumps(error, indent=2)+"\n")
            print(f"[Training program evaluation failed] {exc!r}", flush=True)
        else:
            env._sc0090_program = updated
            print(f"[Training program evaluation] {updated['last_result']} next_phase="
                  f"{mdp.sc0090_program_phase(env)['name']}", flush=True)
            if self.logger.writer is not None:
                self.logger.writer.add_scalar("Program/phase", updated["phase_index"], self.current_learning_iteration)
                self.logger.writer.add_scalar("Program/passed", int(updated["last_result"]["passed"]), self.current_learning_iteration)
        finally:
            self._save_checkpoint(path, infos)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)
