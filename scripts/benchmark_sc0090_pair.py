"""Benchmark both SC0090 tasks together, optionally in a private MPS session.

Only the diagnostic children are supervised. Existing training processes and
system GPU settings are not changed. Inherit the same CUDA compatibility
libraries used by scripts/train_sc0090_local.sh on older driver installations.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--variant", choices=("previous", "optimized"), default="optimized")
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--walk-cpus", help="optional taskset CPU list")
    parser.add_argument("--recovery-cpus", help="optional taskset CPU list")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    checkpoints = {"walk": args.walk_checkpoint.resolve(), "recovery": args.recovery_checkpoint.resolve()}
    if not all(p.is_file() for p in checkpoints.values()):
        parser.error("both checkpoints must exist")
    if args.envs < 1 or args.warmup < 1 or args.iterations < 2 or args.timeout <= 0:
        parser.error("need positive envs, warmup and timeout, and at least 2 measured iterations")
    if args.mps and shutil.which("nvidia-cuda-mps-control") is None:
        parser.error("nvidia-cuda-mps-control is not installed")
    if (args.walk_cpus or args.recovery_cpus) and shutil.which("taskset") is None:
        parser.error("CPU affinity requires taskset")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    if not args.mps:
        # A user's existing MPS daemon must not turn the baseline into an MPS
        # run merely because its pipe directory was inherited.
        env["CUDA_MPS_PIPE_DIRECTORY"] = ""
    for key, value in (("OMP_NUM_THREADS", "8"), ("MKL_NUM_THREADS", "8"),
                       ("OPENBLAS_NUM_THREADS", "1"), ("MUJOCO_GL", "egl"),
                       ("PYTHONUNBUFFERED", "1")):
        env.setdefault(key, value)
    pipe = None
    processes, streams = {}, []
    record = {"mps": args.mps, "started_unix": time.time(), "processes": {}}
    try:
        if args.mps:
            pipe = tempfile.mkdtemp(prefix="microduck-mps-")
            server_logs = args.output.resolve() / "mps"
            server_logs.mkdir()
            env.update(CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=str(server_logs))
            subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True, timeout=15)
        benchmark = Path(__file__).with_name("benchmark_sc0090.py")
        for kind, checkpoint in checkpoints.items():
            folder = args.output / kind
            stream = (args.output / f"{kind}.log").open("w")
            streams.append(stream)
            command = [sys.executable, str(benchmark), kind, "--checkpoint", str(checkpoint),
                       "--output", str(folder), "--variant", args.variant,
                       "--envs", str(args.envs), "--warmup", str(args.warmup),
                       "--iterations", str(args.iterations)]
            if args.export:
                command.append("--export")
            cpus = args.walk_cpus if kind == "walk" else args.recovery_cpus
            if cpus:
                command = ["taskset", "-c", cpus, *command]
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes[kind] = process
            record["processes"][kind] = {"pid": process.pid, "command": command}
        deadline = time.monotonic() + args.timeout
        while any(p.poll() is None for p in processes.values()):
            if any(p.returncode not in (None, 0) for p in processes.values()):
                raise RuntimeError("a benchmark child failed; inspect the task logs")
            if time.monotonic() >= deadline:
                raise TimeoutError("paired benchmark exceeded its time limit")
            time.sleep(.5)
        if any(p.returncode for p in processes.values()):
            raise RuntimeError("a benchmark child failed; inspect the task logs")
        reports = {k: json.loads((args.output / k / "benchmark.json").read_text()) for k in processes}
        measured = {k: r["samples"][r["warmup"]:] for k, r in reports.items()}
        lo = max(v[0]["end_unix"] - v[0]["wall_s"] for v in measured.values())
        hi = min(v[-1]["end_unix"] for v in measured.values())
        record["overlap_unix"] = [lo, hi]
        record["tasks"] = {}
        for kind, samples in measured.items():
            overlap = [s for s in samples if s["end_unix"] - s["wall_s"] >= lo and s["end_unix"] <= hi]
            if len(overlap) < 2:
                raise RuntimeError("insufficient overlap after warmup; increase --iterations")
            median = statistics.median(s["wall_s"] for s in overlap)
            # Recover the runner's actual step count instead of assuming its
            # configured rollout length or adding isolated-task throughput.
            steps = round(reports[kind]["steps_per_second"] * reports[kind]["median_wall_s"])
            record["tasks"][kind] = {"samples": len(overlap), "median_wall_s": median,
                                     "steps_per_second": steps / median}
        record["combined_steps_per_second"] = sum(r["steps_per_second"] for r in record["tasks"].values())
    finally:
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
        for kind, p in processes.items():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            record["processes"][kind]["returncode"] = p.returncode
        for stream in streams:
            stream.close()
        if pipe is not None:
            shutdown = subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=env,
                                      capture_output=True, text=True, timeout=15)
            record["mps_shutdown_code"] = shutdown.returncode
            shutil.rmtree(pipe)
        record["finished_unix"] = time.time()
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
