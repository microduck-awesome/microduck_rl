#!/usr/bin/env python3
"""Export versioned demo policies through microduck_rl's normalized exporter."""
import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walk-checkpoint', type=Path)
    parser.add_argument('--recovery-checkpoint', type=Path)
    parser.add_argument('--walk-task', default='Mjlab-Velocity-Flat-MicroDuck-SC0090-V2')
    parser.add_argument('--recovery-task', default='Mjlab-StandUp-Flat-MicroDuck-SC0090-V3')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--result-file', type=Path, help='Optional machine-readable export result')
    args = parser.parse_args()
    if not (args.walk_checkpoint or args.recovery_checkpoint):
        parser.error('Provide --walk-checkpoint and/or --recovery-checkpoint')
    for name in ('walk_checkpoint', 'recovery_checkpoint'):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve(strict=True))
    os.chdir(REPO)
    import mjlab.tasks  # Populate the task registry before invoking the shared exporter.
    from mjlab_microduck.export import ExportConfig, run_export
    from mjlab_microduck.actuator.sc0090 import SC0090_MODEL_PATH
    folder = args.output_dir.resolve()
    folder.mkdir(exist_ok=True, parents=True)
    with (folder / '.export.lock').open('a') as writer_lock:
        try:
            fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is exporting demo models') from exc
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema': 1, 'policies': {}}
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
        for name, checkpoint, task in [('walking',args.walk_checkpoint,args.walk_task),
                                        ('recovery',args.recovery_checkpoint,args.recovery_task)]:
            if checkpoint is None:
                continue
            checkpoint = checkpoint.resolve(strict=True)
            checkpoint_ref = str(checkpoint.relative_to(REPO)) if checkpoint.is_relative_to(REPO) else str(checkpoint)
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with tempfile.TemporaryDirectory(prefix='.export-', dir=folder) as tmp:
                output = Path(tmp) / 'policy.onnx'
                run_export(task, ExportConfig(checkpoint_file=checkpoint_ref, num_envs=1,
                                              device=args.device, onnx_file=str(output)))
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
                    raise RuntimeError('Checkpoint changed during export; manifest was not updated')
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                filename = f'{name}_{checkpoint.stem}_{digest[:12]}.onnx'
                destination = folder / filename
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                        raise RuntimeError('Existing versioned policy has different contents')
                else:
                    os.replace(output, destination)
            checkpoint_ref = str(checkpoint.relative_to(REPO)) if checkpoint.is_relative_to(REPO) else checkpoint.name
            manifest['policies'][name] = {'file': filename, 'task': task,
                'checkpoint': checkpoint_ref, 'checkpoint_sha256': checkpoint_hash,
                'sha256': digest, 'export_source_commit': commit}
        manifest.update(schema=1, exported_at=datetime.now(timezone.utc).isoformat(),
                        motor_model_sha256=hashlib.sha256(SC0090_MODEL_PATH.read_bytes()).hexdigest(),
                        policy_hz=50, motor_max_speed_rpm=80, observation_dim=61, action_dim=14)
        # A running demo retains its loaded sessions. A new demo sees either the
        # complete previous manifest or the complete new pair, never half a pair.
        with tempfile.NamedTemporaryFile(mode='w', dir=folder, delete=False, suffix='.tmp') as stream:
            json.dump(manifest, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, manifest_path)
        if args.result_file is not None:
            result_file = args.result_file.resolve()
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=result_file.parent, delete=False, suffix='.tmp') as stream:
                json.dump(manifest, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
                result_temp = Path(stream.name)
            os.replace(result_temp, result_file)
        print(manifest_path)



if __name__ == '__main__':
    main()
