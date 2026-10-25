"""Launch the official trainer and record status at 30 minute intervals."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/official-walk/single-gpu'
EVIDENCE.mkdir(parents=True, exist_ok=True)
LOG = EVIDENCE / 'formal.log'
COMMAND = ['bash', 'scripts/v2_python.sh', '-m', 'mjlab.scripts.train',
           'Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline',
           '--gpu-ids', 'all', '--env.scene.num-envs', '4096',
           '--agent.logger', 'tensorboard', '--agent.upload-model', 'False',
           '--agent.run-name', 'official_v2_sc0090_v5_gpu0',
           '--agent.resume', 'True', '--agent.load-run', '2026-10-25_13-47-22_official_v2_sc0090_v5',
           '--agent.load-checkpoint', 'model_250.pt', '--agent.max-iterations', '49750']

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

if __name__ == '__main__':
    started = time.monotonic()
    with LOG.open('x') as stream:
        child = subprocess.Popen(COMMAND, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(status='STARTING', started_utc=now(), trainer_pid=child.pid,
                      supervisor_pid=os.getpid(), command=COMMAND, check_interval_s=1800,
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        (EVIDENCE/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
        while True:
            try:
                code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                code = None
            snapshot = dict(time_utc=now(), elapsed_s=round(time.monotonic()-started),
                            status='RUNNING' if code is None else ('COMPLETED' if code == 0 else 'FAILED'),
                            exit_code=code, trainer_pid=child.pid)
            snapshot['gpus'] = subprocess.check_output([
                'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu', '--format=csv'],text=True)
            snapshot['log_tail'] = subprocess.check_output(['tail','-80',str(LOG)],text=True)
            with (EVIDENCE/'monitor.jsonl').open('a') as out:
                out.write(json.dumps(snapshot)+'\n')
            (EVIDENCE/'status.json').write_text(json.dumps(snapshot,indent=2)+'\n')
            print(json.dumps({k:v for k,v in snapshot.items() if k not in ('log_tail','gpus')}),flush=True)
            if code is not None:
                raise SystemExit(code)
