"""Local checkpoint discovery and bounded, isolated ONNX export for the web UI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

EXPORT_TIMEOUT_S = 300
FAMILIES = {
    'sc0090_walk': ('walking', 'Mjlab-Velocity-Flat-MicroDuck'),
    'sc0090_walk_v2': ('walking', 'Mjlab-Velocity-Flat-MicroDuck-SC0090-V2'),
    'sc0090_walk_v4': ('walking', 'Mjlab-Velocity-Flat-MicroDuck-SC0090-V4'),
    'sc0090_recovery': ('recovery', 'Mjlab-StandUp-Flat-MicroDuck'),
    'sc0090_recovery_v2': ('recovery', 'Mjlab-StandUp-Flat-MicroDuck-SC0090-V2'),
    'sc0090_recovery_v3': ('recovery', 'Mjlab-StandUp-Flat-MicroDuck-SC0090-V3'),
    'sc0090_recovery_v4': ('recovery', 'Mjlab-StandUp-Flat-MicroDuck-SC0090-V4'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class CheckpointLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.repo = self.root.parent
        self.cache = self.root / 'models/cache'
        self.cache.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.entries = {}
        self.current = {}
        self.job = None
        self.status = {'state': 'idle', 'message': ''}
        self.manifest = json.loads((self.root / 'models/manifest.json').read_text())
        self.refresh()

    def refresh(self):
        entries = {}
        bundled = self.manifest['policies']
        for family, (slot, task) in FAMILIES.items():
            for path in (self.repo / 'logs/rsl_rl' / family).glob('*/model_*.pt'):
                if 'smoke' in path.parent.name or path.parent.name.startswith('source'):
                    continue
                match = re.fullmatch(r'model_(\d+)\.pt', path.name)
                if not match:
                    continue
                try:
                    size, modified = fingerprint(path)
                except FileNotFoundError:
                    continue
                if size == 0:
                    continue
                relative = str(path.relative_to(self.repo))
                key = hashlib.sha256(f'{relative}:{size}:{modified}'.encode()).hexdigest()[:24]
                entry = {'id': key, 'slot': slot, 'task': task, 'checkpoint': relative,
                         'iteration': int(match[1]), 'run': path.parent.name, 'family': family,
                         'fingerprint': (size, modified), 'path': str(path)}
                if bundled[slot]['checkpoint'] == relative:
                    entry['bundled'] = bundled[slot]
                    self.current.setdefault(slot, key)
                entries[key] = entry
        for slot, policy in bundled.items():
            if any(e.get('bundled') == policy for e in entries.values()):
                continue
            key = f'bundled-{slot}-{policy["sha256"][:12]}'
            checkpoint = Path(policy['checkpoint'])
            entries[key] = {'id': key, 'slot': slot, 'task': policy['task'],
                            'checkpoint': policy['checkpoint'], 'iteration': int(checkpoint.stem.split('_')[-1]),
                            'run': checkpoint.parent.name, 'family': checkpoint.parent.parent.name,
                            'bundled': policy}
            self.current.setdefault(slot, key)
        with self.lock:
            self.entries = entries
        return self.catalog()

    def catalog(self):
        with self.lock:
            entries = list(self.entries.values())
        result = {'walking': [], 'recovery': []}
        for entry in sorted(entries, key=lambda e: (e['run'], e['iteration']), reverse=True):
            public = {key: entry[key] for key in ('id','iteration','run','family','checkpoint')}
            public['ready'] = 'bundled' in entry or (self.cache / f'{entry["id"]}.json').is_file()
            result[entry['slot']].append(public)
        return result

    def _check_result(self, entry, result, folder):
        policy = result['policies'][entry['slot']]
        if result['motor_model_sha256'] != self.manifest['motor_model_sha256']:
            raise ValueError('SC0090 motor version mismatch')
        if policy['checkpoint'] != entry['checkpoint'] or policy['task'] != entry['task']:
            raise ValueError('Export result belongs to another checkpoint/task')
        if 'path' in entry and policy['checkpoint_sha256'] != sha256(entry['path']):
            raise ValueError('Checkpoint changed since export; refresh the list')
        path = (folder / policy['file']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != policy['sha256']:
            raise ValueError('Exported model checksum mismatch')
        return entry, path

    def begin(self, slot, key):
        if self.job is not None:
            raise ValueError('另一个 checkpoint 正在导出，请等待完成')
        with self.lock:
            entry = dict(self.entries.get(key, {}))
        if not entry or entry['slot'] != slot:
            raise ValueError('checkpoint 不存在或任务类型不匹配，请刷新列表')
        if 'path' in entry and fingerprint(Path(entry['path'])) != entry['fingerprint']:
            raise ValueError('checkpoint 文件发生变化，请刷新列表')
        self.status = {'state':'loading', 'slot':slot, 'id':key,
                       'message':f'正在加载 {entry["family"]} / model_{entry["iteration"]}'}
        if 'bundled' in entry:
            return self._check_result(entry, self.manifest, self.root / 'models')
        result = self.cache / f'{key}.json'
        if result.is_file():
            return self._check_result(entry, json.loads(result.read_text()), self.cache)
        log = self.cache / f'{key}.log'
        command = [sys.executable, str(self.root / 'scripts/export_models.py'),
                   '--device', 'cpu', '--output-dir', str(self.cache), '--result-file', str(result),
                   '--walk-checkpoint' if slot == 'walking' else '--recovery-checkpoint', entry['path'],
                   '--walk-task' if slot == 'walking' else '--recovery-task', entry['task']]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_MPS_PIPE_DIRECTORY='',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=self.repo, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.job = {'process':process, 'started':time.monotonic(), 'entry':entry,
                    'result':result, 'log':log}
        self.status['message'] = '首次加载：正在独立 CPU 进程中导出 ONNX，当前模型继续运行'
        return None

    def poll(self):
        if self.job is None:
            return None
        job = self.job
        process = job['process']
        if process.poll() is None:
            if time.monotonic() - job['started'] > EXPORT_TIMEOUT_S:
                self.close()
                raise RuntimeError(f'导出超过 {EXPORT_TIMEOUT_S} 秒，已取消，保留当前模型')
            return None
        self.job = None
        if process.returncode != 0:
            raise RuntimeError(f'导出失败，保留当前模型。日志：simulator/models/cache/{job["log"].name}')
        return self._check_result(job['entry'], json.loads(job['result'].read_text()), self.cache)

    def loaded(self, entry):
        self.current[entry['slot']] = entry['id']
        self.status = {'state':'ready', 'message':f'已加载 {entry["family"]} / model_{entry["iteration"]}，并重置为站立'}

    def failed(self, error):
        self.status = {'state':'error', 'message':str(error)}

    def close(self):
        if self.job is None:
            return
        process = self.job['process']
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        self.job = None
