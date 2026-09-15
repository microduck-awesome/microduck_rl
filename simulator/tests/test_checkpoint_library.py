import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
import checkpoint_library as library


@pytest.fixture
def bank(tmp_path):
    root = tmp_path / 'repo/simulator'
    (root / 'models').mkdir(parents=True)
    policies = {}
    for slot, family in [('walking','sc0090_walk_v2'),('recovery','sc0090_recovery_v3')]:
        source = f'logs/rsl_rl/{family}/2026-09-15_training/model_5999.pt'
        checkpoint = root.parent / source
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint')
        model = root / 'models' / f'{slot}.onnx'
        model.write_bytes(b'normalized ONNX fixture')
        policies[slot] = dict(file=model.name, checkpoint=source, task=library.FAMILIES[family][1],
                              checkpoint_sha256=library.sha256(checkpoint), sha256=library.sha256(model))
    (root / 'models/manifest.json').write_text(json.dumps(dict(schema=1,policies=policies,motor_model_sha256='motor')))
    return library.CheckpointLibrary(root)


def test_discovery_separates_task_and_run(bank):
    catalog = bank.catalog()
    assert len(catalog['walking']) == len(catalog['recovery']) == 1
    assert catalog['walking'][0]['id'] != catalog['recovery'][0]['id']
    with pytest.raises(ValueError, match='任务类型'):
        bank.begin('walking', catalog['recovery'][0]['id'])
    entry,path = bank.begin('walking',catalog['walking'][0]['id'])
    assert path.name == 'walking.onnx'


def test_checkpoint_mutation_and_corrupt_cache_do_not_load(bank):
    key = bank.current['walking']
    source = Path(bank.entries[key]['path'])
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='发生变化'):
        bank.begin('walking',key)
    bank.refresh()
    fresh = bank.catalog()['walking'][0]['id']
    with pytest.raises(ValueError, match='changed since export'):
        bank.begin('walking',fresh)


def test_shipped_models_work_without_training_logs(bank):
    for entry in bank.entries.values():
        Path(entry['path']).unlink()
    fresh = library.CheckpointLibrary(bank.root)
    entry,path = fresh.begin('recovery',fresh.current['recovery'])
    assert 'path' not in entry and path.is_file()


def test_export_process_is_cpu_only_bounded_and_cancellable(bank, monkeypatch):
    source = bank.repo / 'logs/rsl_rl/sc0090_walk_v2/2026-09-15_training/model_100.pt'
    source.write_bytes(b'older checkpoint')
    bank.refresh()
    key = next(e['id'] for e in bank.entries.values() if e['iteration']==100)
    captured = {}
    class Process:
        pid = 123456789
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout):
            self.returncode = -15
            captured['wait_timeout'] = timeout
            return self.returncode
    def launch(command, **kwargs):
        captured.update(command=command,**kwargs)
        return Process()
    monkeypatch.setattr(library.subprocess,'Popen',launch)
    signals=[]
    monkeypatch.setattr(library.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    assert bank.begin('walking',key) is None
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert captured['env']['CUDA_MPS_PIPE_DIRECTORY'] == ''
    assert captured['start_new_session'] is True
    assert captured['command'][captured['command'].index('--device')+1] == 'cpu'
    with pytest.raises(ValueError, match='等待完成'):
        bank.begin('walking',key)
    bank.job['started'] -= library.EXPORT_TIMEOUT_S + 1
    with pytest.raises(RuntimeError, match='超过'):
        bank.poll()
    assert bank.job is None and signals and captured['wait_timeout'] <= 2
