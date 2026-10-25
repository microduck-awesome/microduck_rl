"""Audit all official files, task configuration and locked dependencies."""
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import re
import subprocess
import tomllib

import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from bam.mjlab import BamActuatorCfg
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
from mjlab_microduck.actuator.sc0090_v5 import SC0090V5Actuator
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID, POLICY_JOINTS

ROOT = Path(__file__).resolve().parents[1]
BASE = 'b02fdea4902c2b93b7d34a83a6f05da4ececb7d3'
OUT = ROOT / 'docs/evidence/official-walk/corrected'
OUT.mkdir(parents=True, exist_ok=True)

def encode(v):
    if is_dataclass(v):
        result = {f.name: encode(getattr(v, f.name)) for f in fields(v)}
        # Resolve implicit SceneEntityCfg defaults before comparing selectors.
        if hasattr(v, 'func') and hasattr(v, 'params') and callable(v.func):
            default = inspect.signature(v.func).parameters.get('asset_cfg')
            if default is not None and default.default is not inspect.Parameter.empty:
                result['params'].setdefault('asset_cfg', encode(default.default))
        return result
    if isinstance(v, dict):
        return {str(k): encode(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [encode(x) for x in v]
    if isinstance(v, Enum):
        return str(v)
    if callable(v):
        return f'{v.__module__}.{v.__qualname__}'
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return repr(v)

def differences(a, b, path=''):
    if type(a) is not type(b):
        return [(path,a,b)]
    if isinstance(a, dict):
        out=[]
        for k in sorted(a.keys() | b.keys()):
            p=f'{path}.{k}' if path else k
            if k not in a or k not in b:
                out.append((p,a.get(k),b.get(k)))
            else:
                out.extend(differences(a[k],b[k],p))
        return out
    if isinstance(a,list):
        if len(a) != len(b):
            return [(path,a,b)]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,f'{path}.{i}')]
    return [] if a == b else [(path,a,b)]

protected=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
source_changes=[]
for name in protected:
    original=subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT)
    actual=(ROOT/name).read_bytes()
    if actual != original:
        if name == 'src/mjlab_microduck/tasks/__init__.py':
            assert actual == original + b'\n# V2 structure/HOME and frozen SC0090 V5 BAM adapter.\nfrom mjlab_microduck.custom_v2.microduck_v2 import register as _register_v2\n_register_v2()\n'
        else:
            source_changes.append(name)
assert not source_changes, source_changes

stock, custom=load_env_cfg('Mjlab-Velocity-Flat-MicroDuck'),load_env_cfg(TASK_ID)
rawdiff=differences(encode(stock),encode(custom))
classified=[]
for path, before, after in rawdiff:
    reason=None
    if path.startswith('scene.entities.robot.'):
        reason='V2 structure, HOME, calibrated motor and voltage replacement'
    elif path in ('events.mouth_hold','events.mouth_reset'):
        reason='Independent V2 mouth channel outside the 14D policy'
    elif path.startswith('actions.joint_pos.actuator_names') or path == 'actions.joint_pos.preserve_order':
        reason='Explicit 14-joint V2 action order'
    elif re.search(r'\.asset_cfg\.(joint_names|joint_ids|preserve_order)(\.|$)',path):
        reason='Select the same 14 policy joints and exclude V2 mouth'
    elif path.startswith('events.reset_base.params.pose_range.z.'):
        reason='Translate spawn-height origin to V2 HOME, retaining interval width'
    assert reason is not None, (path,before,after)
    classified.append(dict(path=path,official=before,v2=after,reason=reason))

# No event is removed. No reward, curriculum stage, command or observation is skipped.
assert set(custom.events) == set(stock.events) | {'mouth_hold','mouth_reset'}
assert custom.commands == stock.commands
assert custom.sim == stock.sim
assert custom.decimation == stock.decimation
assert custom.episode_length_s == stock.episode_length_s
assert load_runner_cls(TASK_ID) is load_runner_cls('Mjlab-Velocity-Flat-MicroDuck')
rl_diff=differences(encode(load_rl_cfg('Mjlab-Velocity-Flat-MicroDuck')),encode(load_rl_cfg(TASK_ID)))
assert all(p in ('experiment_name','run_name') for p,_,_ in rl_diff),rl_diff

upstream=stock.scene.entities['robot'].articulation.actuators[0]
for motor in custom.scene.entities['robot'].articulation.actuators:
    for f in fields(BamActuatorCfg):
        if f.init and f.name not in {'motor_name','model','json_path','target_names_expr','kp_fw','vin','vin_range','vin_min'}:
            assert getattr(motor,f.name) == getattr(upstream,f.name), f.name
    assert motor.vin_range == (10.2,12.4)
    assert motor.vin_min == 10.2
assert issubclass(SC0090V5Actuator,FrictionDRBamActuator), 'Official friction event must recognize the SC0090 actuator'

lock=tomllib.loads((ROOT/'uv.lock').read_text())
locked={}
for p in lock['package']:
    locked.setdefault(p['name'],set()).add(p['version'])
installed={}
for name,versions in locked.items():
    if name == 'mjlab-microduck':
        continue
    try:
        current=metadata.version(name)
    except metadata.PackageNotFoundError:
        continue  # platform-conditional dependencies can be absent
    assert current in versions,(name,current,versions)
    installed[name]=current
bam_url=metadata.distribution('better-actuator-models').read_text('direct_url.json')
assert '62bd8ce12154340be97e06f7f41a0ca8f116d967' in bam_url,bam_url

report=dict(status='PASS',upstream_revision=BASE,official_files_checked=len(protected),
            modified_official_files=['src/mjlab_microduck/tasks/__init__.py (registration suffix only)'],
            environment_differences=classified,rl_differences=rl_diff,
            all_official_events_preserved=True,official_runner=True,
            rewards_and_curricula='all terms and parameters compared; only V2 policy selectors adapted',
            dependencies=installed,bam_source=json.loads(bam_url))
(OUT/'alignment-audit.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'official-env.json').write_text(json.dumps(encode(stock),indent=2)+'\n')
(OUT/'v2-env.json').write_text(json.dumps(encode(custom),indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('environment_differences','dependencies','bam_source')},indent=2))
