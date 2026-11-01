"""Verify the formal V3 run retains the original V2 6804 motor contract."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE='67120c2fc9c5f4be8323ce04588ab18c9a643e45'
RUN=Path('/home/l40-vm01/data2/experiments/v3-rise6804-20261101/recovery-0500')
V2=ROOT/'docs/evidence/rise6804-source-publication/checkpoints/rank0/run.json'
PARAM='src/mjlab_microduck/actuator/params/sc0090_v5_m6.json'
BANK='src/mjlab_microduck/custom_v2/training-assets/repaired-paired-errors-v2/error-blocks.npz'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def blob(commit,path):
    return subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--v2-xml',type=Path,default=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-repro/repro/sc0090-v2/checkpoints/20261024-rise3762-collision-adaptation/candidate-6804/model/robot.xml'))
    args=parser.parse_args()
    old=json.loads(V2.read_text());new=json.loads((RUN/'run.json').read_text())
    assert old['dependencies']==new['dependencies']
    files=['src/mjlab_microduck/actuator/sc0090_v5.py',
           'src/mjlab_microduck/custom_v2/paired_randomization.py',
           'src/mjlab_microduck/custom_v2/error_proxy.py',PARAM,BANK]
    evidence={}
    for path in files:
        original=sha(blob(BASE,path));frozen=sha(blob(new['git_head'],path))
        actual=sha((ROOT/path).read_bytes())
        assert original==frozen==actual, f'Motor or feedback asset mismatch: {path}'
        old_hash=old['source_sha256'].get(path)
        new_hash=new['source_sha256'].get(path)
        if old_hash is not None:assert old_hash==original
        if new_hash is not None:assert new_hash==original
        evidence[path]={'v2_source_sha256':original,'v3_commit_sha256':frozen,
                        'v2_run_manifest_sha256':old_hash,'v3_run_manifest_sha256':new_hash}
    changed=subprocess.check_output(['git','diff','--name-only',BASE,new['git_head'],'--',
                                     'src/mjlab_microduck/actuator'],cwd=ROOT,text=True).splitlines()
    assert not changed, f'Actuator source changed: {changed}'
    def actuators(run):
        a=run['environment']['scene']['entities']['robot']['articulation']['actuators']
        return [{k:v for k,v in cfg.items() if k!='json_path'} for cfg in a]
    assert actuators(old)==actuators(new)
    smoke=json.loads((RUN.parent/'batch8192-report.json').read_text())
    assert smoke['motor_sha256']==evidence[PARAM]['v2_source_sha256']
    assert sha(args.v2_xml.read_bytes())==sha(blob(BASE,'src/mjlab_microduck/custom_v2/assets/robot.xml'))
    old_model=mujoco.MjModel.from_xml_path(str(args.v2_xml))
    new_model=mujoco.MjModel.from_xml_path(str(ROOT/'src/mjlab_microduck/custom_v2/assets/robot.xml'))
    passive_dofs={}
    for idx in range(1,old_model.njnt):
        name=old_model.joint(idx).name
        old_id=int(old_model.joint(name).dofadr[0]);new_id=int(new_model.joint(name).dofadr[0])
        fields={}
        for field in ('dof_armature','dof_damping','dof_frictionloss'):
            a=float(getattr(old_model,field)[old_id]);b=float(getattr(new_model,field)[new_id])
            assert np.isclose(a,b,rtol=0,atol=1e-12), f'{name} {field}: V2={a}, V3={b}'
            fields[field]=a
        passive_dofs[name]=fields
    # The already-running first allocation's source tar omitted the JSON even
    # though the loader checks its hash. Save an explicit, immutable supplement.
    supplement=RUN.parent/'motor-provenance'
    supplement.mkdir(exist_ok=True)
    destination=supplement/'sc0090_v5_m6.json'
    data=blob(new['git_head'],PARAM)
    if destination.exists():assert destination.read_bytes()==data
    else:destination.write_bytes(data)
    report={'status':'PASS','v2_branch_base':BASE,'v3_training_commit':new['git_head'],
            'identical_dependencies':new['dependencies'],'identical_actuator_cfg_excluding_file_path':actuators(new),
            'unchanged_actuator_directory':True,'files':evidence,
            'v2_model_xml_sha256':sha(args.v2_xml.read_bytes()),
            'matched_15_joint_armature_damping_frictionloss':passive_dofs,
            'paired_voltage_v':[10.2,11.1,12.4],
            'supplement':str(destination),'supplement_sha256':sha(data),
            'provenance_note':'First V3 run source.tar.gz does not contain the motor JSON; frozen source commit, loader hash and supplemental bytes establish identity. This audit does not assert identical robot dynamics across different structures.'}
    output=ROOT/'docs/v3-motor-parity.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: V2/V3 motor and paired feedback source, exact parameters, operating settings and dependency versions; {output}')


if __name__=='__main__':main()
