"""Compare original 6804 recovery-start configuration with the V3 run."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='67120c2fc9c5f4be8323ce04588ab18c9a643e45'
OLD=ROOT/'repro/history/recovery-start/rank0/run.json'
OLD_RANK1=ROOT/'repro/history/recovery-start/rank1/run.json'
NEW=Path('/home/l40-vm01/data2/experiments/v3-rise6804-20261101/recovery-0500/run.json')
HEIGHT_FIELDS={
    '/rewards/body_pose_tracking/params/nominal_height',
    '/rewards/height_stand/params/target_height',
    '/rewards/height_stand_l1/params/target_height',
    '/rewards/height_stand_sharp/params/target_height',
    '/rewards/remaining_fallen/params/release_z_above',
    '/rewards/standing_composite/params/target_height',
    '/rewards/upright_sharp/params/height_high',
}
HEAD_FIELDS={f'/curriculum/head_pose_range/params/range_stages/{i}/ranges/{j}/{k}'
             for i,j,k in ((3,0,0),(4,0,0),(4,1,1))}
RESET_FIELDS={f'/events/set_ground_state/params/sitting_joint_overrides/{i}'
              for i in (1,2,3,4,10,11,12,13)} | {
    '/events/set_ground_state/params/sitting_z_min','/events/set_ground_state/params/sitting_z_max',
    '/events/set_ground_state/params/standing_z_min','/events/set_ground_state/params/standing_z_max',
    '/events/v3_reset_ground_clearance'}
BANK_FIELDS={'/events/paired_error_reset/params/bank_path',
             '/observations/actor/terms/joint_pos/params/bank_path',
             '/observations/actor/terms/joint_vel/params/bank_path'}


def normalized(value):
    if isinstance(value,str):return re.sub(r' at 0x[0-9a-f]+',' at <address>',value)
    if isinstance(value,list):return [normalized(x) for x in value]
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
    return value


def differences(old,new,path=''):
    if type(old)!=type(new):return [{'path':path,'old':old,'new':new}]
    if isinstance(old,dict):
        out=[]
        for k in sorted(set(old)|set(new)):
            key=path+'/'+k
            if k not in old:out.append({'path':key,'old':'<absent>','new':new[k]})
            elif k not in new:out.append({'path':key,'old':old[k],'new':'<absent>'})
            else:out.extend(differences(old[k],new[k],key))
        return out
    if isinstance(old,list):
        if len(old)!=len(new):return [{'path':path,'old':old,'new':new}]
        return [v for i,(a,b) in enumerate(zip(old,new)) for v in differences(a,b,path+'/'+str(i))]
    return [] if old==new else [{'path':path,'old':old,'new':new}]


def main():
    a=json.loads(OLD.read_text());b=json.loads(NEW.read_text());rank1=json.loads(OLD_RANK1.read_text())
    assert a['num_envs']+rank1['num_envs']==b['num_envs']==8192
    assert a['agent']==b['agent'] and a['dependencies']==b['dependencies']
    assert a['head_precision_recipe'] is None and b['head_precision_recipe'] is None
    sections=('rewards','curriculum','commands','events','sim','actions','observations')
    changes=[]
    for section in sections:
        changes.extend(differences(normalized(a['environment'][section]),normalized(b['environment'][section]),'/'+section))
    allowed=HEIGHT_FIELDS|HEAD_FIELDS|RESET_FIELDS|BANK_FIELDS
    unknown=[x for x in changes if x['path'] not in allowed]
    assert not unknown, f'Unexpected V2/V3 recipe changes: {unknown}'
    assert {x['path'] for x in changes}==allowed
    source_changes=subprocess.check_output(['git','diff','--name-only',BASE,'--','src'],cwd=ROOT,text=True).splitlines()
    code=[p for p in source_changes if p.endswith(('.py','.cpp'))]
    assert set(code)=={'src/mjlab_microduck/custom_v2/microduck_v2.py','src/mjlab_microduck/custom_v2/training.py'}
    motor=json.loads((ROOT/'docs/v3-motor-parity.json').read_text());assert motor['status']=='PASS'
    out={'status':'PASS','v2_initial_rank0_run_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),
         'v3_initial_run_sha256':hashlib.sha256(NEW.read_bytes()).hexdigest(),
         'global_envs':8192,'v2_rank_envs':[a['num_envs'],rank1['num_envs']],
         'agent_and_dependencies_equal':True,'compared_sections':sections,
         'allowed_differences':changes,'numerical_source_code_changes':code,
         'classification':'Only V3 geometry-dependent standing height, HOME-relative head bounds, feasible seven-pose reset geometry and the identical paired-error bank path differ in compared configurations. Single-GPU execution changes floating-point order and cannot be a bitwise DDP reproduction.',
         'limitation':'Compares frozen first-segment configuration; subsequent HOME guard fix and reset evaluation fix need their own preflight. This does not imply physical hardware equivalence.'}
    dest=ROOT/'docs/v3-6804-recipe-audit.json';dest.write_text(json.dumps(out,indent=2)+'\n')
    print(f'PASS: V2 6804 PPO and compared recipe match except documented V3 geometry/HOME adaptations; {dest}')


if __name__=='__main__':main()
