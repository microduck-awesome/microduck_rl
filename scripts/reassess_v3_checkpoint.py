"""Apply the original recovery promotion rule to a corrected evaluation.

The input checkpoint must predate the invalid 500-step assessment. No PPO
updates are performed; all policy, optimizer, normalizer, RNG and clock bytes
are semantically verified unchanged. Output goes to a new directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab_microduck.custom_v2 import recovery


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return isinstance(b,type(a)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.evaluation.read_text())
    parent_run=args.checkpoint.parent/'run.json'
    run=json.loads(parent_run.read_text())
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    info=source['infos'];v2=info['v2']
    assert args.checkpoint.name=='checkpoint_000500.pt' and v2['completed_updates']==500
    assert v2['recovery']['stage']==0 and not v2.get('eval_history')
    assert report['checkpoint_sha256']==sha(args.checkpoint) and report['evaluation_stage']==0
    assert report['task']==run['task'] and report['robot_xml_sha256']=='45a9ce13c521ac07080381e9d5597b040e1988b98fb9f89c981008a7dac778dd'
    assert {case['seed'] for case in report['cases']}=={2026101501,2026101502}
    assert all(g['trials']==128 and g['faults']==0 for c in report['cases'] for g in c['groups'].values())
    for seed in (2026101501,2026101502):
        evidence=json.loads((args.evaluation.parent/f'clearance-{seed}.json').read_text())
        assert evidence['samples_per_bucket']==128 and evidence['non_sitting_initial_home_max_abs_error_rad'] < 1e-6
        assert len(evidence['joint_results'])==7 and all(len(joints)==14 for joints in evidence['joint_results'].values())
    env=SimpleNamespace(num_envs=1,device='cpu',common_step_counter=info['env_state']['common_step_counter'])
    env._sc0090_recovery=recovery.SC0090RecoveryState(env)
    env._sc0090_recovery.load_state_dict(v2['recovery'])
    env._sc0090_eval_history=dict(v2.get('eval_history',{}))
    request={'schema':1,'checkpoint_sha256':sha(args.checkpoint),'config_sha256':sha(parent_run),
             'iteration':500,'stage':0,'seeds':[2026101501,2026101502],
             'samples_per_group':128,'duration_s':8.,'deterministic':True}
    decision=recovery.sc0090_apply_recovery_evaluation(env,{**request,'cases':report['cases']},request)
    old_recovery=v2['recovery'];old_history=v2.get('eval_history')
    v2['recovery']=env._sc0090_recovery.state_dict()
    v2['eval_history']=decision
    v2['total_environment_steps']=500*run['num_envs']*run['agent']['num_steps_per_env']
    assert v2['total_environment_steps']==98_304_000
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'run.json').write_bytes(parent_run.read_bytes())
    target=args.output/'resume_000500.pt'
    torch.save(source,target)
    restored=torch.load(target,map_location='cpu',weights_only=False)
    assert same(source,restored)
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    modified=restored['infos']['v2']
    modified['recovery']=old_recovery
    modified['eval_history']=old_history
    del modified['total_environment_steps']
    assert same(original,restored), 'Non-curriculum state changed'
    result={'status':'PASS','parent_checkpoint':str(args.checkpoint.resolve()),
            'parent_checkpoint_sha256':sha(args.checkpoint),'evaluation':str(args.evaluation.resolve()),
            'evaluation_sha256':sha(args.evaluation),'corrected_resume':str(target.resolve()),
            'corrected_resume_sha256':sha(target),'prior_controls':98_304_000,'decision':decision,
            'scope':'Original promotion logic on corrected V3 HOME evaluation; actor, critic, normalizers, optimizer, torch RNG and clock unchanged. Proxy generator was not in original checkpoint.'}
    (args.output/'reassessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
