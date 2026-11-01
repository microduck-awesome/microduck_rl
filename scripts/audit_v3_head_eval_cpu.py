"""Run a small V3 HOME-relative head evaluator smoke on CPU with a fixture.

The input checkpoint must be a disposable smoke fixture, not a formal policy.
This checks the evaluation path, not policy competence or hardware readiness.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from evaluate_v2_head import evaluate_seed
from mjlab_microduck.custom_v2.microduck_v2 import HOME_HEIGHT, HOME_JOINT_POSITIONS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['infos']['v2'].get('fixture_only') is True
    args.output.mkdir(parents=True,exist_ok=False)
    config=SimpleNamespace(task='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline',
        position_recipe=None,head_recipe=None,repeats=1,device='cpu',checkpoint=args.checkpoint,
        direct_head_target=False)
    cases=[{'name':'idle','range':'zero','target':[0.,0.,0.,0.]},
           {'name':'neck_positive','range':'small','target':[.05,0.,0.,0.]},
           {'name':'head_pitch_negative','range':'small','target':[0.,-.05,0.,0.]}]
    rows=[]
    for seed in (2026101503,2026101504):
        rows.extend(evaluate_seed(config,seed,cases,args.output))
        cfg=json.loads((args.output/f'eval-config-{seed}.json').read_text())
        init=cfg['scene']['entities']['robot']['init_state']
        assert init['joint_pos']==HOME_JOINT_POSITIONS
        assert abs(init['pos'][2]-HOME_HEIGHT)<1e-12
        trace=np.load(args.output/f'traces-{seed}.npz')
        assert trace['physical_head_rad'].shape==(300,3,4)
        assert trace['camera_quat_w'].shape==(1201,3,4)
        assert np.isfinite(trace['physical_head_rad']).all() and np.isfinite(trace['camera_quat_w']).all()
        np.testing.assert_allclose(trace['command_rad'][:100],0.,atol=0,rtol=0)
        expected=np.broadcast_to(np.array([c['target'] for c in cases],dtype=np.float32),(100,3,4))
        np.testing.assert_allclose(trace['command_rad'][100:200],expected,atol=0,rtol=0)
        np.testing.assert_allclose(trace['command_rad'][200:],0.,atol=0,rtol=0)
    report={'status':'PASS','device':'cpu','fixture_only':True,'seeds':[2026101503,2026101504],
            'cases_per_seed':len(cases),'protocol':'2 s idle + 2 s HOME-relative step + 2 s return; 200 Hz camera',
            'checks':['V3 default joint positions and root height','command timing and direction',
                      'head reward parity','finite physical head and camera traces'],
            'limitations':'Three diagnostic cases only, not the full command grid or a formal policy evaluation.',
            'results':rows}
    (args.output/'fixture-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS: V3 HOME-relative head evaluator; two seeds, three fixture cases each')


if __name__=='__main__':main()
