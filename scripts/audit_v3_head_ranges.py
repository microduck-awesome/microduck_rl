"""Check each V2 6804 head curriculum/precision command against V3 HOME."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.custom_v2.commands import sample_head
from mjlab_microduck.custom_v2.microduck_v2 import ASSETS, HOME_JOINT_POSITIONS
from mjlab_microduck.custom_v2.training import configure_head_precision_learning

HEAD=('neck_pitch','head_pitch','head_yaw','head_roll')


def main():
    cfg=load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline')
    model=mujoco.MjModel.from_xml_path(str(ASSETS/'robot.xml'))
    home=np.array([HOME_JOINT_POSITIONS[n] for n in HEAD])
    limits=np.array([model.joint(n).range for n in HEAD])
    safety=limits*.8
    torch.manual_seed(20261101)
    rows=[]
    def check(label,ranges,precision=0.):
        bounds=np.asarray(ranges,dtype=np.float64)
        assert bounds.shape==(4,2) and np.isfinite(bounds).all()
        assert np.all(bounds[:,0]<=0) and np.all(bounds[:,1]>=0)
        absolute=bounds+home[:,None]
        assert np.all(absolute[:,0]>=safety[:,0]-1e-9)
        assert np.all(absolute[:,1]<=safety[:,1]+1e-9)
        sampled=sample_head(ranges,4096,'cpu',.6,precision).numpy()
        assert np.all(sampled>=bounds[:,0]-1e-6) and np.all(sampled<=bounds[:,1]+1e-6)
        assert np.all(sampled+home>=limits[:,0]-1e-6) and np.all(sampled+home<=limits[:,1]+1e-6)
        rows.append({'name':label,'delta_ranges_rad':bounds.tolist(),
                     'absolute_target_ranges_rad':absolute.tolist(),
                     'minimum_hard_limit_clearance_deg':float(np.degrees(np.min(np.minimum(
                         absolute[:,0]-limits[:,0],limits[:,1]-absolute[:,1])))),
                     'sampled_commands':4096,'precision_probability':precision})
    check('initial',cfg.commands['head_pose'].ranges)
    for stage in cfg.curriculum['head_pose_range'].params['range_stages']:
        check('curriculum_step_'+str(stage['step']),stage['ranges'])
    for stage in (1,2,3,4):
        child=deepcopy(cfg)
        recipe=configure_head_precision_learning(child,stage)
        check('head_precision_'+str(stage),recipe['ranges'],child.commands['head_pose'].precision_probability)
    out={'status':'PASS','model_sha256':hashlib.sha256((ASSETS/'robot.xml').read_bytes()).hexdigest(),
         'head_order':HEAD,'home_rad':home.tolist(),'hard_limits_rad':limits.tolist(),
         'source':'V3 CAD model; physical servo calibration and physical stops not measured',
         'checks':'All command bounds and 4096 samples per stage respect the model range; 80%-of-CAD-bound policy follows the inherited V2 guard. Actor motor targets may intentionally overshoot to generate torque and are evaluated separately.',
         'ranges':rows}
    output=Path(__file__).resolve().parents[1]/'docs/v3-head-command-range-audit.json'
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(f'PASS: {len(rows)} V3 HOME-relative head ranges, {len(rows)*4096} sampled commands; {output}')


if __name__=='__main__':main()
