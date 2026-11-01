"""Explain 50 Hz near-limit recovery events using recorded states/actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'src/mjlab_microduck/custom_v2/assets/robot.xml'
NAMES=('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
       'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw',
       'right_hip_roll','right_hip_pitch','right_knee','right_ankle')
BUCKETS=('standing','sitting','prone','frontier','left_side','right_side','supine')


def longest_runs(mask):
    count=np.zeros(mask.shape[1],dtype=np.int32)
    best=np.zeros_like(count);end=np.zeros_like(count)
    for i,row in enumerate(mask):
        count=np.where(row,count+1,0)
        improve=count>best
        best=np.maximum(best,count)
        end=np.where(improve,i,end)
    return best,end


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();m=mujoco.MjModel.from_xml_path(str(MODEL))
    metadata=json.loads((MODEL.parent/'model_metadata.json').read_text())
    records=[]
    for seed in (2026101501,2026101502):
        path=args.traces/f'diagnosis-trace-{seed}.npz'
        with np.load(path,allow_pickle=False) as loaded:
            q=loaded['qpos']; actions=loaded['actions'];dt=float(loaded['dt'])
        assert dt==.02 and q.shape[0]==401 and actions.shape[0]==400 and q.shape[1]%7==0
        group_size=q.shape[1]//7
        assert actions.shape[1:]==(q.shape[1],14)
        for group,bucket in enumerate(BUCKETS):
            begin=group*group_size;end=begin+group_size
            for action_index,name in enumerate(NAMES):
                joint=m.joint(name);lo,hi=map(float,joint.range);adr=int(joint.qposadr[0])
                data=q[1:,begin:end,adr];clearance=np.minimum(data-lo,hi-data)
                best,ends=longest_runs(clearance<=math.radians(5))
                k=int(np.argmax(best));lane=begin+k;last=int(ends[k]);first=last-int(best[k])+1
                record={'seed':seed,'bucket':bucket,'joint':name,'samples':group_size,
                        'near_5deg_over_1s_samples':int((best>50).sum()),
                        'max_near_5deg_s':float(best[k]*dt),
                        'hard_limit_violation_samples':int((clearance<0).any(axis=0).sum()),
                        'minimum_clearance_rad':float(clearance.min())}
                if best[k]>50:
                    times=sorted(set((0,first,min(first+50,399),last,399)))
                    points=[]
                    for t in times:
                        qq=q[t+1,lane];quat=qq[3:7]
                        tilt=math.degrees(math.acos(float(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1))))
                        points.append({'time_s':float((t+1)*dt),'physical_q_rad':float(qq[adr]),
                                       'policy_target_rad':float(actions[t,lane,action_index]+metadata['home_joint_positions_rad'][name]),
                                       'root_height_m':float(qq[2]),'root_tilt_deg':tilt})
                    record['example']={'world':lane,'start_s':float((first+1)*dt),
                                       'end_s':float((last+1)*dt),'limits_rad':[lo,hi],
                                       'points':points}
                records.append(record)
    result={'status':'PASS','samples_at_hz':50,'interpretation':'Observed actor target and physical position; <=20 ms sampling can underestimate 200 Hz peak violations. Root height/tilt alone do not establish stable recovery.',
            'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'records':records}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {len(records)} seed/bucket/joint diagnostic records: {args.output}')


if __name__=='__main__':main()
