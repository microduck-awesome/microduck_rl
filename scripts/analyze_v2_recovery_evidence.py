"""Summarize saved physical states without changing the acceptance metric."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('evidence',type=Path)
args=p.parse_args()
names=('standing','sitting','prone','frontier','left_side','right_side','supine')
cases=[]
for path in sorted(args.evidence.glob('traces-*.npz')):
    with np.load(path) as d:
        x=d['states'];nq=int(d['nq']);nv=int(d['nv'])
        bucket=d['bucket'];origins=d['env_origins']
        q=x[:,:,:nq];v=x[:,:,nq:nq+nv]
        height=q[:,:,2]-origins[None,:,2]
        cosine=1-2*(q[:,:,4]**2+q[:,:,5]**2)
        tilt=np.rad2deg(np.arccos(cosine.clip(-1,1)))
        linear=np.linalg.norm(v[:,:,:3],axis=-1)
        angular=np.linalg.norm(v[:,:,3:6],axis=-1)
        groups={}
        for i,name in enumerate(names):
            keep=bucket==i
            def tail(a):return a[-50:,keep]
            groups[name]=dict(trials=int(keep.sum()),
                final_stable=int((x[-1,keep,-2]!=0).sum()),
                ever_invalid=int((x[-1,keep,-1]!=0).sum()),
                last_second_height_mean_m=float(tail(height).mean()),
                last_second_tilt_median_deg=float(np.median(tail(tilt))),
                last_second_tilt_p95_deg=float(np.quantile(tail(tilt),.95)),
                last_second_linear_speed_p95=float(np.quantile(tail(linear),.95)),
                last_second_angular_speed_p95=float(np.quantile(tail(angular),.95)),
                last_second_head_mean_deg=np.rad2deg(tail(q[:,:,12:16]).mean((0,1))).tolist(),
                last_second_joint_mean_rad=tail(q[:,:,7:]).mean((0,1)).tolist())
            if name=='standing':
                fallen=(tilt[10:,keep]>45.) | (height[10:,keep]<.08)
                was_stable=x[:,:, -2]!=0
                groups[name].update(
                    fell_after_initial_200ms=int(fallen.any(0).sum()),
                    lost_achieved_half_second_stability=int((was_stable[:-1,keep]&~was_stable[1:,keep]).any(0).sum()))
        cases.append(dict(trace=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=groups))
assert cases,'No saved physical traces found'
result=dict(schema=1,scope='physical-state diagnosis; does not replace reward-boundary stability/fault acceptance',
    standing_fall_audit=dict(ignore_initial_seconds=.2,tilt_above_deg=45.,height_below_m=.08,
        gate='tilt OR height',initial_condition='registered noisy/upright standing spawn, not settled HOME'),
    tail_seconds=1.,cases=cases)
output=args.evidence/'physical-diagnostics.json'
output.write_text(json.dumps(result,indent=2)+'\n')
print(output)
