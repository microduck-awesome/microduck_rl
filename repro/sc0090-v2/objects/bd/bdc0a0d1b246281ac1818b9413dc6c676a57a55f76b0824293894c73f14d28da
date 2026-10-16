"""Separate failure to catch a near-standing reset from falling after stability."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(__doc__);p.add_argument('evidence',type=Path);a=p.parse_args();cases=[]
for f in sorted(a.evidence.glob('traces-*.npz')):
 raw=np.load(f);x=raw['states'][:,raw['bucket']==0]
 h=x[:,:,2]-raw['env_origins'][raw['bucket']==0,2][None,:];q=x[:,:,3:7]
 tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,:,1]**2+q[:,:,2]**2),-1,1)))
 t=(np.arange(len(x))+1)*.02
 fallen=(t[:,None]>.2)&(tilt>45)&(h<.08)
 stable=x[:,:,-2].astype(bool)
 prior=np.concatenate([np.zeros((1,x.shape[1]),dtype=bool),np.maximum.accumulate(stable,axis=0)[:-1]],axis=0)
 fall_trials=fallen.any(0)
 first=np.where(fallen,np.arange(len(x))[:,None],len(x)).min(0)
 after_first=int(sum(prior[first[i],i] for i in np.flatnonzero(fall_trials)))
 cases.append(dict(seed=int(f.stem.split('-')[-1]),trials=x.shape[1],ever_completed_stable=int(stable.any(0).sum()),any_fall_after_0p2=int(fall_trials.sum()),first_fall_after_prior_half_s_stable=after_first,first_fall_before_any_half_s_stable=int(fall_trials.sum())-after_first,any_later_fall_after_half_s_stable=int((fallen&prior).any(0).sum()),trace_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
assert cases,'No saved standing traces found'
r=dict(schema=1,scope='Read-only existing standing-bucket trajectories',fall='t>.2s AND tilt>45deg AND z<.08m',established_stand='Saved stable flag: exact reward boundary >=.5s consecutive height/tilt/velocity/two-foot support success, strictly before the fall',cases=cases,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
r['totals']={k:sum(c[k] for c in cases) for k in ['trials','ever_completed_stable','any_fall_after_0p2','first_fall_after_prior_half_s_stable','first_fall_before_any_half_s_stable','any_later_fall_after_half_s_stable']}
(a.evidence/'standing-order.json').write_text(json.dumps(r,indent=2)+'\n');print(a.evidence,r['totals'])
