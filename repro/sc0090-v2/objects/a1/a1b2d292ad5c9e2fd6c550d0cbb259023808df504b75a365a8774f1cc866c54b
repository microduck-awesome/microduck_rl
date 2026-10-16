import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);args=p.parse_args()
cases=[]
for path in sorted(args.evidence.glob('traces-*.npz')):
 a=np.load(path); states=a['states']; bucket=a['bucket']; nq=int(a['nq']); origins=a['env_origins']
 x=states[:,bucket==0]; h=x[:,:,2]-origins[bucket==0,2][None,:]
 q=x[:,:,3:7]; tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,:,1]**2+q[:,:,2]**2),-1,1)))
 times=(np.arange(len(x))+1)*.02
 fallen=(times[:,None]>.2)&(tilt>45)&(h<.08)
 anyfallen=fallen.any(0); finalstable=x[-1,:,-2].astype(bool)
 invalid=x[-1,:,-1].astype(bool); recovered=finalstable&~invalid
 cases.append(dict(seed=int(path.stem.split('-')[-1]),trials=int(len(anyfallen)),fell_after_0p2s=int(anyfallen.sum()),fell_then_final_stable=int((anyfallen&recovered).sum()),final_stable=int(recovered.sum()),faults=int(invalid.sum()),fall_trial_indices=np.flatnonzero(anyfallen).tolist(),first_fall_time_s=[float(times[np.flatnonzero(fallen[:,i])[0]]) for i in np.flatnonzero(anyfallen)],trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
r=dict(schema=1,metric='standing initial bucket; any t > 0.2 s with physical root tilt > 45 deg AND root height above env origin < 0.08 m; terminal stability/faults retained separately',source='existing immutable evaluation traces; no new simulation and no changed policy',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases)
r['totals']={k:sum(c[k] for c in cases) for k in ['trials','fell_after_0p2s','fell_then_final_stable','final_stable','faults']}
(args.evidence/'standing-retention.json').write_text(json.dumps(r,indent=2)+'\n');print(args.evidence,r['totals'])
