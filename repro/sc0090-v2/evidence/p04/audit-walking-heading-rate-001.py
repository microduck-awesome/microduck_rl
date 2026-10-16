"""Saved-state audit: actual heading change vs body-z and Euler heading rate."""
from pathlib import Path
import torch,json,math,hashlib
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');torch.set_num_threads(2);rows=[]
def yaw(q):
 w,x,y,z=q.unbind(-1);return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def stats(x,mask):
 def m(v):return dict(mean=float(v.mean()),abs_mean=float(v.abs().mean()),abs_p95=float(torch.quantile(v.abs(),.95)))if v.numel()else None
 return dict(all=m(x),healthy=m(x[mask]))
for stage in ['nominal','measured']:
 for arm in ['unmapped','mapped']:
  d=R/f'velstand-teacher-zero-{arm}-{stage}-001';rep=json.loads((d/'report.json').read_text());buckets={}
  for seed in [2026101613,2026101614]:
   t=torch.load(d/f'trace-{seed}.pt',map_location='cpu',weights_only=False);x=t['measurements'];q=torch.cat((t['initial']['qpos'][None,:,3:7],x['root_quat_current']),0);angle=yaw(q);inc=torch.atan2(torch.sin(torch.diff(angle,dim=0)),torch.cos(torch.diff(angle,dim=0)));heading_fd=inc/.02
   # Current-qpos orientation vs original reward-boundary body angular velocity;
   # timestamps can differ by5ms; do not assert samplewise sensorparity.
   w,xx,y,z=x['root_quat_current'].unbind(-1);roll=torch.atan2(2*(w*xx+y*z),1-2*(xx*xx+y*y));sinpitch=(2*(w*y-z*xx)).clamp(-1,1);cospitch=(1-sinpitch.square()).clamp_min(1e-8).sqrt();omega=x['angular']
   predicted=(torch.sin(roll)*omega[:,:,1]+torch.cos(roll)*omega[:,:,2])/cospitch
   healthy=((x['height'][300:600]>=.102)&(x['tilt'][300:600]<=math.radians(20))&x['finite'][300:600]&x['feedback_ready'][300:600]).all(0)&~x['failed'].any(0)
   vals=dict(actual_heading_net_rate=heading_fd[300:600].mean(0),body_z_mean=omega[300:600,:,2].mean(0),predicted_euler_heading_mean=predicted[300:600].mean(0),orientation_correction_mean=(predicted[300:600]-omega[300:600,:,2]).mean(0),actual_minus_body_mean=(heading_fd[300:600]-omega[300:600,:,2]).mean(0),actual_minus_predicted_mean=(heading_fd[300:600]-predicted[300:600]).mean(0),actual_minus_predicted_frame_rmse=(heading_fd[300:600]-predicted[300:600]).square().mean(0).sqrt())
   for i,row in enumerate([v for v in rep['records']if v['seed']==seed]):
    sl=slice(i*16,(i+1)*16);buckets.setdefault(row['requested_twist'][0],[]).append((healthy[sl],{k:v[sl]for k,v in vals.items()}))
  for vx,items in buckets.items():
   healthy=torch.cat([v[0]for v in items]);vals={k:torch.cat([v[1][k]for v in items])for k in items[0][1]};rows.append(dict(stage=stage,arm=arm,requested_vx=vx,external_yaw=0.,trials=len(healthy),healthy_trials=int(healthy.sum()),metrics={k:stats(v,healthy)for k,v in vals.items()}))
out=dict(status='SAVED_TRAJECTORY_HEADING_RATE_AUDIT',new_GPU_controls=0,new_formal_controls=0,window='last6s of11s motion; actual heading from integratedrootquaternion, net derivative mean; notbodywz integral',rows=rows,limitations=['Bodyvelocity sensor at original rewardboundary maylag currentqpos by5ms; predictedinstantaneous Euler rate not claimedexact to50Hz secant.','Pertrial signed/absolute rates and all/healthy denominators retained.','This diagnoses measurement/objective alignment only; no newreward or effectiveness claim.','Raw actorIMU notstored inthisbattery; cannotinfer encoder/gyro calibrationerror fromphysicalangularvelocity.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(R/'walking-heading-rate-audit-001.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([v for v in rows if v['requested_vx']==.3],indent=2))
