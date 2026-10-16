"""Recompute straightness in each trial's initial heading frame, saved data only."""
from pathlib import Path
import torch,json,hashlib,math
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
prefixes={'walk512':'walking-yaw-fine-000512-{stage}', 'vel256':'velstand-0256-commands-{stage}', 'head320':'velstand-head-free-bc-0320-commands-{stage}', 'bias384':'velstand-head-bias-0384-commands-{stage}', 'bias448':'velstand-head-bias-0448-commands-{stage}', 'bias512':'velstand-head-bias-0512-commands-{stage}', 'long_full320':'velstand-head-bc-longhold-full_bc-{stage}-001','long_head320':'velstand-head-bc-longhold-leg_bc-{stage}-001'}
def stats(v):return dict(mean=float(v.mean()),abs_mean=float(v.abs().mean()),abs_p95=float(torch.quantile(v.abs(),.95)),min=float(v.min()),max=float(v.max()))
result=dict(status='SAVED_STRAIGHT_PATHS_AUDITED',formal_controls_added=0,new_simulation_controls=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 protocol='Reference line through initial root XY, direction=initial projected body+x. Each world rotated separately before pooling. Commanded yaw is zero in selected forward-only cases. This measures path straightness relative to initial heading, not unobserved policy heading feedback. Body wz remains separately labeled; it is not integrated as true heading.',
 limitation='Saved ordinary command traces lack time-varying trunk quaternion: true heading drift cannot be reconstructed from body-wz samples. Root positions use original after-reward derived geometry with5ms lag. Initial-frame cross-track includes initial-idle drift; phase-added cross-track is also reported.',results={})
for label,pattern in prefixes.items():
 result['results'][label]={}
 for stage in ('nominal','measured'):
  directory=R/pattern.format(stage=stage);report=json.loads((directory/'report.json').read_text());phases=report['command_protocol'];out=[]
  for seed in sorted(set(row['seed'] for row in report['records'])):
   tr=torch.load(directory/f'trace-{seed}.pt',weights_only=False,map_location='cpu');x=tr['measurements'];initial=tr['initial'];rows=[z for z in report['records'] if z['seed']==seed]
   q=initial['qpos'][:,3:7];w,xx,y,z=q.unbind(-1);f=torch.stack((1-2*(y*y+z*z),2*(xx*y+w*z)),-1);f=f/f.norm(dim=-1,keepdim=True)
   origin=initial['qpos'][:,:2]-initial['origins'][:,:2];delta=x['root_position'][...,:2]-origin[None]
   forward=(delta*f[None]).sum(-1);cross=delta[...,1]*f[None,:,0]-delta[...,0]*f[None,:,1]
   start=0
   for row in rows:
    count=row['trials'];sl=slice(start,start+count);start+=count
    if not row['name'].startswith('forward'):continue
    for phase,a,b in phases:
     if phase not in ('move','reverse'):continue
     assert (x['command'][a:b,sl,1:]==0).all()
     reference_cross=cross[a:b,sl];previous_cross=cross[a-1,sl] if a else torch.zeros(count);previous_forward=forward[a-1,sl] if a else torch.zeros(count)
     gate=((x['height'][a:b,sl]>=.102)&(x['tilt'][a:b,sl]<=math.radians(20))&x['finite'][a:b,sl]&x['feedback_ready'][a:b,sl]).all(0)&~x['failed'][:b,sl].any(0)
     arrays=dict(cross_track_end_m=reference_cross[-1],cross_track_max_abs_m=reference_cross.abs().amax(0),cross_track_rms_m=reference_cross.square().mean(0).sqrt(),phase_added_cross_m=reference_cross[-1]-previous_cross,along_reference_speed_mps=(forward[b-1,sl]-previous_forward)/((b-a)*.02),body_vx_mean_mps=x['linear'][a:b,sl,0].mean(0),body_wz_sample_mean_rad_s=x['angular'][a:b,sl,2].mean(0))
     out.append(dict(seed=seed,case=row['name'],phase=phase,duration_s=(b-a)*.02,trials=count,eligible=int(gate.sum()),requested_vx=float(x['command'][a,sl,0][0]),all={k:stats(v) for k,v in arrays.items()},eligible_only={k:stats(v[gate]) for k,v in arrays.items()} if gate.any() else None))
  result['results'][label][stage]=dict(rows=out,report_sha256=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest())
p=R/'v2-straight-paths-audit-001.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for label,stages in result['results'].items():
 for stage,data in stages.items():
  rows=[x for x in data['rows'] if x['case']=='forward_+0.30' and x['phase']=='move'];print(label,stage,dict(duration=rows[0]['duration_s'],cross_abs_mean=sum(x['all']['cross_track_end_m']['abs_mean'] for x in rows)/len(rows),phase_cross_abs_mean=sum(x['all']['phase_added_cross_m']['abs_mean'] for x in rows)/len(rows),along_speed=sum(x['all']['along_reference_speed_mps']['mean'] for x in rows)/len(rows),body_wz=sum(x['all']['body_wz_sample_mean_rad_s']['mean'] for x in rows)/len(rows)))
