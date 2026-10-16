"""Closed-loop zero-request comparison, true heading and path with two references."""
from pathlib import Path
import torch,json,math,runpy,hashlib
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
cm=runpy.run_path(str(R/'velstand-head-bias-worktree/src/mjlab_microduck/custom_v2/head_metrics.py'))['camera_metrics']
def yaw(q):
 w,x,y,z=q.unbind(-1);return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def summarize(v,mask):
 return dict(all_mean=float(v.mean()),all_abs_mean=float(v.abs().mean()),all_abs_p95=float(torch.quantile(v.abs(),.95)),eligible_mean=float(v[mask].mean()) if mask.any() else None,eligible_abs_mean=float(v[mask].abs().mean()) if mask.any() else None)
results={};initials={};prefixes={};paths={};source={}
for stage in ('nominal','measured'):
 results[stage]={}
 for arm in ('unmapped','mapped'):
  directory=R/f'velstand-teacher-zero-{arm}-{stage}-001';report=json.loads((directory/'report.json').read_text());bucket={};source[f'{stage}/{arm}']=hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
  for seed in (2026101613,2026101614):
   tr=torch.load(directory/f'trace-{seed}.pt',map_location='cpu',weights_only=False);x=tr['measurements'];initial=tr['initial'];initials[(stage,arm,seed)]=initial
   q=torch.cat((initial['qpos'][None,:,3:7],x['root_quat_current']),0);p=torch.cat(((initial['qpos'][:,:3]-initial['origins'])[None],x['root_position_current']),0)
   prefixes[(stage,arm,seed)]=dict(q=q[:51],p=p[:51])
   angle=yaw(q);inc=torch.atan2(torch.sin(torch.diff(angle,dim=0)),torch.cos(torch.diff(angle,dim=0)));u=torch.cat((angle[:1],angle[:1]+inc.cumsum(0)),0)
   time=torch.arange(301,dtype=torch.float64)*.02;t=time-time.mean();rate=(t[:,None]*u[300:601].double()).sum(0)/t.square().sum()
   gate=((x['height'][300:600]>=.102)&(x['tilt'][300:600]<=math.radians(20))&x['finite'][300:600]&x['feedback_ready'][300:600]).all(0)&~x['failed'].any(0)
   delta=p-p[50:51];fx=torch.cos(angle[50]);fy=torch.sin(angle[50]);along=delta[...,0]*fx+delta[...,1]*fy;cross=delta[...,1]*fx-delta[...,0]*fy
   reset_delta=p-p[:1];reset_cross=reset_delta[...,1]*torch.cos(angle[0])-reset_delta[...,0]*torch.sin(angle[0])
   camera=cm(tr['camera'][1200:2401]);head=(x['head_position'][300:600]-x['head_command'][300:600]).abs().mean(0)*180/math.pi
   rows=[row for row in report['records'] if row['seed']==seed]
   for i,row in enumerate(rows):
    sl=slice(i*16,(i+1)*16);vx=row['requested_twist'][0]
    arrays=dict(heading_rate_rad_s=rate[sl],heading_change_11s_deg=(u[600,sl]-u[50,sl])*180/math.pi,onset_reference_cross_end_m=cross[600,sl],onset_reference_cross_rms_m=cross[50:601,sl].square().mean(0).sqrt(),reset_reference_cross_end_m=reset_cross[600,sl],onset_reference_along_speed_mps=along[600,sl]/11,body_vx_mean_mps=x['linear'][300:600,sl,0].mean(0),body_wz_sample_mean_rad_s=x['angular'][300:600,sl,2].mean(0),camera_rms_rad_s=camera['angular_speed_rms_rad_s'][sl],camera_p95_rad_s=camera['angular_speed_p95_rad_s'][sl],exposure20ms_p95_rad=camera['exposure_span_p95_rad'][sl],head_mae_mean_deg=head[sl].mean(-1))
    bucket.setdefault(vx,[]).append((gate[sl],~x['failed'][:,sl].any(0),x['feedback_ready'][:,sl].all(0),arrays))
    if vx==.3:paths[(stage,arm,seed)]=torch.stack((along[50:601,sl],cross[50:601,sl]),-1)
  out={}
  for vx,items in bucket.items():
   mask=torch.cat([x[0] for x in items]);survived=torch.cat([x[1] for x in items]);feedback=torch.cat([x[2] for x in items]);arrays={k:torch.cat([x[3][k] for x in items]) for k in items[0][3]}
   out[str(vx)]=dict(trials=len(mask),eligible=int(mask.sum()),survived=int(survived.sum()),faults=int((~feedback).sum()),metrics={k:summarize(v,mask) for k,v in arrays.items()})
  results[stage][arm]=out
comparisons=[]
for stage in ('nominal','measured'):
 for seed in (2026101613,2026101614):
  a=initials[(stage,'unmapped',seed)];b=initials[(stage,'mapped',seed)];assert all(torch.equal(a[k],b[k]) for k in a)
  pa=prefixes[(stage,'unmapped',seed)];pb=prefixes[(stage,'mapped',seed)]
  comparisons.append(dict(stage=stage,seed=seed,initial_exact=True,first_idle_second={k:dict(exact=torch.equal(pa[k],pb[k]),max_abs=float((pa[k]-pb[k]).abs().max())) for k in pa}))
result=dict(status='CLOSED_LOOP_TEACHER_ZERO_YAW_REVIEW_NOT_STUDENT_TRAINING',formal_controls=0,diagnostic_controls=537600,results=results,paired_initial_and_prefix=comparisons,report_sha256=source,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),criteria='Whole-trajectory walking survival and final6s height>=.102/tilt<=20/finite/feedback eligibility; all and eligible separate.',reference='Zero externalyaw. Geometric yaw fromcurrentrootqpos; reference path reported bothinitial-reset and actualcommand-onset heading. Onset reference does not erase reset-reference results. Native evaluator command errors are fed-command errors, not externaltracking.',scope='One fixed table fit onlynominalseed1612; physicaltestseed1613/1614,nominal/measured. Two arms exactinitials; idle prefix deviations explicitly reported. No policy update, no runtime change, no assertion all learned skills passed.')
p=R/'velstand-teacher-zero-review-001.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({s:{a:{v:dict(survived=x['survived'],rate_abs=x['metrics']['heading_rate_rad_s']['all_abs_mean'],cross_abs=x['metrics']['onset_reference_cross_end_m']['all_abs_mean'],vx=x['metrics']['body_vx_mean_mps']['all_mean']) for v,x in d.items()} for a,d in arms.items()} for s,arms in results.items()},indent=2))
# Saved-state scientific plot, not a new simulation/rendering claim.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
for ax,stage in zip(axes,('nominal','measured')):
 for arm,color in [('unmapped','tab:red'),('mapped','tab:blue')]:
  xy=torch.cat([paths[(stage,arm,s)] for s in (2026101613,2026101614)],1).numpy()
  for trial in range(xy.shape[1]):ax.plot(xy[:,trial,0],xy[:,trial,1],color=color,alpha=.16,lw=.6)
  mean=xy.mean(1);ax.plot(mean[:,0],mean[:,1],color=color,lw=2,label=arm+' mean (32 trials)')
 ax.plot([0,3.3],[0,0],'k--',label='requested straight path');ax.set(title=stage,xlabel='Along onset heading (m)',ylabel='Cross-track (m)');ax.axis('equal');ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.suptitle('Frozen walking512: +0.30 m/s, external yaw=0, 11 s movement\nCoordinates aligned to actual heading at command onset; all trajectories shown')
fig.savefig(R/'velstand-teacher-zero-paths-001.png',dpi=160)
