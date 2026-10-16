from pathlib import Path
import json,hashlib,time
r=Path(__file__).resolve().parent;rows=[];surveys=[];files={}
for update in (128,256):
 for stage in ('nominal','measured'):
  p=r/f'velstand-{update:04d}-commands-{stage}/report.json';d=json.loads(p.read_text());files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for case in ('idle','forward_+0.10','forward_+0.30','forward_-0.30','yaw_+0.50','yaw_-0.50','walking_head_+1'):
   records=[v for v in d['records'] if v['name']==case];assert len(records)==2
   phases={}
   for phase in ('move','stop','reverse','return'):
    n=sum(v['trials'] for v in records)
    def mean(key):
     values=[v['phases'][phase][key] for v in records]
     return [sum(x[i]*row['trials'] for x,row in zip(values,records))/n for i in range(len(values[0]))] if isinstance(values[0],list) else sum(x*row['trials'] for x,row in zip(values,records))/n
    phases[phase]=dict(velocity=mean('velocity_tail_mean'),head_mae_deg=mean('head_tail_mae_deg'),touchdown_raw_hz=mean('touchdown_frequency_per_foot_hz'),camera_per_trial_mean={k:sum(v['phases'][phase]['camera'][k]*v['trials'] for v in records)/n for k in records[0]['phases'][phase]['camera']})
   rows.append(dict(update=update,stage=stage,case=case,trials=sum(v['trials'] for v in records),survived=sum(v['survived_trials'] for v in records),feedback_invalid=sum(v['feedback_invalid_trials'] for v in records),phases=phases))
  p=r/f'velstand-{update:04d}-survey-{stage}/report.json';d=json.loads(p.read_text());files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for v in d['rows']:
   surveys.append(dict(update=update,stage=stage,case=v['case'],trials=v['trials'],healthy_hold=v['tail_healthy_hold_trials'],camera_all=v['camera_tail_per_trial_mean'],camera_healthy=v['camera_tail_healthy_per_trial_mean'],healthy_camera_denominator=v['camera_tail_healthy_trials'],camera_upright=v['camera_tail_upright_finite_per_trial_mean'],upright_denominator=v['camera_tail_upright_finite_trials'],feedback_invalid=v['feedback_invalid_trials']))
report=dict(status='VELSTAND_STAGE_BUDGET_CLOSED_PARTIALLY_QUALIFIED',created_unix=time.time(),formal_controls=150994944,commands=rows,surveys=surveys,inputs_sha256=files,metric_notes='Camera summaries are means of per-trial statistics, not pooled RMS/quantiles. Common command protocol/seeds; native Warp trajectories not claimed bitwise paired. Raw50Hz touchdown edges not debounced.',decision='Retain128 and256.256 two-condition common command survival336/336 each and improves isolated idle/head stability; moving camera remains6-7rad/s and recovery holds incomplete. No further automatic VelStand/head allocation. Standalone rise/stand candidates remain separate alternatives; hardware and whole skill suite not accepted.')
p=r/'velstand-stage-256-review.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
for st in ('nominal','measured'):
 print(st)
 for v in [x for x in surveys if x['update']==256 and x['stage']==st and x['case'] in ('idle','prone','crouch','sit')]:print(v['case'],v['healthy_hold'],v['camera_all']['angular_speed_rms_rad_s'],v['camera_healthy']['angular_speed_rms_rad_s'])
 for v in [x for x in rows if x['update']==256 and x['stage']==st and x['case'] in ('idle','forward_+0.30','yaw_+0.50','yaw_-0.50','walking_head_+1')]:print(v['case'],'move',v['phases']['move']['velocity'],'camera',v['phases']['move']['camera_per_trial_mean']['angular_speed_rms_rad_s'],'stopcam',v['phases']['stop']['camera_per_trial_mean']['angular_speed_rms_rad_s'],'returnstep',v['phases']['return']['touchdown_raw_hz'])
