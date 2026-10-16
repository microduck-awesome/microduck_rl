import gc,hashlib,json,os,runpy,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'velstand-camera-worktree';status=r/'velstand-camera-ppo-admission-status.json'
def state(**kw):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(dict(updated_unix=time.time(),pid=os.getpid(),**kw),indent=2)+'\n');p.replace(status)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 assert json.loads((r/'velstand-camera-physical-003/report.json').read_text())['status']=='PASS_PHYSICS_SAMPLING_NOT_PPO_QUALIFIED'
 state(status='LOADING_AND_VALIDATING_SOURCE_TRANSITION',formal_controls=0)
 sys.path.insert(0,str(w/'scripts'));ns=runpy.run_path(str(w/'scripts/train_v2_velstand.py'),run_name='camera_trainer_preload')
 import torch
 from mjlab_microduck.custom_v2.camera_stability import make_camera_env_cfg
 parentpath=r/'velstand-v8-allocation-002/checkpoint_000256.pt';parent=torch.load(parentpath,map_location='cpu',weights_only=False);meta=parent['infos']['v2'];hashes=ns['source_hashes'](w)
 parentrun=json.loads((parentpath.parent/'run.json').read_text());task=meta['task']
 manifest=dict(schema=1,kind='camera_reward_and_readonly_fault_audit',task=task,parent_updates=256,parent_checkpoint_sha256=sha(parentpath),parent_run_sha256=sha(parentpath.parent/'run.json'),parent_source_sha256=meta['source_sha256'],target_source_sha256=hashes,changed_files=['scripts/train_v2_velstand.py','src/mjlab_microduck/custom_v2/camera_stability.py','src/mjlab_microduck/custom_v2/camera_feedback.py'],allowed_weights=[0.,-.01],equivalent_source_roots=[str(r/'velstand-v8-repaired-worktree'),str(w)])
 mp=r/'velstand-camera-source-transition-002.json';assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2)+'\n')
 for weight in (0.,-.01):
  cfg=make_camera_env_cfg(camera_weight=weight);cfg.scene.num_envs=24576;cfg.seed=parentrun['arguments']['seed'];ns['validate_camera_transition'](mp,parentpath,parent,hashes,task,cfg,weight)
 rejected=False
 try:ns['validate_camera_transition'](mp,parentpath,parent,{**hashes,'src/mjlab_microduck/actuator/sc0090_v5.py':'altered'},task,cfg,-.01)
 except AssertionError:rejected=True
 assert rejected
 (r/'velstand-camera-transition-cpu-002.json').write_text(json.dumps(dict(status='PASS_DECLARED_CAMERA_ONLY_TRANSITION',exact_parent_updates=256,clock=parent['infos']['env_state']['common_step_counter'] if 'env_state' in parent['infos'] else 'native load must verify6144',parent_sha256=sha(parentpath),physical_change_rejected=True,target_source_sha256=hashes,scope='CPU source/config contract; real model/N/twoAdam/RNG equality checked during actual runner load'),indent=2)+'\n')
 del parent;gc.collect()
 base=['--device','cuda:0','--standing-teacher',parentrun['arguments']['standing_teacher'],'--walking-teacher',parentrun['arguments']['walking_teacher'],'--validation-only']
 from mjlab_microduck.custom_v2.camera_feedback import explained_native_range
 for fixture in ('velstand-camera-mature-feedback-001/feedback-first-faults-0.pt','velstand-v8-mature-feedback-001/feedback-first-faults-0.pt'):
  event=torch.load(r/fixture,map_location='cpu',weights_only=False)['first'];raw=event['raw'].reshape(-1,7);valid=~event['invalid_lanes'].reshape(-1).bool();reason=event['reason'].reshape(-1);source=event['source_status'].reshape(-1)
  assert torch.equal(explained_native_range(valid,reason,source,raw),~valid)
  for kind in ('valid_count','bad_packet','numeric'):
   changed=raw.clone();why=reason.clone()
   if kind=='valid_count':changed[:,0]=512
   elif kind=='bad_packet':changed[:,4]=0
   else:why[:]=11
   assert not bool(explained_native_range(valid,why,source,changed).any())
 old_smoke=json.loads((r/'velstand-camera-smoke64-001/report.json').read_text());assert old_smoke['onnx_max_abs_error']<2e-5
 assert old_smoke['source_sha256']['src/mjlab_microduck/custom_v2/camera_stability.py']==hashes['src/mjlab_microduck/custom_v2/camera_stability.py']
 changed={k for k in set(hashes)|set(old_smoke['source_sha256']) if hashes.get(k)!=old_smoke['source_sha256'].get(k)}
 assert changed=={'scripts/train_v2_velstand.py','src/mjlab_microduck/custom_v2/camera_feedback.py'}
 (r/'velstand-camera-fault-audit-cpu.json').write_text(json.dumps(dict(status='PASS',real_fault_fixtures=2,negative_fixtures=6,scope='Read-only classification and stopping guard only; camera numerical function/PPO/BC/physics unchanged from passed64 smoke. No native episode termination removed.'),indent=2)+'\n')
 cases=[]
 for label,weight in (('mild',-.01),('control',0.)):
  cases.append((f'velstand-camera-mature24576-{label}-002',['--resume',str(parentpath),'--source-transition',str(mp),'--num-envs','24576','--iterations','16','--camera-weight',str(weight),'--force-timeouts','--native-range-budget','8']))
 reports=[]
 for name,args in cases:
  state(status='RUNNING_VALIDATION',phase=name,formal_controls=0);start=time.monotonic()
  sys.argv=[str(w/'scripts/train_v2_velstand.py'),'--output',str(r/name),*base,*args];ns['main']()
  a=json.loads((r/name/'report.json').read_text());assert a['status']=='ALLOCATION_COMPLETE'
  assert a['camera_audit']['raw_gated_fault_nonfinite_totals'][3]==0
  assert a['feedback_audit']['unexplained_invalid_lanes']==0 and a['feedback_audit']['invalid_controls']<=8
  if 'smoke64' in name:assert a['completed_updates']==5 and a['terminal_bootstrap_samples']>=32 and a['onnx_max_abs_error']<2e-5
  else:assert a['completed_updates']==272 and a['terminal_bootstrap_samples']>=12288
  reports.append(str(r/name/'report.json'));(r/name/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,scope='Per-case call after shared imports; complete supervisor import time separately visible; validation only'),indent=2)+'\n')
  gc.collect();torch.cuda.empty_cache()
 state(status='PPO_ADMISSION_REVIEW_READY',reports=reports,formal_controls=0,stage_max_formal_controls=75497472)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),formal_controls=0);raise
