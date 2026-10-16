"""Last unchanged-v8 formal allocation with read-only native-range fault attribution."""
from pathlib import Path
import importlib.util,sys,json,hashlib,signal,torch
r=Path(__file__).resolve().parent;w=r/'velstand-v8-repaired-worktree';out=r/'velstand-v8-allocation-002'
spec=importlib.util.spec_from_file_location('unchanged_velstand_controller',w/'scripts/train_v2_velstand.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from mjlab_microduck.custom_v2.feedback_diagnostics import FeedbackFaultRecorder
Original=m.WalkingTrainingSummary
instances=[]
def expected_native_range_fault(valid, stream_reason, source_status, source_raw):
 count=source_raw[:,0]
 return (~valid) & (stream_reason==13) & (source_status==-1) & (source_raw[:,4]>.5) & torch.isfinite(count) & ((count<0)|(count>1023))
# Real captured failure plus negative fixtures: never accept a valid-count spike,
# missing/invalid packet or numeric observer error as this explained event.
sample=torch.load(r/'velstand-v8-mature-feedback-001/feedback-first-faults-0.pt',map_location='cpu',weights_only=False)
raw=sample['first']['raw'].reshape(-1,7);valid=~sample['first']['invalid_lanes'].reshape(-1).bool();reason=sample['first']['reason'].reshape(-1);status=sample['first']['source_status'].reshape(-1)
explained=expected_native_range_fault(valid,reason,status,raw);assert int(explained.sum())==1 and torch.equal(explained,~valid)
for kind in ('valid_count','bad_packet','numeric'):
 test_raw=raw.clone();test_reason=reason.clone()
 if kind=='valid_count':test_raw[:,0]=512
 elif kind=='bad_packet':test_raw[:,4]=0
 else:test_reason[:]=11
 assert not bool(expected_native_range_fault(valid,test_reason,status,test_raw).any())
del sample,raw,valid,reason,status,explained
original_diagnostics=m.attach_velstand_diagnostics
def guarded_diagnostics(runner,path):
 monitor=original_diagnostics(runner,path);native=runner.alg.update
 def update():
  losses=native()
  summary=instances[-1]
  unexplained=int(summary.unexplained_invalid_lanes)
  nonfinite=int(summary.nonfinite_samples)
  invalid=int(summary.invalid_controls)
  if unexplained or nonfinite or invalid>32:
   (out/'readonly-feedback-stop.json').write_text(json.dumps(dict(unexplained_invalid_lanes=unexplained,nonfinite_samples=nonfinite,invalid_controls=invalid,maximum_native_range_invalid_controls=32),indent=2)+'\n')
   # Native bounded controller handles SIGTERM after counting this update,
   # then saves weights/normalizers/both Adam states before stopping.
   signal.raise_signal(signal.SIGTERM)
  return losses
 runner.alg.update=update
 return monitor
m.attach_velstand_diagnostics=guarded_diagnostics
class CapturedSummary(Original):
 def __init__(self,env,**kw):
  super().__init__(env,**kw);instances.append(self)
  f=env._v2_joint_feedback;self.faults=FeedbackFaultRecorder(f)
  self.first_qpos=torch.zeros_like(env.sim.data.qpos);self.first_qvel=torch.zeros_like(env.sim.data.qvel)
  self.first_estimated_q=torch.zeros_like(f.last_q);self.first_estimated_v=torch.zeros_like(f.last_v)
  self.first_current_raw=torch.zeros_like(f.pipeline.last_raw).reshape(env.num_envs,f.joints,-1)
  self.nonfinite_samples=torch.zeros((),dtype=torch.int64,device=env.device)
  self.unexplained_invalid_lanes=torch.zeros_like(self.nonfinite_samples)
  self.invalid_controls=torch.zeros_like(self.nonfinite_samples)
  (out/'readonly-feedback-hook.py').write_text(Path(__file__).read_text())
  (out/'readonly-feedback-hook.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Read-only GPU measurements only; original source/teacher/config/RNG unchanged. Native invalid-count episodes retain original termination. Any other feedback reason/nonfinite or >32 invalid controls stops after current optimizer update and saves.',maximum_native_range_invalid_controls=32),indent=2)+'\n')
  original=env.reward_manager.compute
  def capture(dt):
   result=original(dt);fresh=~f.ready & (self.faults.first_step<0)
   for target,value in ((self.first_qpos,env.sim.data.qpos),(self.first_qvel,env.sim.data.qvel),(self.first_estimated_q,f.last_q),(self.first_estimated_v,f.last_v),(self.first_current_raw,f.pipeline.last_raw.reshape(env.num_envs,f.joints,-1))):
    target.copy_(torch.where(fresh.reshape(-1,*([1]*(target.ndim-1))),value,target))
   self.faults.control_boundary(env.common_step_counter)
   _,valid=f.pipeline.query(f._time())
   expected=expected_native_range_fault(valid,self.faults.stream.reason,self.faults.last_source_status,self.faults.last_source_raw)
   self.unexplained_invalid_lanes.add_((~valid & ~expected).sum())
   self.invalid_controls.add_((~f.ready).sum())
   self.nonfinite_samples.add_((~self.latest['finite']).sum())
   return result
  env.reward_manager.compute=capture
 def report(self):
  result=super().report();result['diagnostic_nonfinite_control_samples']=int(self.nonfinite_samples);result['diagnostic_unexplained_invalid_lanes']=int(self.unexplained_invalid_lanes);result['diagnostic_invalid_controls']=int(self.invalid_controls)
  return result
m.WalkingTrainingSummary=CapturedSummary
sys.argv=['train_v2_velstand.py','--output',str(out),'--device','cuda:1','--num-envs','24576','--iterations','128','--resume',str(r/'velstand-v8-allocation-001/checkpoint_000128.pt'),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--walking-teacher',str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'),'--qualification',str(r/'velstand-v8-mature-qualified-native-range.json')]
try:
 m.main()
finally:
 for i,summary in enumerate(instances):
  f=summary.faults;env=summary.env;mask=f.first_step>=0;ids=mask.nonzero().flatten()
  payload=dict(env_ids=ids.cpu(),first_step=f.first_step[ids].cpu(),first={k:v[ids].cpu() for k,v in f.first.items()},qpos=summary.first_qpos[ids].cpu(),qvel=summary.first_qvel[ids].cpu(),estimated_q=summary.first_estimated_q[ids].cpu(),estimated_v=summary.first_estimated_v[ids].cpu(),current_raw=summary.first_current_raw[ids].cpu(),invalid_lane_control_counts=f.bad_controls.reshape(f.shape)[ids].cpu(),source_events=f.source_events.reshape(f.shape)[ids].cpu(),numeric_events=f.numeric_events.reshape(f.shape)[ids].cpu(),joint_names=list(env.scene['robot'].joint_names),nonfinite_samples=int(summary.nonfinite_samples),hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  torch.save(payload,out/f'feedback-first-faults-{i}.pt')
  report=dict(status='CAPTURE_SAVED',worlds_with_feedback_failure=len(ids),nonfinite_samples=int(summary.nonfinite_samples),first_steps=payload['first_step'].tolist(),scope='Read-only extra GPU buffers/pure query; no source/physics/actor/BC/RNG change. Native Warp trajectories are not promised bitwise identical across independent processes.',payload=str(out/f'feedback-first-faults-{i}.pt'))
  (out/f'feedback-capture-{i}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);f.close()
