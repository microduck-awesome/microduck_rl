import os,sys,json,hashlib,ast,math,copy,itertools,inspect,functools
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.pop('MICRODUCK_WARM_START',None)
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-health-history-worktree';B=R/'groundpick-repaired-worktree';sys.path.insert(0,str(W/'scripts'))
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.envs import mdp as native_mdp
from mjlab_microduck.custom_v2 import ground_pick_history as h
from mjlab_microduck.custom_v2.ground_pick import GROUND_PICK_TASK
from train_v2_ground_pick import source_hashes
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(2)
old=load_env_cfg(GROUND_PICK_TASK);new=load_env_cfg(h.HISTORY_TASK)
def fn_signature(fn):
 if isinstance(fn,functools.partial):return ('partial',fn_signature(fn.func),repr(fn.args),repr(fn.keywords))
 return (fn.__module__,fn.__qualname__,inspect.getsource(fn),tuple(fn_signature(c.cell_contents)if callable(c.cell_contents)else repr(c.cell_contents)for c in(fn.__closure__or())))
assert fn_signature(old.scene.terrain.spec_fn)==fn_signature(new.scene.terrain.spec_fn)
terrain_factory_equal=True
changed=[]
def diff(a,b,path=''):
 if type(a)!=type(b):changed.append(path);return
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:diff(a[k],b[k],path+'.'+str(k))
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b),path
  for i,(x,y)in enumerate(zip(a,b)):diff(x,y,path+f'[{i}]')
 elif a!=b:changed.append(path)
diff(asdict(old),asdict(new))
allowed={'.rewards.mouth_ground_proximity.func','.rewards.mouth_perpendicular_to_ground.func','.events.reset_robot_joints.func','.commands.twist.class_type','.scene.terrain.spec_fn'}
assert all(x in allowed or(x.startswith('.curriculum.')and x.endswith('.step')) for x in changed),changed
assert set(allowed)<=set(changed)
oldagent=asdict(load_rl_cfg(GROUND_PICK_TASK));newagent=asdict(load_rl_cfg(h.HISTORY_TASK));changedagent=[];prior=changed;changed=[];diff(oldagent,newagent);changedagent=changed;changed=prior
assert set(changedagent)=={'.algorithm.num_mini_batches','.run_name'},changedagent
assert newagent['algorithm']['num_mini_batches']==24 and newagent['algorithm']['num_learning_epochs']==5
assert 24576*24//24==4096*24//4==24576
assert 512*24576*24==301989888 and 512*5*24==61440
playold=load_env_cfg(GROUND_PICK_TASK,play=True);playnew=load_env_cfg(h.HISTORY_TASK,play=True);saved=changed;changed=[];diff(asdict(playold),asdict(playnew));playdiff=changed;changed=saved
assert fn_signature(playold.scene.terrain.spec_fn)==fn_signature(playnew.scene.terrain.spec_fn)
assert set(playdiff)=={'.rewards.mouth_ground_proximity.func','.rewards.mouth_perpendicular_to_ground.func','.scene.terrain.spec_fn'}
for force in (0.,.7,1.,2.):
 for head,selfcontact in itertools.product((False,True),repeat=2):
  raw=torch.linspace(-2,2,1001);clean=torch.full_like(raw,not(head or selfcontact),dtype=torch.bool);got=h.gate_positive_reward(raw,clean);expected=torch.tensor([float(v)if (not(head or selfcontact))or v<=0 else 0. for v in raw]);assert torch.equal(got,expected);assert torch.equal(got[raw<=0],raw[raw<=0])
class Robot:
 def __init__(self,n):
  self.data=SimpleNamespace(default_joint_pos=torch.zeros(n,14),default_root_state=torch.zeros(n,13));self.data.default_root_state[:,3]=1;self.q=torch.full((n,14),99.);self.root=torch.full((n,13),98.);self.vel=torch.full((n,14),97.)
 def find_joints(self,*a,**kw):return list(range(14)),list(h.POLICY_JOINTS)
 def write_root_state_to_sim(self,x,env_ids):self.root[env_ids]=x
 def write_joint_state_to_sim(self,q,v,env_ids):self.q[env_ids]=q;self.vel[env_ids]=v
class Scene(dict):pass
n=4096;robot=Robot(n);scene=Scene(robot=robot);scene.env_origins=torch.zeros(n,3);env=SimpleNamespace(num_envs=n,device='cpu',scene=scene);ids=torch.arange(n);native=native_mdp.reset_joints_by_offset
try:
 def home(env,env_ids,**params):env.scene['robot'].q[env_ids]=0;env.scene['robot'].vel[env_ids]=0
 native_mdp.reset_joints_by_offset=home;torch.manual_seed(2618)
 h.reset_phase_consistent_low(env,ids,(0,0),(0,0),None);mask=env._gp_low_reset_mask;assert .22<float(mask.float().mean())<.28
 assert torch.equal(robot.q[~mask],torch.zeros_like(robot.q[~mask]));assert float((robot.q[mask]-torch.tensor(h.LOW_Q)).abs().max())<=.003001
 cmd=h.PhaseConsistentGroundPickCommand.__new__(h.PhaseConsistentGroundPickCommand);cmd._env=env;cmd._gp_phase=torch.zeros(n);cmd._randomize_phase=True;cmd._period=4.;cmd.vel_command_b=torch.zeros(n,3)
 cmd.reset(ids);assert bool((cmd._gp_phase[mask]==.375).all());assert float(cmd._gp_phase[~mask].std())>.25
 cmd.compute(0.);assert torch.equal(cmd.vel_command_b[:,0],torch.cos(2*torch.pi*cmd._gp_phase));assert torch.equal(cmd.vel_command_b[:,1],torch.sin(2*torch.pi*cmd._gp_phase));assert bool((cmd.vel_command_b[:,2]==0).all())
 selected=ids[::17];untouched=torch.ones(n,dtype=torch.bool);untouched[selected]=False
 priorq=robot.q.clone();priorroot=robot.root.clone();priorphase=cmd._gp_phase.clone();priormask=mask.clone();h.reset_phase_consistent_low(env,selected,(0,0),(0,0),None);cmd.reset(selected)
 assert torch.equal(robot.q[untouched],priorq[untouched]) and torch.equal(robot.root[untouched],priorroot[untouched]);assert torch.equal(cmd._gp_phase[untouched],priorphase[untouched]) and torch.equal(mask[untouched],priormask[untouched])
 assert bool((cmd._gp_phase[selected][mask[selected]]==.375).all())
 shadowenv=copy.copy(env);shadow=copy.deepcopy(cmd,{id(env):id(env)}) if False else copy.deepcopy(cmd,{id(env):shadowenv});phase=cmd._gp_phase.clone();shadow.compute(.02);assert torch.equal(cmd._gp_phase,phase) and shadow._env is shadowenv
 exposure=env._gp_reset_exposure.tolist()
finally:native_mdp.reset_joints_by_offset=native
candidate=json.loads((R/'groundpick-torque-balanced-margin-008.json').read_text());assert sha(R/'groundpick-torque-balanced-margin-008.json')==h.LOW_GEOMETRY_SHA256
assert tuple(candidate['desired_physical_q'])==h.LOW_Q and candidate['root_pitch_rad']==h.LOW_ROOT_PITCH and candidate['root_z']==h.LOW_ROOT_HEIGHT
course={}
for name,term in old.curriculum.items():
 for key,stages in term.params.items():
  if isinstance(stages,list):
   course[name]=[{'old_step':a['step'],'new_step':b['step']}for a,b in zip(stages,new.curriculum[name].params[key]) if isinstance(a,dict)and 'step'in a]
assert max(v['new_step']for s in course.values()for v in s)==8000
assets=[p for p in(B/'src').rglob('*')if p.is_file()and p.suffix not in('.py','.pyc','.cpp')and '__pycache__'not in p.parts]
assert all(sha(p)==sha(W/p.relative_to(B))for p in assets)
changes=[]
for p in sorted([*W.glob('src/**/*.py'),W/'scripts/train_v2_ground_pick.py']):
 prior=B/p.relative_to(W)
 if not prior.exists()or sha(prior)!=sha(p):changes.append(dict(path=str(p.relative_to(W)),old_sha256=sha(prior)if prior.exists()else None,new_sha256=sha(p)))
result=dict(status='CPU_CONTRACT_PASS_GPU_NOT_QUALIFIED',source=str(W),task=h.HISTORY_TASK,formal_added=0,gpu_added=0,config_changes=changed,agent_changes=changedagent,play_changes=playdiff,source_changes=changes,source_sha256=source_hashes(W),assets_exact=len(assets),negative_orientation_preserved=True,head_force_threshold_not_used_for_gate=True,partial_reset_selected_only=True,mask_binds_physical_pose_and_phase=True,phase_compute_inherited=True,terrain_closure_source_values_exact=terrain_factory_equal,shadow_phase_copy_cpu=True,actual_reset_exposure=exposure,physical_endpoint=dict(candidate_sha256=h.LOW_GEOMETRY_SHA256,nominal_low_goal=64,measured_low_goal=61,total_each=64,measured_failures='Three unloaded mouth heights>25mm; all128 nohead/noself at50Hz rewardboundary and nofeedback/nonfinite; not200Hzcontactproof',reset_uses='desired physicalq, originalprobe root+.002 and±.003rad noise; NEVER compensatedcommand',payload='Original10–40g sampling unchanged; lowφ.375 has0force, rises after.425 under nativeevent;0/30g endpoint is not40gqualification'),curriculum_mapping=course,budget=dict(N=24576,T=24,E=5,M=24,updates=512,new_controls=301989888,prior_controls=75497472,total_cap=377487360,Adam_steps=61440,end_clock=12288,max_course_clock=8000,physical_period_s=4,physical_episode_s=20),remaining=['GPU actual reset masks/obs phase/payload timing before actor firstaction','Realpartialreset and terminal-shadow with lowstart/phasewrap/force ramp, not typeallowlist alone','64x5 ONNX fresh and actual24576x16 correctness; throughputreported','Formal512 notstarted; GPU0 waits parentRoulade completion and explicit handoff'],script_sha256=sha(Path(__file__)))
(R/'groundpick-health-history-cpu-003.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['source_sha256','source_changes']},indent=2))
