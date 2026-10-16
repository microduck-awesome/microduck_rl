import dataclasses,hashlib,json
from pathlib import Path
import mujoco
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.custom_v2.ground_pick import GROUND_PICK_TASK
from mjlab_microduck.custom_v2.training import get_full_spec
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg
m=get_full_spec().compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
sites={name:{'id':int(m.site(name).id),'body':m.body(int(m.site_bodyid[m.site(name).id])).name,'local_position_m':m.site(name).pos.tolist(),'home_world_position_m':d.site_xpos[m.site(name).id].tolist()} for name in ('mouth_tip','head_camera','left_foot','right_foot')}
cfg=load_env_cfg(GROUND_PICK_TASK);a=load_rl_cfg(GROUND_PICK_TASK);old=make_microduck_ground_pick_env_cfg()
assert cfg.actions['joint_pos'].spike_rejection is False and cfg.actions['joint_pos'].o0_local_history
assert cfg.actions['joint_pos'].compile_measurement and cfg.actions['joint_pos'].fuse_registers
assert 'mouth_payload_force' not in cfg.rewards and cfg.events['mouth_payload'].mode=='step'
assert cfg.commands['twist'].period==4. and cfg.commands['twist'].randomize_phase is True
assert a.num_steps_per_env==24 and a.algorithm.num_learning_epochs==5 and a.algorithm.num_mini_batches==4
assert set(cfg.rewards)==set(old.rewards)-{'mouth_payload_force'}
assert {n:t.weight for n,t in cfg.rewards.items()}=={n:t.weight for n,t in old.rewards.items() if n!='mouth_payload_force'}
assert cfg.observations['actor'].terms['head_command'].params['dim']==4 and cfg.observations['actor'].terms['body_command'].params['dim']==6
r=dict(status='CPU_CONFIGURATION_AUDIT_ONLY',task=GROUND_PICK_TASK,sites=sites,body_jaw_soft_id=m.body('jaw_soft').id,policy_joints=POLICY_JOINTS,episode_s=cfg.episode_length_s,phase_period_s=cfg.commands['twist'].period,randomized_reset_phase=cfg.commands['twist'].randomize_phase,solver=dataclasses.asdict(cfg.sim.mujoco),capacity_ncon=cfg.sim.nconmax,rewards={n:{'function':t.func.__name__,'weight':t.weight} for n,t in cfg.rewards.items()},head_semantics='zero-padded command; phase task controls head, evaluate mouth direction and world camera separately',payload_bug='Historical weight=0 reward skipped by RewardManager; explicit step event pending force/terminal validation',terminal_qualified=False,physical_reach_qualified=False)
Path('runs-v2/groundpick-cpu-audit-001.json').write_text(json.dumps(r,indent=2,default=str)+'\n');print(json.dumps(r,default=str))
