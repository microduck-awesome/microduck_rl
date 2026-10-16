"""V2 GroundPick on actual historical task-space rewards, with explicit payload."""
from copy import deepcopy
from . import microduck_v2 as v2
from .training import _full_contact_robot,guard_reset_joints
GROUND_PICK_TASK='Mjlab-GroundPick-Flat-MicroDuck-V2-Repaired-S0-O0-R1'

def mouth_payload_step(env,env_ids,**params):
    """Apply the next control interval's world wrench after native phase compute.

    The historical weight-zero reward never executed under locked mjlab.
    This event writes (including zero) once each control step. No reward call
    applies physics, and terminal capture must write only isolated xfrc.
    """
    from mjlab_microduck.tasks.mdp import apply_mouth_payload_force
    assert env_ids is None, 'Payload step is full-batch only'
    apply_mouth_payload_force(env,**params)

def initialize_ground_pick_payload(env):
    """Explicit initial/reset force for the phase already present in actor obs."""
    matches=[t for t in env.event_manager._mode_term_cfgs['step'] if t.func is mouth_payload_step]
    assert len(matches)==1
    mouth_payload_step(env,None,**matches[0].params)

def make_ground_pick_env_cfg(play=False,stage='nominal'):
    from mjlab.managers import EventTermCfg,TerminationTermCfg
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg
    from .runtime import configure_origin_markers
    base=v2.make_v2_sensor_env_cfg(play=play,stage=stage)
    cfg=deepcopy(make_microduck_ground_pick_env_cfg(play=play,rough=False))
    cfg.scene.entities['robot']=_full_contact_robot(base.scene.entities['robot'])
    cfg.scene.spec_fn=base.scene.spec_fn
    for sensor in cfg.scene.sensors:
        if sensor.name=='head_impact_contact':sensor.fields=('found','force')
    cfg.actions['joint_pos']=deepcopy(base.actions['joint_pos'])
    action=cfg.actions['joint_pos'];action.scale=1.
    action.spike_rejection=False;action.o0_local_history=True
    action.compile_measurement=True;action.fuse_registers=True;action.reset_graph_chunk=19
    v2._policy_selectors(cfg)
    for name in ('randomize_joint_friction','randomize_joint_damping','randomize_armature','randomize_motor_gains','encoder_bias'):
        cfg.events.pop(name,None)
    for name in ('expand_bam_friction_fields','v2_runtime','mouth_reset','mouth_hold'):
        if name in base.events:cfg.events[name]=deepcopy(base.events[name])
        else:cfg.events.pop(name,None)
    for group in ('actor','critic'):
        for name in ('joint_pos','joint_vel'):
            cfg.observations[group].terms[name]=deepcopy(base.observations[group].terms[name])
        for term in cfg.observations[group].terms.values():
            if term.func is mdp.raw_accelerometer:term.func=v2.body_accelerometer_direction;term.params={}
    for name in ('foot_air_time','foot_contact_forces'):
        if name in cfg.observations['critic'].terms:
            cfg.observations['critic'].terms[name]=deepcopy(base.observations['critic'].terms[name])
    cfg.terminations['sc0090_feedback_fault']=TerminationTermCfg(func=mdp.v2_feedback_fault,time_out=False)
    cfg.events['reset_base'].params['pose_range']['z']=(0.,.01)
    cfg.events['v2_reset_joint_guards']=EventTermCfg(func=guard_reset_joints,mode='reset')
    payload=cfg.rewards.pop('mouth_payload_force')
    assert payload.weight==0. and payload.func is mdp.apply_mouth_payload_force
    cfg.events['mouth_payload']=EventTermCfg(func=mouth_payload_step,mode='step',params=deepcopy(payload.params))
    cfg.sim.mujoco.timestep=.005;cfg.decimation=4
    # Full CAD contact; capacity only, independent overflow guards are required.
    cfg.sim.nconmax=64
    configure_origin_markers(cfg)
    return cfg

def register_ground_pick_task():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import MicroduckGroundPickRlCfg
    for task,stage in ((GROUND_PICK_TASK,'nominal'),(GROUND_PICK_TASK+'-Measured','measured')):
        agent=deepcopy(MicroduckGroundPickRlCfg)
        agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
        agent.experiment_name='microduck_v2_p04_groundpick';agent.run_name='groundpick_'+stage
        register_mjlab_task(task_id=task,env_cfg=make_ground_pick_env_cfg(stage=stage),
            play_env_cfg=make_ground_pick_env_cfg(play=True,stage=stage),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
