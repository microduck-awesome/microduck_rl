"""V2 commanded sit/stand; historical slew curriculum on frozen SC0090 physics."""
from copy import deepcopy
from . import microduck_v2 as v2
from .training import _full_contact_robot,guard_reset_joints,SIT_CANDIDATE,SIT_HEIGHT,STAND_HEIGHT,get_full_spec

TASK='Mjlab-SitStand-Flat-MicroDuck-V2-Repaired-S0-O0-R1'

def make_sitstand_env_cfg(play=False, stage='nominal'):
    from mjlab.managers import EventTermCfg,TerminationTermCfg
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
    from .runtime import configure_origin_markers
    from .commands import BalancedHeadCommandCfg
    base=v2.make_v2_sensor_env_cfg(play=play,stage=stage)
    cfg=deepcopy(make_microduck_sitstand_env_cfg(play=play,rough=False))
    cfg.scene.entities['robot']=_full_contact_robot(base.scene.entities['robot'])
    cfg.scene.spec_fn=base.scene.spec_fn
    cfg.actions['joint_pos']=deepcopy(base.actions['joint_pos'])
    action=cfg.actions['joint_pos'];action.scale=1.0
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
            if term.func is mdp.raw_accelerometer:
                term.func=v2.body_accelerometer_direction;term.params={}
    # The historical SitStand template predates the shared finite sensor guards.
    # Preserve its critic fields/order while using the qualified sensor readers.
    for name in ('foot_air_time','foot_contact_forces'):
        cfg.observations['critic'].terms[name]=deepcopy(base.observations['critic'].terms[name])
    cfg.terminations['sc0090_feedback_fault']=TerminationTermCfg(func=mdp.v2_feedback_fault,time_out=False)
    cfg.events['reset_base'].params['pose_range']['z']=(0.,.01)
    ground=cfg.events['set_ground_state'].params
    ground.update(sitting_joint_overrides=SIT_CANDIDATE.copy(),sitting_z_min=SIT_HEIGHT,
        sitting_z_max=SIT_HEIGHT+.015,standing_z_min=v2.HOME_HEIGHT,standing_z_max=v2.HOME_HEIGHT+.01)
    cfg.events['v2_reset_joint_guards']=EventTermCfg(func=guard_reset_joints,mode='reset')
    for section in (cfg.rewards,cfg.curriculum):
        for term in section.values():
            params=term.params
            if 'sit_overrides' in params:params['sit_overrides']=SIT_CANDIDATE.copy()
            if 'sit_z' in params:params['sit_z']=SIT_HEIGHT
            if 'stand_z' in params:params['stand_z']=STAND_HEIGHT
    cfg.commands['twist'].sit_z=SIT_HEIGHT;cfg.commands['twist'].stand_z=STAND_HEIGHT
    cfg.commands['head_pose']=BalancedHeadCommandCfg(**{**vars(cfg.commands['head_pose']),'zero_command_prob':.5})
    model=get_full_spec().compile()
    for stage_cfg in cfg.curriculum['head_pose_range'].params['range_stages']:
        stage_cfg['ranges']=tuple((max(lo,float(model.joint(n).range[0])*.8),min(hi,float(model.joint(n).range[1])*.8))
            for n,(lo,hi) in zip(('neck_pitch','head_pitch','head_yaw','head_roll'),stage_cfg['ranges']))
    cfg.sim.mujoco.timestep=.005;cfg.decimation=4
    # Historical XL330's 200 contacts/world also allocates the convex EPA
    # workspace at 200*N; it OOMs at N=32768 before one control step on L40.
    # A bounded V2 capacity, with per-physics-step and CCD-overflow admission
    # guards. Contact masks and solver 30/50 are unchanged.
    cfg.sim.nconmax=64
    configure_origin_markers(cfg)
    return cfg

def register_sitstand_task():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
    for task,stage in ((TASK,'nominal'),(TASK+'-Measured','measured')):
        agent=deepcopy(MicroduckSitStandRlCfg)
        agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
        agent.experiment_name='microduck_v2_p04_sitstand';agent.run_name='sitstand_'+stage
        register_mjlab_task(task_id=task,env_cfg=make_sitstand_env_cfg(stage=stage),
            play_env_cfg=make_sitstand_env_cfg(play=True,stage=stage),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
