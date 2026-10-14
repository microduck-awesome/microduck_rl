"""Microduck V2 CAD task: 14 policy actions plus an independent mouth.

Targets mjlab 1.3.0 and microduck_rl b02fdea4902c2b93b7d34a83a6f05da4ececb7d3.
The XML PD is a structural preview. The BAM task uses frozen v5 SC0090 dynamics.
"""
from copy import deepcopy
from dataclasses import fields,is_dataclass
import json
from pathlib import Path
import re
import mujoco
import torch
# Keep mjlab/task imports inside functions. Its plugin discovery imports this
# module from tasks/__init__.py; eagerly importing tasks here also made a direct
# import re-enter the stock registrations before this module defined register().

POLICY_JOINTS=(
    'left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
    'neck_pitch','head_pitch','head_yaw','head_roll',
    'right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle',
)
POLICY_PATTERN='^('+'|'.join(POLICY_JOINTS)+')$'
ASSETS=Path(__file__).resolve().parent/'assets'
META=json.loads((ASSETS/'model_metadata.json').read_text(encoding='utf-8'))
HOME_HEIGHT=META['home_height_m']
TASK_ID='Mjlab-Velocity-Flat-MicroDuck-V2-PD'
BAM_DEBUG_TASK_ID='Mjlab-Velocity-Flat-MicroDuck-V2-BAM-Ideal-Debug'
SENSOR_TASK_ID='Mjlab-Velocity-Flat-MicroDuck-V2-S0-O0-R1'

def get_spec():
    return mujoco.MjSpec.from_file(str(ASSETS/'robot_walk.xml'))

def _is_v2(asset):
    names=getattr(asset,'joint_names',())
    return 'mouth' in names and all(n in names for n in POLICY_JOINTS)

def install_policy_mapping():
    from mjlab_microduck.tasks import mdp
    # Preserve stock tasks. Only a robot with this full named layout uses the
    # new mapping; an actuated mouth is never mislabeled as passive_*.
    if getattr(mdp,'_microduck_v2_mapping',False):return
    original_servo=mdp._servo_joint_ids
    def servo_ids(env,asset):
        if _is_v2(asset):
            ids,names=asset.find_joints(POLICY_JOINTS,preserve_order=True)
            if tuple(names)!=POLICY_JOINTS:raise ValueError('V2 policy joint order mismatch')
            return ids
        return original_servo(env,asset)
    mdp._servo_joint_ids=servo_ids
    original_positions=mdp._servo_joint_pos
    def servo_positions(env,asset):
        if _is_v2(asset):
            feedback=getattr(env.unwrapped,'_v2_joint_feedback',None)
            if feedback is not None:return feedback.last_q[:,:14]
        return original_positions(env,asset)
    mdp._servo_joint_pos=servo_positions
    original_metadata=mdp._get_base_metadata_no_passive
    def metadata(env,run_path):
        result=original_metadata(env,run_path)
        if _is_v2(env.scene['robot']):
            ids=[result['joint_names'].index(n) for n in POLICY_JOINTS]
            for field in ('joint_names','joint_stiffness','joint_damping','default_joint_pos'):
                result[field]=[result[field][i] for i in ids]
            if getattr(env.unwrapped,'_v2_joint_feedback',None) is not None:
                from mjlab_microduck.sc0090_v2.contract import export_contract,load_contract
                result.update(export_contract(env,load_contract(ASSETS/'training.json')))
            else:
                result['microduck_robot']='microduck-v2-sc0090'
                result['microduck_runtime']='diagnostic-only'
        return result
    mdp._get_base_metadata_no_passive=metadata
    mdp._exporter_utils.get_base_metadata=metadata
    # Some mjlab releases expose metadata only through exporter_utils.
    try:
        from mjlab.tasks.velocity.rl import exporter
    except ImportError:
        exporter=None
    if exporter is not None and hasattr(exporter,'get_base_metadata'):
        exporter.get_base_metadata=metadata
    # mjlab 1.3.0 exports from VelocityOnPolicyRunner.save(), whose module
    # captured get_base_metadata before task registration. Updating only
    # exporter_utils leaves that reference returning all 15 V2 joints.
    from mjlab.tasks.velocity.rl import runner as velocity_runner
    original_runner_metadata=velocity_runner.get_base_metadata
    def runner_metadata(env,run_path):
        if _is_v2(env.scene['robot']):return metadata(env,run_path)
        return original_runner_metadata(env,run_path)
    velocity_runner.get_base_metadata=runner_metadata
    mdp._microduck_v2_mapping=True

def _policy_selectors(value,seen=None):
    """Narrow robot joint selectors throughout the copied task configuration."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    if seen is None:seen=set()
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,SceneEntityCfg) and value.name=='robot':
        patterns=value.joint_names
        if patterns is None:patterns=('.*',)
        if isinstance(patterns,str):patterns=(patterns,)
        selected=tuple(n for n in POLICY_JOINTS if any(re.fullmatch(p,n) for p in patterns))
        if selected:
            value.joint_names=selected;value.preserve_order=True
        return
    if is_dataclass(value):
        for f in fields(value):_policy_selectors(getattr(value,f.name),seen)
    elif isinstance(value,dict):
        for v in value.values():_policy_selectors(v,seen)
    elif isinstance(value,(tuple,list)):
        for v in value:_policy_selectors(v,seen)

def set_mouth_target(env,target_rad):
    """External control channel; the policy action vector remains length 14."""
    target=torch.as_tensor(target_rad,dtype=torch.float32,device=env.device)
    if not torch.isfinite(target).all():raise ValueError('Mouth target must be finite')
    if target.numel() not in (1,env.num_envs):raise ValueError('Expected scalar or one mouth target per environment')
    robot=env.scene['robot'];ids,_=robot.find_joints('^mouth$')
    limits=robot.data.joint_pos_limits[:,ids[0],:]
    target=target.reshape(-1,1).expand(env.num_envs,1)
    env.unwrapped._v2_mouth_target=target.clamp(limits[:,0:1],limits[:,1:2])
    if getattr(env.unwrapped,'_v2_joint_feedback',None) is None:
        hold_mouth(env,None)

def hold_mouth(env,env_ids):
    robot=env.scene['robot'];ids,_=robot.find_joints('^mouth$')
    target=getattr(env.unwrapped,'_v2_mouth_target',0.)
    target=torch.as_tensor(target,device=env.device).expand(env.num_envs,1)
    robot.set_joint_position_target(target,joint_ids=ids)

def reset_mouth(env,env_ids):
    robot=env.scene['robot'];ids,_=robot.find_joints('^mouth$')
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    zero=torch.zeros((len(env_ids),1),device=env.device)
    robot.write_joint_state_to_sim(zero,zero,joint_ids=ids,env_ids=env_ids)

def body_accelerometer_direction(env):
    """Optional accelerometer observation, resolved by sensor name in body axes.

    Matches the stock normalized negative-specific-force convention. The V2
    sensor ordering differs from stock, so a numeric sensor index is invalid.
    This is not an attitude estimator; the default policy still uses gravity.
    """
    specific_force=env.scene['robot/imu_accel'].data
    negative=-specific_force
    norm=torch.linalg.vector_norm(negative,dim=-1,keepdim=True)
    return torch.where(norm>.1,negative/norm.clamp_min(.1),
                       env.scene['robot'].data.projected_gravity_b)

def make_v2_env_cfg(play=False):
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityCfg,EntityArticulationInfoCfg
    from mjlab.managers import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
    install_policy_mapping()
    cfg=deepcopy(make_microduck_velocity_env_cfg(play=play,rough=False))
    stock_robot=cfg.scene.entities['robot']
    reset_z=cfg.events['reset_base'].params['pose_range']['z']
    model=get_spec().compile()
    collisions=deepcopy(stock_robot.collisions)
    for collision in collisions:
        # Upstream self proxies are unnamed: they retain masks 2/2 and condim=3.
        # V2 names them, so explicitly preserve these effective settings rather
        # than applying the generic named '*_collision' condim=1 rule to them.
        names=[model.geom(i).name for i in range(model.ngeom)
               if any(re.fullmatch(p,model.geom(i).name) for p in collision.geom_names_expr)]
        # resolve_field uses re.match, so anchor the end as well: the foot
        # name is a prefix of disabled left_foot_collision1_collision proxies.
        collision.contype={'^'+re.escape(n)+'$':int(model.geom(n).contype[0]) for n in names}
        collision.conaffinity={'^'+re.escape(n)+'$':int(model.geom(n).conaffinity[0]) for n in names}
        collision.condim={'^'+re.escape(n)+'$':int(model.geom(n).condim[0]) for n in names}
    cfg.scene.entities['robot']=EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,HOME_HEIGHT),joint_pos={'.*':0.},joint_vel={'.*':0.}),
        collisions=collisions,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(POLICY_PATTERN,)),
                       XmlActuatorCfg(target_names_expr=('^mouth$',))),
            soft_joint_pos_limit_factor=stock_robot.articulation.soft_joint_pos_limit_factor),
    )
    cfg.actions['joint_pos'].actuator_names=POLICY_JOINTS
    cfg.actions['joint_pos'].preserve_order=True
    _policy_selectors(cfg)
    cfg.rewards['dof_pos_limits'].params['asset_cfg']=SceneEntityCfg(
        'robot',joint_names=POLICY_JOINTS,preserve_order=True)
    # These hooks are specifically for calibrated BAM, which is intentionally
    # deferred. Other mass/CoM/IMU randomizations from the walking recipe stay.
    for event in ('expand_bam_friction_fields','randomize_joint_friction','randomize_motor_gains'):
        cfg.events.pop(event,None)
    # Keep the official head-CoM selector, including its documented legacy
    # bearing_roll (right hip yaw) entry. Its (bottom_head_shell|jaw_soft)
    # alternation selects both bodies on V2 because the jaw is independent.
    for group in cfg.observations.values():
        for term in group.terms.values():
            if term.func is mdp.raw_accelerometer:
                term.func=body_accelerometer_direction;term.params={}
    cfg.events['mouth_hold']=EventTermCfg(func=hold_mouth,mode='step')
    cfg.events['mouth_reset']=EventTermCfg(func=reset_mouth,mode='reset')
    # reset_root_state_uniform ADDS this offset to init_state.pos.
    # Translate the stock spawn-height interval to the V2 CAD HOME height,
    # retaining its width. x/y/yaw and optional tilt DR remain inherited.
    cfg.events['reset_base'].params['pose_range']['z']=(0.,reset_z[1]-reset_z[0])
    # The original XL330 head curriculum exceeds V2's negative neck limit.
    head=('neck_pitch','head_pitch','head_yaw','head_roll')
    for stage in cfg.curriculum['head_pose_range'].params['range_stages']:
        stage['ranges']=tuple((max(lo,float(model.joint(n).range[0])*.8),
                               min(hi,float(model.joint(n).range[1])*.8))
                              for n,(lo,hi) in zip(head,stage['ranges']))
    # The delivered XML matches the official 5 ms physics / 4 substep recipe.
    # Keep the official 50 Hz policy period if timestep is edited for experiments.
    # Other solver/capacity options remain inherited from the official flat task.
    policy_dt=cfg.decimation*cfg.sim.mujoco.timestep
    cfg.sim.mujoco.timestep=float(model.opt.timestep)
    cfg.decimation=round(policy_dt/cfg.sim.mujoco.timestep)
    if abs(cfg.decimation*cfg.sim.mujoco.timestep-policy_dt)>1e-8:
        raise ValueError('Model timestep must divide the 50 Hz policy period')
    return cfg

def make_v2_bam_env_cfg(play=False):
    """Mechanics/actuator diagnostic with ideal joint feedback, never S0/O0 training.

    The formal sensor task builds on this configuration and installs its measured
    feedback pipeline. Body, contact, IMU, command and PPO settings stay upstream.
    """
    from mjlab.managers import EventTermCfg
    from mjlab_microduck.actuator.sc0090_v5 import SC0090V5ActuatorCfg
    from mjlab_microduck.sc0090_v2.contract import load_contract
    from mjlab_microduck.tasks import mdp
    contract=load_contract(ASSETS/'training.json')
    dynamics=contract['dynamics']
    cfg=make_v2_env_cfg(play)
    if abs(float(get_spec().compile().body_mass.sum())-contract['mechanics']['nominal_mass_kg'])>1e-8:
        raise ValueError('V2 CAD mass differs from the installed training contract')
    kwargs=dict(kp_fw=dynamics['firmware_p'],kd_fw=dynamics['firmware_d'],
                vin=dynamics['nominal_voltage_v'])
    cfg.scene.entities['robot'].articulation.actuators=(
        SC0090V5ActuatorCfg(target_names_expr=(POLICY_PATTERN,),**kwargs),
        SC0090V5ActuatorCfg(target_names_expr=('^mouth$',),**kwargs),
    )
    cfg.events['expand_bam_friction_fields']=EventTermCfg(
        func=mdp.expand_bam_friction_fields,mode='startup')
    # R1 randomizes measured operating conditions, not the fitted coefficients.
    # These XL330 motor assumptions would modify the fixed v5 fit or count its
    # sensor effects twice. The remaining official events/curricula are retained.
    for event in ('randomize_joint_friction','randomize_joint_damping',
                  'randomize_armature','randomize_motor_gains','encoder_bias'):
        cfg.events.pop(event,None)
    for group in cfg.observations.values():
        for name in ('joint_pos','joint_vel'):
            term=group.terms[name]
            term.noise=None
            term.delay_min_lag=0;term.delay_max_lag=0
            if 'biased' in term.params:term.params['biased']=False
    return cfg

def make_v2_sensor_env_cfg(play=False,stage='nominal'):
    """Formal V2 training: S0/O0 from reset, staged measured R1 conditions."""
    from mjlab.managers import TerminationTermCfg
    from mjlab_microduck.sc0090_v2.action import V2JointPositionActionCfg
    from mjlab_microduck.sc0090_v2.contract import load_contract
    from mjlab_microduck.tasks import mdp
    contract=load_contract(ASSETS/'training.json')
    cfg=make_v2_bam_env_cfg(play)
    action=cfg.actions['joint_pos']
    cfg.actions['joint_pos']=V2JointPositionActionCfg(
        entity_name='robot',actuator_names=POLICY_JOINTS,preserve_order=True,
        scale=action.scale,use_default_offset=True,
        archive_path=str(ASSETS.parent/'training-assets'),randomization_stage=stage,
        command_rate_rad_s=contract['commands']['command_rate_rad_s'])
    cfg.events.pop('mouth_hold',None)  # The same native command path owns all 15 targets.
    for name,function in (('joint_pos',mdp.v2_observed_joint_pos),
                          ('joint_vel',mdp.v2_observed_joint_vel)):
        term=cfg.observations['actor'].terms[name]
        term.func=function;term.params={}
    cfg.terminations['sc0090_feedback_fault']=TerminationTermCfg(
        func=mdp.v2_feedback_fault,time_out=False)
    return cfg

def register():
    from mjlab.tasks.registry import list_tasks,register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    for task,factory,name in ((TASK_ID,make_v2_env_cfg,'microduck_v2_pd'),
                              (BAM_DEBUG_TASK_ID,make_v2_bam_env_cfg,'microduck_v2_bam_ideal_debug'),
                              (SENSOR_TASK_ID,make_v2_sensor_env_cfg,'microduck_v2_s0_o0_r1'),
                              (SENSOR_TASK_ID+'-Measured',
                               lambda play=False:make_v2_sensor_env_cfg(play,'measured'),
                               'microduck_v2_s0_o0_r1_measured')):
        if task in list_tasks():continue
        rl=deepcopy(MicroduckRlCfg)
        rl.experiment_name=name;rl.run_name=name
        register_mjlab_task(task_id=task,env_cfg=factory(),play_env_cfg=factory(True),
                           rl_cfg=rl,runner_cls=MicroduckOnPolicyRunner)
