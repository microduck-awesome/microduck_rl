"""Microduck V2 CAD task: 14 policy actions plus an independent mouth.

Targets mjlab 1.3.0 and microduck_rl b02fdea4902c2b93b7d34a83a6f05da4ececb7d3.
Official observation noise/delay and PPO are retained with frozen v5 SC0090 dynamics.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import re
import mujoco
import torch

# Keep mjlab/task imports inside functions. Its plugin discovery imports this
# module from tasks/__init__.py; eagerly importing tasks here also made a direct
# import re-enter the stock registrations before this module defined register().

POLICY_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
POLICY_PATTERN = "^(" + "|".join(POLICY_JOINTS) + ")$"
ASSETS = Path(__file__).resolve().parent / "assets"
META = json.loads((ASSETS / "model_metadata.json").read_text(encoding="utf-8"))
HOME_HEIGHT = META["home_height_m"]
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck-V2-Official-Baseline"
RUNTIME = "microduck-v2-sc0090-official-baseline-v1"
NOMINAL_MASS_KG = 0.600941825


def get_spec():
    return mujoco.MjSpec.from_file(str(ASSETS / "robot_walk.xml"))


def _is_v2(asset):
    names = getattr(asset, "joint_names", ())
    return "mouth" in names and all(n in names for n in POLICY_JOINTS)


def install_policy_mapping():
    from mjlab_microduck.tasks import mdp

    # Preserve stock tasks. Only a robot with this full named layout uses the
    # new mapping; an actuated mouth is never mislabeled as passive_*.
    if getattr(mdp, "_microduck_v2_mapping", False):
        return
    original_servo = mdp._servo_joint_ids

    def servo_ids(env, asset):
        cached = getattr(asset, '_v2_policy_servo_ids', None)
        if cached is not None:
            return cached
        if _is_v2(asset):
            ids, names = asset.find_joints(POLICY_JOINTS, preserve_order=True)
            if tuple(names) != POLICY_JOINTS:
                raise ValueError("V2 policy joint order mismatch")
            asset._v2_policy_servo_ids = ids
            return ids
        return original_servo(env, asset)

    mdp._servo_joint_ids = servo_ids
    from mjlab_microduck.actuator.sc0090_v5 import MODEL_SHA256

    original_metadata = mdp._get_base_metadata_no_passive

    def metadata(env, run_path):
        result = original_metadata(env, run_path)
        if _is_v2(env.scene["robot"]):
            ids = [result["joint_names"].index(n) for n in POLICY_JOINTS]
            for field in (
                "joint_names",
                "joint_stiffness",
                "joint_damping",
                "default_joint_pos",
            ):
                result[field] = [result[field][i] for i in ids]
            result["microduck_robot"] = "microduck-v2-sc0090"
            result["microduck_runtime"] = RUNTIME
            result["microduck_feedback"] = "official-joint-noise-delay-and-encoder-bias"
            result["microduck_motor_sha256"] = MODEL_SHA256
            result["microduck_motor_operating_point"] = (
                "P15/D30/I0;11.1V;goal_profile_zeroed"
            )
            result["microduck_deployment"] = "simulation-baseline-only"
            if hasattr(env.unwrapped, '_v2_sensor_proxy'):
                result['microduck_feedback'] = 'empirical-paired-qv-block-randomization;not-full-S0-O0'
                result['microduck_error_bank_sha256'] = env.unwrapped._v2_sensor_proxy['bank_sha256']
                result['microduck_motor_operating_point'] = 'P15/D30/I0;vin-random-10.2,11.1,12.4V;goal_profile_zeroed'
        return result

    mdp._get_base_metadata_no_passive = metadata
    mdp._exporter_utils.get_base_metadata = metadata
    # Some mjlab releases expose metadata only through exporter_utils.
    try:
        from mjlab.tasks.velocity.rl import exporter
    except ImportError:
        exporter = None
    if exporter is not None and hasattr(exporter, "get_base_metadata"):
        exporter.get_base_metadata = metadata
    # mjlab 1.3.0 exports from VelocityOnPolicyRunner.save(), whose module
    # captured get_base_metadata before task registration. Updating only
    # exporter_utils leaves that reference returning all 15 V2 joints.
    from mjlab.tasks.velocity.rl import runner as velocity_runner

    original_runner_metadata = velocity_runner.get_base_metadata

    def runner_metadata(env, run_path):
        if _is_v2(env.scene["robot"]):
            return metadata(env, run_path)
        return original_runner_metadata(env, run_path)

    velocity_runner.get_base_metadata = runner_metadata
    mdp._microduck_v2_mapping = True


def _policy_selectors(value, seen=None):
    """Narrow robot joint selectors throughout the copied task configuration."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, SceneEntityCfg) and value.name == "robot":
        patterns = value.joint_names
        if patterns is None:
            patterns = (".*",)
        if isinstance(patterns, str):
            patterns = (patterns,)
        selected = tuple(
            n for n in POLICY_JOINTS if any(re.fullmatch(p, n) for p in patterns)
        )
        if selected:
            value.joint_names = selected
            value.preserve_order = True
        return
    if is_dataclass(value):
        for f in fields(value):
            _policy_selectors(getattr(value, f.name), seen)
    elif isinstance(value, dict):
        for v in value.values():
            _policy_selectors(v, seen)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _policy_selectors(v, seen)


def set_mouth_target(env, target_rad):
    """External control channel; the policy action vector remains length 14."""
    target = torch.as_tensor(target_rad, dtype=torch.float32, device=env.device)
    if not torch.isfinite(target).all():
        raise ValueError("Mouth target must be finite")
    if target.numel() not in (1, env.num_envs):
        raise ValueError("Expected scalar or one mouth target per environment")
    robot = env.scene["robot"]
    ids, _ = robot.find_joints("^mouth$")
    limits = robot.data.joint_pos_limits[:, ids[0], :]
    target = target.reshape(-1, 1).expand(env.num_envs, 1)
    env.unwrapped._v2_mouth_target = target.clamp(limits[:, 0:1], limits[:, 1:2])
    hold_mouth(env, None)


def hold_mouth(env, env_ids):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints("^mouth$")
    target = getattr(env.unwrapped, "_v2_mouth_target", 0.0)
    target = torch.as_tensor(target, device=env.device).expand(env.num_envs, 1)
    robot.set_joint_position_target(target, joint_ids=ids)


def reset_mouth(env, env_ids):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints("^mouth$")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    zero = torch.zeros((len(env_ids), 1), device=env.device)
    robot.write_joint_state_to_sim(zero, zero, joint_ids=ids, env_ids=env_ids)


def body_accelerometer_direction(env):
    """Optional accelerometer observation, resolved by sensor name in body axes.

    Matches the stock normalized negative-specific-force convention. The V2
    sensor ordering differs from stock, so a numeric sensor index is invalid.
    This is not an attitude estimator; the default policy still uses gravity.
    """
    specific_force = env.scene["robot/imu_accel"].data
    negative = -specific_force
    norm = torch.linalg.vector_norm(negative, dim=-1, keepdim=True)
    return torch.where(
        norm > 0.1,
        negative / norm.clamp_min(0.1),
        env.scene["robot"].data.projected_gravity_b,
    )


def _make_v2_geometry_env_cfg(play=False):
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from mjlab.managers import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    install_policy_mapping()
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play, rough=False))
    stock_robot = cfg.scene.entities["robot"]
    reset_z = cfg.events["reset_base"].params["pose_range"]["z"]
    model = get_spec().compile()
    collisions = deepcopy(stock_robot.collisions)
    for collision in collisions:
        # Upstream self proxies are unnamed: they retain masks 2/2 and condim=3.
        # V2 names them, so explicitly preserve these effective settings rather
        # than applying the generic named '*_collision' condim=1 rule to them.
        names = [
            model.geom(i).name
            for i in range(model.ngeom)
            if any(
                re.fullmatch(p, model.geom(i).name) for p in collision.geom_names_expr
            )
        ]
        # resolve_field uses re.match, so anchor the end as well: the foot
        # name is a prefix of disabled left_foot_collision1_collision proxies.
        collision.contype = {
            "^" + re.escape(n) + "$": int(model.geom(n).contype[0]) for n in names
        }
        collision.conaffinity = {
            "^" + re.escape(n) + "$": int(model.geom(n).conaffinity[0]) for n in names
        }
        collision.condim = {
            "^" + re.escape(n) + "$": int(model.geom(n).condim[0]) for n in names
        }
    cfg.scene.entities["robot"] = EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, HOME_HEIGHT), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        collisions=collisions,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=(POLICY_PATTERN,)),
                XmlActuatorCfg(target_names_expr=("^mouth$",)),
            ),
            soft_joint_pos_limit_factor=stock_robot.articulation.soft_joint_pos_limit_factor,
        ),
    )
    cfg.actions["joint_pos"].actuator_names = POLICY_JOINTS
    cfg.actions["joint_pos"].preserve_order = True
    _policy_selectors(cfg)
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=POLICY_JOINTS, preserve_order=True
    )
    # The geometry helper uses preview XML actuators; the baseline below replaces
    # them with frozen M6. Body/contact/IMU randomizations stay upstream.
    for event in (
        "expand_bam_friction_fields",
        "randomize_joint_friction",
        "randomize_motor_gains",
    ):
        cfg.events.pop(event, None)
    # Keep the official head-CoM selector, including its documented legacy
    # bearing_roll (right hip yaw) entry. Its (bottom_head_shell|jaw_soft)
    # alternation selects both bodies on V2 because the jaw is independent.
    for group in cfg.observations.values():
        for term in group.terms.values():
            if term.func is mdp.raw_accelerometer:
                term.func = body_accelerometer_direction
                term.params = {}
    cfg.events["mouth_hold"] = EventTermCfg(func=hold_mouth, mode="step")
    cfg.events["mouth_reset"] = EventTermCfg(func=reset_mouth, mode="reset")
    # reset_root_state_uniform ADDS this offset to init_state.pos.
    # Translate the stock spawn-height interval to the V2 CAD HOME height,
    # retaining its width. x/y/yaw and optional tilt DR remain inherited.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, reset_z[1] - reset_z[0])
    # The original XL330 head curriculum exceeds V2's negative neck limit.
    head = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    for stage in cfg.curriculum["head_pose_range"].params["range_stages"]:
        stage["ranges"] = tuple(
            (
                max(lo, float(model.joint(n).range[0]) * 0.8),
                min(hi, float(model.joint(n).range[1]) * 0.8),
            )
            for n, (lo, hi) in zip(head, stage["ranges"])
        )
    # The delivered XML matches the official 5 ms physics / 4 substep recipe.
    # Keep the official 50 Hz policy period if timestep is edited for experiments.
    # Other solver/capacity options remain inherited from the official flat task.
    policy_dt = cfg.decimation * cfg.sim.mujoco.timestep
    cfg.sim.mujoco.timestep = float(model.opt.timestep)
    cfg.decimation = round(policy_dt / cfg.sim.mujoco.timestep)
    if abs(cfg.decimation * cfg.sim.mujoco.timestep - policy_dt) > 1e-8:
        raise ValueError("Model timestep must divide the 50 Hz policy period")
    return cfg


def make_v2_baseline_env_cfg(play=False):
    """V2 mechanics and frozen M6 with the official observation/training recipe.

    Preserve the upstream noisy, delayed and encoder-biased joint observations.
    Commands remain continuous physical joint targets through JointPositionAction.
    No measured feedback stream, native-count target mapper or residual archive is used.
    """
    from mjlab.managers import EventTermCfg
    from mjlab_microduck.actuator.sc0090_v5 import SC0090V5ActuatorCfg
    from mjlab_microduck.tasks import mdp

    cfg = _make_v2_geometry_env_cfg(play)
    if abs(float(get_spec().compile().body_mass.sum()) - NOMINAL_MASS_KG) > 1e-8:
        raise ValueError("V2 CAD mass differs from the baseline provenance")
    kwargs = dict(kp_fw=15.0, kd_fw=30.0, vin=11.1)
    cfg.scene.entities["robot"].articulation.actuators = (
        SC0090V5ActuatorCfg(target_names_expr=(POLICY_PATTERN,), **kwargs),
        SC0090V5ActuatorCfg(target_names_expr=("^mouth$",), **kwargs),
    )
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=mdp.expand_bam_friction_fields, mode="startup"
    )
    # Freeze the identified actuator coefficients and operating point. These stock
    # XL330 perturbations would change that model, unlike generic observation DR.
    for event in (
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_armature",
        "randomize_motor_gains",
    ):
        cfg.events.pop(event, None)
    from .runtime import configure_runtime
    return configure_runtime(cfg)


def register():
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg

    if TASK_ID in list_tasks():
        return
    rl = deepcopy(MicroduckRlCfg)
    rl.experiment_name = "microduck_v2_official_baseline"
    rl.run_name = "microduck_v2_official_baseline"
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_v2_baseline_env_cfg(),
        play_env_cfg=make_v2_baseline_env_cfg(True),
        rl_cfg=rl,
        runner_cls=MicroduckOnPolicyRunner,
    )
