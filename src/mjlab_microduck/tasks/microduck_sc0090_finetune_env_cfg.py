"""Failure-focused SC0090 fine-tuning; seed each task from its own expert.

Keep the original tasks as reproducible baselines. V2 changes the training
distribution and objectives, not motor physics or the runtime observation ABI.
"""

from copy import deepcopy
from dataclasses import fields

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg

from . import mdp
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
from .microduck_standup_env_cfg import make_microduck_standup_env_cfg, MicroduckStandUpRlCfg
from .microduck_velstand_env_cfg import _collapse_curricula_to_final


def _pin_inherited(cfg):
    _collapse_curricula_to_final(cfg)
    # Apply command/event ranges immediately, including play and step zero.
    for term in cfg.curriculum.values():
        p = term.params
        if "range_stages" in p:
            stage = p["range_stages"][-1]
            if "command_name" in p:
                cfg.commands[p["command_name"]].ranges = deepcopy(stage["ranges"])
            elif "event_name" in p:
                r = stage["range"]
                cfg.events[p["event_name"]].params["ranges"] = (-r, r)


def _enable_execution_optimizations(cfg):
    # Base factories reference shared robot constants. Own the entity before
    # changing execution flags so the reference tasks remain reproducible.
    cfg.scene.entities["robot"] = deepcopy(cfg.scene.entities["robot"])
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.fast_friction_writes = True
    cfg.events["cache_reset_constants"] = EventTermCfg(
        func=mdp.sc0090_cache_reset_constants, mode="startup",
    )
    # Flat V2 tasks have no existing spec callback. Preserve any callback on
    # other configurations (e.g. rough terrain contact adjustments).
    if cfg.scene.spec_fn is None:
        cfg.scene.spec_fn = mdp.sc0090_remove_origin_markers


def make_sc0090_walk_v2_env_cfg(play=False, rough=False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    _pin_inherited(cfg)
    _enable_execution_optimizations(cfg)
    cfg.curriculum.pop("standing_envs", None)
    cfg.curriculum.pop("action_rate_weight", None)
    source = cfg.commands["twist"]
    cfg.commands["twist"] = mdp.SC0090VelocityCommandCfg(**{
        f.name: deepcopy(getattr(source, f.name)) for f in fields(source)
    })
    command = cfg.commands["twist"]
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = command.rel_world_envs = command.rel_forward_envs = 0.0
    command.rel_standing_envs = command.rel_turn_in_place_envs = 0.0
    cfg.commands["head_pose"].zero_command_prob = 0.5
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=mdp.sc0090_track_linear_velocity, weight=3.0,
        params={"command_name": "twist", "absolute_std": 0.035, "relative_std": 0.45},
    )
    cfg.rewards["track_angular_velocity"] = RewardTermCfg(
        func=mdp.sc0090_track_yaw_velocity, weight=2.0,
        params={"command_name": "twist", "std": 0.3},
    )
    cfg.rewards["yaw_error"] = RewardTermCfg(
        func=mdp.sc0090_yaw_error, weight=-0.5, params={"command_name": "twist"},
    )
    # Low-speed walking may have longer double support; do not pay as much
    # for keeping a foot airborne as for following the small velocity command.
    cfg.rewards["air_time"].func = mdp.sc0090_feet_air_time
    cfg.rewards["air_time"].params["threshold_min"] = 0.075
    cfg.rewards["air_time"].weight = 1.5
    cfg.rewards["action_rate_l2"].weight = -0.5
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.08, 0.08), "y": (-0.08, 0.08)
        }
    return cfg


def make_sc0090_recovery_v2_env_cfg(play=False, rough=False):
    cfg = make_microduck_standup_env_cfg(play=play, rough=rough)
    _pin_inherited(cfg)
    _enable_execution_optimizations(cfg)
    # Preserve learned head/DR conditions; replace time-driven difficulty and
    # polishing with measured success. Never relax the final standing target.
    cfg.curriculum = {k: v for k, v in cfg.curriculum.items()
                      if k in ("head_pose_range", "com_range", "head_com_range")}
    cfg.episode_length_s = 8.0
    cfg.commands["twist"].rel_forward_envs = 0.0  # upstream would force vx >= .3!
    cfg.commands["head_pose"].zero_command_prob = 0.6
    cfg.commands["body_pose"].ranges = (
        (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
        (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05),
    )
    weights = {
        "action_rate_l2": -0.2, "arrival_damping": 0.0,
        "joint_torque_rate_l2": 0.0, "body_pose_tracking": 0.0,
        "head_pose_bias": 0.0, "height_stand_sharp": 1.0,
        "upright_sharp": 1.5, "standing_composite": 3.75,
        "pose_stand_legs": 1.0, "pose_stand_l1": 0.5,
        "body_ang_vel": -0.02, "angular_momentum": -0.01,
    }
    for key, weight in weights.items():
        cfg.rewards[key].weight = weight
    cfg.rewards.pop("com_upward_velocity")
    cfg.rewards["recovery_progress"] = RewardTermCfg(
        func=mdp.sc0090_recovery_progress, weight=1.0,
    )
    cfg.rewards["stable_standing"] = RewardTermCfg(
        func=mdp.sc0090_stable_standing, weight=1.0,
    )
    cfg.rewards["remaining_fallen"] = RewardTermCfg(
        func=mdp.fallen_state_penalty, weight=-0.5,
        params={"gate_tilt_above_deg": 40.0, "release_tilt_below_deg": 20.0,
                "release_z_above": 0.105},
    )
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05)
        }
    ground = cfg.events["set_ground_state"]
    ground.func = mdp.sc0090_reset_recovery
    ground.params = {k: v for k, v in ground.params.items()
                     if k not in ("face_down_prob", "face_up_prob", "sitting_prob",
                                  "standing_prob", "face_up_roll_max")}
    cfg.curriculum["recovery_success"] = CurriculumTermCfg(
        func=mdp.sc0090_recovery_curriculum,
        params={"min_frontier_trials": 256, "min_stage_steps": 100 * 24,
                "success_threshold": 0.7},
    )
    return cfg


SC0090WalkV2RlCfg = deepcopy(MicroduckRlCfg)
SC0090RecoveryV2RlCfg = deepcopy(MicroduckStandUpRlCfg)
for cfg, name in ((SC0090WalkV2RlCfg, "sc0090_walk_v2"),
                  (SC0090RecoveryV2RlCfg, "sc0090_recovery_v2")):
    cfg.experiment_name = name
    cfg.run_name = "failure_finetune"
    cfg.max_iterations = 6000
    cfg.save_interval = 50
    cfg.algorithm.learning_rate = 2e-4
    cfg.algorithm.desired_kl = 0.005
