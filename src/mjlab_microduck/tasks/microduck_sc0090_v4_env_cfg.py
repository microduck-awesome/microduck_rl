"""One staged recipe for fresh training, legacy continuation and exact V4 resume."""
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Literal

from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from . import mdp
from .microduck_sc0090_finetune_env_cfg import make_sc0090_walk_v2_env_cfg, SC0090WalkV2RlCfg
from .microduck_sc0090_recovery_v3_env_cfg import make_sc0090_recovery_v3_env_cfg, SC0090RecoveryV3RlCfg


@dataclass
class TrainingProgramRunnerCfg:
    checkpoint: str | None = None
    # Scheduling only: at 24 steps/update this is 4800 steps per environment.
    # Chosen to amortize the measured 1–3 minute assessment, not to gate learning.
    # Actual assessments occur at the first checkpoint save after this interval.
    interval_iterations: int = 200
    samples_per_group: int = 128
    seeds: tuple[int, ...] = (2026091501, 2026091502)
    timeout_seconds: float = 300.0
    # Needed only when a legacy checkpoint has neither a task tag nor its
    # original params/agent.yaml sidecar. Never infer task from tensor shapes.
    legacy_task: Literal["auto", "walk", "recovery"] = "auto"


@dataclass
class SC0090ProgramRunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training budget must be explicit; inheriting the V2 expert's 6000-update
    # budget could stop a fresh V4 run before its longer program finished.
    max_iterations: int | None = None
    training_program: TrainingProgramRunnerCfg = field(default_factory=TrainingProgramRunnerCfg)


def make_training_program(task):
    """Explicit stage table; no rule depends on reaching global iteration 6000."""
    if task not in ("walk", "recovery"):
        raise ValueError(task)
    base_push = .08 if task == "walk" else .05
    common = dict(frontier=0, push=base_push, head_scale=1., com_range=.015,
                  head_com_range=.01, body_down=.005, body_up=.005,
                  body_angle=3., body_weight=0., action_rate=-.5,
                  arrival=0., torque_rate=0.)
    phases = []
    for i, (scale, com, head_com) in enumerate(((.05, .003, .003), (.35, .006, .005),
                                                (.65, .01, .007), (1., .015, .01))):
        phases.append(dict(common, name=f"foundation_{i}", kind="foundation",
                           head_scale=scale, com_range=com, head_com_range=head_com,
                           action_rate=(-.2 if task == "recovery" else (-.1, -.2, -.35, -.5)[i])))
    if task == "recovery":
        for frontier in range(1, 6):
            phases.append(dict(common, name=f"roll_{frontier}", kind="roll",
                               frontier=frontier, action_rate=-.2))
        common.update(frontier=5, arrival=-.025, torque_rate=-.0005)
    phases.append(dict(common, name="consolidate", kind="consolidate"))
    for push in (.10, .15, .20, .30):
        phases.append(dict(common, name=f"push_{round(push*100):02d}", kind="push", push=push))
    common["push"] = .30
    # Deliberately modest extension: actual reachable heights must be validated
    # before adding the original recipe's much larger +30 mm command.
    for name, down, up, angle, weight in (("small", .005, .005, 3., 1.),
                                        ("medium", .015, .005, 8., 2.),
                                        ("full", .025, .010, 12., 4.)):
        phases.append(dict(common, name=f"body_{name}", kind="body", body_down=down,
                           body_up=up, body_angle=angle, body_weight=weight))
    common.update({k: phases[-1][k] for k in ("body_down", "body_up", "body_angle", "body_weight")})
    for name, changes in (("action_07", {"action_rate": -.7}),
                          ("action_10", {"action_rate": -1.}),
                          ("torque", {"torque_rate": -.001})):
        common.update(changes)
        phases.append(dict(common, name=f"smooth_{name}", kind="smooth"))
    if task == "recovery":
        common["arrival"] = -.05
        phases.append(dict(common, name="smooth_arrival", kind="smooth"))
    return dict(schema=1, task=task, phases=phases, min_phase_steps=2400,
                retention_threshold=.95, challenge_threshold=.90,
                discovery_threshold=.70, required_passes=2, rehearsal_probability=.25)


def make_sc0090_v4_env_cfg(task, play=False, rough=False):
    cfg = (make_sc0090_walk_v2_env_cfg if task == "walk" else make_sc0090_recovery_v3_env_cfg)(
        play=play, rough=rough)
    cfg.curriculum = {"training_program": CurriculumTermCfg(
        func=mdp.sc0090_program_curriculum, params={"program": make_training_program(task)})}
    cfg.events["program_episode"] = EventTermCfg(func=mdp.sc0090_program_reset, mode="reset")
    if task == "walk":
        velocity = cfg.commands["twist"]
        cfg.commands["twist"] = mdp.SC0090ProgramVelocityCommandCfg(**{
            f.name: deepcopy(getattr(velocity, f.name)) for f in fields(velocity)
        })
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].func = mdp.sc0090_program_push
    body = cfg.commands["body_pose"]
    cfg.commands["body_pose"] = mdp.SC0090ProgramBodyCommandCfg(**{
        f.name: deepcopy(getattr(body, f.name)) for f in fields(body)
    }, recovery=task == "recovery")
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=mdp.body_pose_tracking_locomotion, weight=0.,
        params={"command_name": "body_pose", "nominal_height": .115,
                "z_std": .01, "angle_std": .08726646259971647,
                "axis_weights": (0., 0., 1., 1., 1., 0.), "vel_gate_command_name": None})
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=mdp.sc0090_program_torque_rate, weight=0.)
    if task == "recovery":
        cfg.rewards["stable_standing"].func = mdp.sc0090_program_stable_standing
    return cfg


def make_sc0090_walk_v4_env_cfg(play=False, rough=False):
    return make_sc0090_v4_env_cfg("walk", play, rough)


def make_sc0090_recovery_v4_env_cfg(play=False, rough=False):
    return make_sc0090_v4_env_cfg("recovery", play, rough)


def make_program_rl_cfg(task):
    source = SC0090WalkV2RlCfg if task == "walk" else SC0090RecoveryV3RlCfg
    cfg = SC0090ProgramRunnerCfg(**{f.name: deepcopy(getattr(source, f.name))
                                  for f in fields(RslRlOnPolicyRunnerCfg)})
    cfg.experiment_name = f"sc0090_{task}_v4"
    cfg.run_name = "staged_training"
    cfg.max_iterations = None
    return cfg


SC0090WalkV4RlCfg = make_program_rl_cfg("walk")
SC0090RecoveryV4RlCfg = make_program_rl_cfg("recovery")
