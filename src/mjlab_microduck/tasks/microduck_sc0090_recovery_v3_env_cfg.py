"""Recovery with separately evaluated curriculum and measured neck-stop cost."""

from copy import deepcopy
from dataclasses import dataclass, field, fields

from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from . import mdp
from .microduck_sc0090_finetune_env_cfg import (
    SC0090RecoveryV2RlCfg, make_sc0090_recovery_v2_env_cfg,
)


@dataclass
class RecoveryEvaluationCfg:
    interval_iterations: int = 100
    samples_per_group: int = 128
    seeds: tuple[int, ...] = (2026091501, 2026091502)
    timeout_seconds: float = 240.0


@dataclass
class RecoveryEvalRunnerCfg(RslRlOnPolicyRunnerCfg):
    recovery_evaluation: RecoveryEvaluationCfg = field(default_factory=RecoveryEvaluationCfg)


def make_sc0090_recovery_v3_env_cfg(play=False, rough=False):
    cfg = make_sc0090_recovery_v2_env_cfg(play=play, rough=rough)
    # Keep stochastic-rollout success as a diagnostic, but only separately
    # evaluated deterministic policies may advance the reset distribution.
    cfg.curriculum["recovery_success"].params["advance_on_training"] = False
    # This is a physical-position cost, not a command clamp. Targets retain
    # their original range; torque, speed, controller and action ABI are intact.
    cfg.rewards["neck_limit_proximity"] = RewardTermCfg(
        func=mdp.joint_pos_limit_proximity, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("neck_pitch",)),
                "margin": 0.20},
    )
    return cfg


SC0090RecoveryV3RlCfg = RecoveryEvalRunnerCfg(**{
    f.name: deepcopy(getattr(SC0090RecoveryV2RlCfg, f.name))
    for f in fields(SC0090RecoveryV2RlCfg)
})
SC0090RecoveryV3RlCfg.experiment_name = "sc0090_recovery_v3"
SC0090RecoveryV3RlCfg.run_name = "evaluated_curriculum"
