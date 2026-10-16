"""Explicit execution repair; frozen S0, O0 coefficients and R1 stay unchanged.

An unconstrained predictor residual is not a packet-health observation. The old
8 degree rule rejected valid fast position changes and entered an O1-only long
rewarm period. This variant accepts valid raw measurements and prepares O0 from
five current samples, retaining actual health, numerical and query guards.
It is a simulation candidate, not a qualified hardware runtime.
"""
from copy import deepcopy
from .training import make_standup_env_cfg

TASK = 'Mjlab-StandUp-Flat-MicroDuck-V2-Repaired-S0-O0-R1'
EXECUTION = 'p04-o0-valid-raw-local-history-v1'


def make_repaired_env_cfg(play=False):
    cfg = make_standup_env_cfg(play=play)
    cfg.actions['joint_pos'].spike_rejection = False
    cfg.actions['joint_pos'].o0_local_history = True
    return cfg


def register_repaired_task():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_standup_env_cfg import MicroduckStandUpRlCfg
    agent=deepcopy(MicroduckStandUpRlCfg)
    agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
    agent.experiment_name='microduck_v2_p04_repaired_standup'
    register_mjlab_task(task_id=TASK,env_cfg=make_repaired_env_cfg(),
        play_env_cfg=make_repaired_env_cfg(True),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
