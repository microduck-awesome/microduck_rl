"""From-random P04 discovery; temporary valid-raw frontend is explicit.

The final anomaly/tolerance frontend is developed separately. This lineage never
loads a baseline expert. Head precision/polish is enabled only after recovery.
"""
from copy import deepcopy
from .feedback_repair import make_repaired_env_cfg

TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-Repaired-S0-O0-R1'
ROLL_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-RollCurriculum-Repaired-S0-O0-R1'
OVERLAP_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-OverlapRoll-Repaired-S0-O0-R1'
OVERLAP_FRONTIERS=(150.,120.,105.,90.,75.,60.,45.,30.,15.,0.)
DIRECTION_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-DirectionGate-Repaired-S0-O0-R1'
DIRECTION_FRONTIERS=(150.,135.,120.,105.,90.,75.,60.,45.,30.,15.,0.)
M4_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-DirectionGate-M4-Repaired-S0-O0-R1'
PHASE_ACCURACY_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-TimeoutPhase-AccuracyBonus05-Repaired-S0-O0-R1'
TIMEOUT_PHASE_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-TimeoutPhase-Repaired-S0-O0-R1'
HEAD_CAP005_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap005-MicroExposure025-Repaired-S0-O0-R1'
MICRO_EXPOSURE_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-MicroExposure025-Repaired-S0-O0-R1'
HEAD_STD_CAP_LR_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-WarmLR1e5-Repaired-S0-O0-R1'
HEAD_STD_CAP_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-HeadStdCap001-Repaired-S0-O0-R1'
LATE_ENTROPY_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fresh-M4-LateEntropy005-Repaired-S0-O0-R1'


def make_fresh_env_cfg(play=False,backroll_before_sit=False,overlap=False,directional=False):
    cfg=make_repaired_env_cfg(play=play)
    cfg.events['set_ground_state'].params['discovery_curriculum']=True
    if backroll_before_sit:
        cfg.events['set_ground_state'].params['backroll_before_sit']=True
    if overlap:
        assert backroll_before_sit
        # Preserve mastered stages 0/1 and checkpoint stage numbering. From
        # stage 2 onward keep 50% of each 30-degree window in the next window.
        cfg.events['set_ground_state'].params['roll_frontiers']=OVERLAP_FRONTIERS
    if directional:
        assert overlap
        # At restored stage 2 consolidate 120-150deg: the previous aggregate
        # 73.8% score hid a 52.3% positive-direction score. Both directions must
        # independently pass on both seeds before this window may advance.
        cfg.events['set_ground_state'].params.update(
            roll_frontiers=DIRECTION_FRONTIERS,frontier_direction_gate=True)
    return cfg


def register_fresh_task():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_standup_env_cfg import MicroduckStandUpRlCfg
    agent=deepcopy(MicroduckStandUpRlCfg)
    agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
    # 32768*24/16 = 49152 samples/minibatch, matching official 8192*24/4.
    # Keep T=24 and five epochs; validate the actual end-to-end throughput.
    agent.algorithm.num_mini_batches=16
    agent.experiment_name='microduck_v2_p04_from_random'
    register_mjlab_task(task_id=TASK,env_cfg=make_fresh_env_cfg(),
        play_env_cfg=make_fresh_env_cfg(True),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
    register_mjlab_task(task_id=ROLL_TASK,env_cfg=make_fresh_env_cfg(backroll_before_sit=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True),rl_cfg=deepcopy(agent),runner_cls=MicroduckOnPolicyRunner)
    register_mjlab_task(task_id=OVERLAP_TASK,
        env_cfg=make_fresh_env_cfg(backroll_before_sit=True,overlap=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True,overlap=True),
        rl_cfg=deepcopy(agent),runner_cls=MicroduckOnPolicyRunner)
    register_mjlab_task(task_id=DIRECTION_TASK,
        env_cfg=make_fresh_env_cfg(backroll_before_sit=True,overlap=True,directional=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True,overlap=True,directional=True),
        rl_cfg=deepcopy(agent),runner_cls=MicroduckOnPolicyRunner)
    m4=deepcopy(agent)
    # Bounded ablation: same N/T/epochs, four larger minibatches instead of
    # sixteen. This changes 80 Adam/LR-schedule steps to 20 per rollout.
    m4.algorithm.num_mini_batches=4
    register_mjlab_task(task_id=M4_TASK,
        env_cfg=make_fresh_env_cfg(backroll_before_sit=True,overlap=True,directional=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True,overlap=True,directional=True),
        rl_cfg=m4,runner_cls=MicroduckOnPolicyRunner)

    # Explicit mature-actor controlled continuation; no sigma or Adam reset.
    late_entropy=deepcopy(m4)
    late_entropy.algorithm.entropy_coef=.005
    register_mjlab_task(task_id=LATE_ENTROPY_TASK,
        env_cfg=make_fresh_env_cfg(backroll_before_sit=True,overlap=True,directional=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True,overlap=True,directional=True),
        rl_cfg=late_entropy,runner_cls=MicroduckOnPolicyRunner)

    head_cap=deepcopy(late_entropy)
    head_cap.actor.distribution_cfg={**head_cap.actor.distribution_cfg,
        'class_name':'mjlab_microduck.custom_v2.head_exploration:HeadCappedGaussianDistribution',
        'head_std_cap':.01}
    register_mjlab_task(task_id=HEAD_STD_CAP_TASK,
        env_cfg=make_fresh_env_cfg(backroll_before_sit=True,overlap=True,directional=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True,overlap=True,directional=True),
        rl_cfg=head_cap,runner_cls=MicroduckOnPolicyRunner)

    head_cap_lr=deepcopy(head_cap)
    head_cap_lr.algorithm.learning_rate=1e-5
    register_mjlab_task(task_id=HEAD_STD_CAP_LR_TASK,
        env_cfg=make_fresh_env_cfg(backroll_before_sit=True,overlap=True,directional=True),
        play_env_cfg=make_fresh_env_cfg(True,backroll_before_sit=True,overlap=True,directional=True),
        rl_cfg=head_cap_lr,runner_cls=MicroduckOnPolicyRunner)

    register_mjlab_task(task_id=MICRO_EXPOSURE_TASK,
        env_cfg=make_micro_exposure_env_cfg(),play_env_cfg=make_micro_exposure_env_cfg(True),
        rl_cfg=deepcopy(head_cap_lr),runner_cls=MicroduckOnPolicyRunner)

    cap005=deepcopy(head_cap_lr)
    cap005.actor.distribution_cfg['head_std_cap']=.005
    register_mjlab_task(task_id=HEAD_CAP005_TASK,
        env_cfg=make_micro_exposure_env_cfg(),play_env_cfg=make_micro_exposure_env_cfg(True),
        rl_cfg=cap005,runner_cls=MicroduckOnPolicyRunner)

    from mjlab.managers import EventTermCfg
    from .timeout_phase import phased_time_out,reset_timeout_deadlines
    phased=make_micro_exposure_env_cfg()
    phased.terminations['time_out'].func=phased_time_out
    phased.events['first_timeout_deadline_reset']=EventTermCfg(func=reset_timeout_deadlines,mode='reset')
    register_mjlab_task(task_id=TIMEOUT_PHASE_TASK,
        env_cfg=phased,play_env_cfg=make_micro_exposure_env_cfg(True),
        rl_cfg=deepcopy(cap005),runner_cls=MicroduckOnPolicyRunner)

    from mjlab.managers import RewardTermCfg
    from .head_accuracy import physical_head_command_accuracy
    precise_train=deepcopy(phased)
    precise_play=make_micro_exposure_env_cfg(True)
    for precise in (precise_train,precise_play):
        precise.rewards['head_command_accuracy']=RewardTermCfg(func=physical_head_command_accuracy,
            weight=.5,params=dict(relative_band=.1,minimum_band=.002))
    register_mjlab_task(task_id=PHASE_ACCURACY_TASK,
        env_cfg=precise_train,play_env_cfg=precise_play,
        rl_cfg=deepcopy(cap005),runner_cls=MicroduckOnPolicyRunner)


def make_micro_exposure_env_cfg(play=False):
    cfg=make_fresh_env_cfg(play,backroll_before_sit=True,overlap=True,directional=True)
    cfg.commands['head_pose'].precision_probability=.25
    return cfg
