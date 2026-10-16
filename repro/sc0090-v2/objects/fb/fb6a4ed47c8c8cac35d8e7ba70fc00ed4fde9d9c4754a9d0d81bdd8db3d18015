"""Bounded initial endpoint-exposure lesson; full 12 s evaluation retained."""
from copy import deepcopy
from .sitstand_clear import make_clear_sitstand_env_cfg
LESSON_TASK='Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Lesson4s-Repaired-S0-O0-R1'

def make_lesson_sitstand_env_cfg(play=False,stage='nominal'):
    cfg=make_clear_sitstand_env_cfg(play=play,stage=stage)
    if not play:cfg.episode_length_s=4.
    return cfg

def register_lesson_sitstand_task():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
    for task,stage in ((LESSON_TASK,'nominal'),(LESSON_TASK+'-Measured','measured')):
        agent=deepcopy(MicroduckSitStandRlCfg)
        agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
        agent.experiment_name='microduck_v2_p04_sitstand';agent.run_name='lesson4s_'+stage
        register_mjlab_task(task_id=task,env_cfg=make_lesson_sitstand_env_cfg(stage=stage),
            play_env_cfg=make_lesson_sitstand_env_cfg(play=True,stage=stage),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
