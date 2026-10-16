"""Finite mature GroundPick precision stage; original physical acceptance unchanged."""
from copy import deepcopy
from .ground_pick_history import make_history_ground_pick_env_cfg,HISTORY_TASK
PRECISION_TASK='Mjlab-GroundPick-Flat-MicroDuck-V2-TargetPrecision-Repaired-S0-O0-R1'

def make_precision_ground_pick_env_cfg(play=False,stage='nominal'):
    cfg=make_history_ground_pick_env_cfg(play=play,stage=stage)
    term=cfg.rewards['mouth_ground_proximity']
    assert term.params['std']==.10 and term.params['target_height']==0.
    term.params['std']=.025
    term.params['target_height']=.020
    return cfg

def register_precision_ground_pick_tasks():
    from mjlab.tasks.registry import register_mjlab_task,load_rl_cfg
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    for stage in ('nominal','measured'):
        suffix='-Measured' if stage=='measured' else ''
        agent=deepcopy(load_rl_cfg(HISTORY_TASK+suffix))
        register_mjlab_task(task_id=PRECISION_TASK+suffix,
            env_cfg=make_precision_ground_pick_env_cfg(stage=stage),
            play_env_cfg=make_precision_ground_pick_env_cfg(play=True,stage=stage),
            rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
