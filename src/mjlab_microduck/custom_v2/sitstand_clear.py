"""SitStand physical target repair; original frozen CAD and reward weights retained."""
from copy import deepcopy
from .sitstand import make_sitstand_env_cfg
CLEAR_TASK='Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Repaired-S0-O0-R1'
SIT_POSE={1: 0.0873, 2: -0.7, 3: -0.8, 4: -0.15, 10: -0.0873, 11: -0.7, 12: -0.8, 13: -0.15}
SIT_HEIGHT=0.075999
POSE_EVIDENCE={'report': '/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/sitstand-clear-pose-batch-002/report.json', 'sha256': '80ab3c0a915351b3f3dca06696617a2e59ad224dc092b7449a38e53cba24ee3e', 'qualified_worlds': 32}

def make_clear_sitstand_env_cfg(play=False,stage='nominal'):
    assert SIT_POSE is not None and SIT_HEIGHT is not None, 'Physical pose qualification required'
    cfg=make_sitstand_env_cfg(play=play,stage=stage)
    ground=cfg.events['set_ground_state'].params
    ground.update(sitting_joint_overrides=SIT_POSE.copy(),sitting_z_min=SIT_HEIGHT,sitting_z_max=SIT_HEIGHT+.015)
    cfg.commands['twist'].sit_z=SIT_HEIGHT
    for section in (cfg.rewards,cfg.curriculum):
        for term in section.values():
            if 'sit_overrides' in term.params:term.params['sit_overrides']=SIT_POSE.copy()
            if 'sit_z' in term.params:term.params['sit_z']=SIT_HEIGHT
    return cfg

def register_clear_sitstand_task():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
    for task,stage in ((CLEAR_TASK,'nominal'),(CLEAR_TASK+'-Measured','measured')):
        agent=deepcopy(MicroduckSitStandRlCfg)
        agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
        agent.experiment_name='microduck_v2_p04_sitstand';agent.run_name='clear_seat_'+stage
        register_mjlab_task(task_id=task,env_cfg=make_clear_sitstand_env_cfg(stage=stage),
            play_env_cfg=make_clear_sitstand_env_cfg(play=True,stage=stage),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
