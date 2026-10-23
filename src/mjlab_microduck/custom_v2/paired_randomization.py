"""Explicit approximate baseline DR from repaired-model paired error blocks."""
from pathlib import Path
from mjlab.managers import EventTermCfg
from . import error_proxy

TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'
BANK=Path(__file__).parent/'training-assets/repaired-paired-errors-v2'


def reset_paired(env,env_ids,bank_path):
    import torch
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    state=error_proxy.proxy_state(env,bank_path)
    state.reset(env_ids,env.common_step_counter)
    if not hasattr(state,'robot_voltage'):
        state.robot_voltage=torch.full((env.num_envs,),11.1,device=env.device)
    voltage=state.bank.new_tensor([10.2,11.1,12.4])
    selection=torch.randint(3,(len(env_ids),),generator=state.rng,device=env.device)
    state.robot_voltage[env_ids]=voltage[selection]
    for actuator in env.scene['robot'].actuators:
        actuator.vin_tensor[env_ids,0]=state.robot_voltage[env_ids]


def configure(cfg):
    cfg.events.pop('encoder_bias',None)
    cfg.events['paired_error_reset']=EventTermCfg(func=reset_paired,mode='reset',params={'bank_path':str(BANK)})
    for name,func in [('joint_pos',error_proxy.proxy_joint_pos),('joint_vel',error_proxy.proxy_joint_vel)]:
        term=cfg.observations['actor'].terms[name]
        term.func=func;term.params={'bank_path':str(BANK),'strength':1.}
        term.noise=None;term.delay_min_lag=term.delay_max_lag=1
    return cfg


def register():
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls,register_mjlab_task
    base='Mjlab-StandUp-Flat-MicroDuck-V2-Official-Baseline'
    agent=load_rl_cfg(base);agent.experiment_name='v2_paired_error_baseline';agent.run_name=agent.experiment_name
    register_mjlab_task(task_id=TASK,env_cfg=configure(load_env_cfg(base)),
        play_env_cfg=configure(load_env_cfg(base,play=True)),rl_cfg=agent,runner_cls=load_runner_cls(base))
