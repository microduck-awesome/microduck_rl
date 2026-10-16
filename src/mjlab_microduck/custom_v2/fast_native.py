"""Fast skill pretraining with the P04 wire commands and official sensor DR.

This is a separately named approximation, never the full S0/O0/R1 model.
It preserves quantized/clamped command and motor conventions, while retaining
the official GPU observation noise, delay and episode encoder-bias model.
Only an independently evaluated full-P04 adaptation can acquire its runtime
identity. Calibration-derived sensor residual training is a separate variant.
"""
from copy import deepcopy
from dataclasses import dataclass
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab_microduck.actuator.sc0090_v5 import load_v5_model
from mjlab_microduck.sc0090_v2.action import V2JointPositionAction,V2JointPositionActionCfg
from mjlab_microduck.sc0090_v2.artifacts import TrainingArchive

FAST_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Fast-Native'
FAST_RUNTIME='microduck-v2-fast-native-pretraining-v1'
FIRMWARE_TASK='Mjlab-StandUp-Flat-MicroDuck-V2-Firmware-LS3-S0-R1'
DELAY_TASK=FAST_TASK+'-Delay20'
PROXY_TASK=FAST_TASK+'-ErrorBlocks'


class NativeCommandMapping:
    """Same FP32 arithmetic and synthetic origin as WholeRobotFeedback."""
    def __init__(self,archive,device):
        self.mapping=torch.tensor(archive.observer_mapping,device=device,dtype=torch.float32)
        self.scale=.005624596222247351
        self.zero=512.
        self.count_limits=(20,1001)

    def counts(self,target):
        count=(target-self.mapping[0])/self.mapping[1]/self.scale+self.zero
        return torch.round(count.clamp(*self.count_limits))

    def physical_target(self,counts):
        return self.mapping[0]+self.mapping[1]*(counts-self.zero)*self.scale

    def physical_goal_for_motor(self,counts,feedback_bias,feedback_scale):
        return ((counts-self.zero)*self.scale-feedback_bias)/feedback_scale

    def command(self,counts):
        # This approximation has no independent register/event stream. Its
        # inherited command processor still preserves the native wire target.
        pass


@dataclass(kw_only=True)
class FastNativeActionCfg(V2JointPositionActionCfg):
    def build(self,env):return FastNativeAction(self,env)


class FastNativeAction(V2JointPositionAction):
    def __init__(self,cfg,env):
        if cfg.command_rate_rad_s is not None and not 0<cfg.command_rate_rad_s<=20:
            raise ValueError('Invalid native command slew')
        JointPositionAction.__init__(self,cfg,env)
        if self.action_dim!=14:raise ValueError('Expected 14 policy actions')
        archive=TrainingArchive.load(cfg.archive_path)
        self.feedback=NativeCommandMapping(archive,self.device)
        mouth,_=self._entity.find_joints('^mouth$')
        self._all_ids=torch.cat((self._target_ids,torch.tensor(mouth,device=self.device)))
        model=load_v5_model()
        self._feedback_bias,self._feedback_scale=model.feedback_bias.value,model.feedback_scale.value
        self._goal=torch.zeros(self.num_envs,15,device=self.device)
        self._count=self.feedback.counts(self._goal)
        env.unwrapped._v2_fast_native=True
        self.reset()

    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device)
        if env_ids is not None:ids=ids[env_ids] if isinstance(env_ids,slice) else env_ids
        JointPositionAction.reset(self,ids)
        pose=self._entity.data.joint_pos[:,self._all_ids]
        self._count[ids]=self.feedback.counts(pose)[ids]
        self._goal[ids]=self.feedback.physical_target(self._count)[ids]
        self._apply_targets()

    def apply_actions(self):
        self._apply_targets()


def make_fast_standup_cfg(play=False):
    from .training import make_standup_env_cfg
    from .microduck_v2 import _policy_selectors
    from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg
    cfg=make_standup_env_cfg(play)
    official=make_microduck_standup_env_cfg(play=play,rough=False)
    cfg.actions['joint_pos']=FastNativeActionCfg(**vars(cfg.actions['joint_pos']))
    cfg.terminations.pop('sc0090_feedback_fault')
    for name in ('joint_pos','joint_vel'):
        cfg.observations['actor'].terms[name]=deepcopy(official.observations['actor'].terms[name])
    if 'encoder_bias' in official.events:
        cfg.events['encoder_bias']=deepcopy(official.events['encoder_bias'])
    _policy_selectors(cfg)
    return cfg


def register_fast_task():
    from mjlab.tasks.registry import list_tasks,register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_standup_env_cfg import MicroduckStandUpRlCfg
    if FAST_TASK in list_tasks():return
    agent=deepcopy(MicroduckStandUpRlCfg)
    agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO'
    agent.experiment_name=agent.run_name='microduck_v2_fast_native_standup'
    register_mjlab_task(task_id=FAST_TASK,env_cfg=make_fast_standup_cfg(),
        play_env_cfg=make_fast_standup_cfg(True),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
    def delay_cfg(play=False):
        cfg=make_fast_standup_cfg(play)
        # A single-factor diagnostic: joint velocity already has one 20 ms
        # lag in the official recipe; add that lag to position as well.
        term=cfg.observations['actor'].terms['joint_pos']
        term.delay_min_lag=term.delay_max_lag=1
        term.delay_update_period=0
        return cfg
    delay_agent=deepcopy(agent)
    delay_agent.experiment_name=delay_agent.run_name='microduck_v2_fast_native_delay20'
    register_mjlab_task(task_id=DELAY_TASK,env_cfg=delay_cfg(),play_env_cfg=delay_cfg(True),
        rl_cfg=delay_agent,runner_cls=MicroduckOnPolicyRunner)
    def proxy_cfg(play=False):
        from mjlab.managers import EventTermCfg
        from .microduck_v2 import ASSETS
        from .error_proxy import proxy_joint_pos,proxy_joint_vel,reset_proxy
        cfg=delay_cfg(play)
        bank=str(ASSETS.parent/'training-assets/fast-proxy-v1')
        cfg.events.pop('encoder_bias',None)
        cfg.events['sensor_error_reset']=EventTermCfg(func=reset_proxy,mode='reset',params=dict(bank_path=bank))
        for name,func in (('joint_pos',proxy_joint_pos),('joint_vel',proxy_joint_vel)):
            term=cfg.observations['actor'].terms[name]
            term.func=func;term.params=dict(bank_path=bank,strength=1.);term.noise=None
        return cfg
    proxy_agent=deepcopy(agent)
    proxy_agent.experiment_name=proxy_agent.run_name='microduck_v2_fast_native_error_blocks'
    register_mjlab_task(task_id=PROXY_TASK,env_cfg=proxy_cfg(),play_env_cfg=proxy_cfg(True),
        rl_cfg=proxy_agent,runner_cls=MicroduckOnPolicyRunner)
    from .training import make_standup_env_cfg
    def firmware_cfg(play=False):
        cfg=make_standup_env_cfg(play)
        cfg.actions['joint_pos'].observer_frontend='firmware-ls3'
        return cfg
    agent=deepcopy(agent)
    agent.experiment_name=agent.run_name='microduck_v2_firmware_ls3_standup'
    register_mjlab_task(task_id=FIRMWARE_TASK,env_cfg=firmware_cfg(),
        play_env_cfg=firmware_cfg(True),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
