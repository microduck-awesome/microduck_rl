"""Bounded historical GroundPick with phase-consistent physical endpoint resets."""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from mjlab_microduck.tasks import mdp
from .ground_pick import make_ground_pick_env_cfg
from .microduck_v2 import POLICY_JOINTS

HISTORY_TASK='Mjlab-GroundPick-Flat-MicroDuck-V2-HealthHistory-Repaired-S0-O0-R1'
LOW_RESET_PROBABILITY=.25
LOW_PHASE=.375
# Actual physical configuration of the independently validated008 endpoint.
# These are NOT compensated action commands or a joint-space reward target.
LOW_Q=(0.,-.003535603126003419,-.3512309166361186,-1.080117828331044,1.1153121963302182,2.0057699999999707,-.801655058993569,0.,0.,0.,.003535603126003419,-.3512309166361186,-1.080117828331044,1.1153121963302182)
LOW_ROOT_PITCH=.311289129719318
LOW_ROOT_HEIGHT=.07991685881431504
LOW_GEOMETRY_SHA256='cce66e48d1fa7d04041a7a138dd4f113c50d70605d2002310618a26d5293d18f'


def gate_positive_reward(value,clean):
    """Unhealthy samples lose only positive reward; negative orientation stays."""
    return torch.where(clean,value,torch.minimum(value,torch.zeros_like(value)))


def contact_clean(env):
    head=env.scene['head_impact_contact'].data.found.reshape(env.num_envs,-1).any(-1)
    own=env.scene['self_collision'].data.found.reshape(env.num_envs,-1).any(-1)
    return ~(head|own)


def healthy_mouth_proximity(env,**params):
    return gate_positive_reward(mdp.mouth_ground_proximity_phased(env,**params),contact_clean(env))


def healthy_mouth_perpendicular(env,**params):
    return gate_positive_reward(mdp.mouth_perpendicular_phased(env,**params),contact_clean(env))


def reset_phase_consistent_low(env,env_ids,position_range,velocity_range,asset_cfg):
    from mjlab.envs import mdp as native_mdp
    # Execute historical HOME reset first. Nonselected worlds are not overwritten.
    native_mdp.reset_joints_by_offset(env,env_ids,position_range=position_range,velocity_range=velocity_range,asset_cfg=asset_cfg)
    ids=torch.arange(env.num_envs,device=env.device)if env_ids is None else env_ids
    if not hasattr(env,'_gp_low_reset_mask'):
        env._gp_low_reset_mask=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        env._gp_reset_exposure=torch.zeros(2,device=env.device,dtype=torch.int64)
    chosen=torch.rand(len(ids),device=env.device)<LOW_RESET_PROBABILITY
    env._gp_low_reset_mask[ids]=chosen
    env._gp_reset_exposure+=torch.stack(((~chosen).sum(),chosen.sum()))
    low_ids=ids[chosen]
    if len(low_ids)==0:return
    robot=env.scene['robot'];joint_ids,names=robot.find_joints(POLICY_JOINTS,preserve_order=True)
    assert tuple(names)==POLICY_JOINTS
    q=robot.data.default_joint_pos[low_ids].clone();v=torch.zeros_like(q)
    q[:,joint_ids]=torch.tensor(LOW_Q,device=env.device)+torch.rand(len(low_ids),14,device=env.device)*.006-.003
    root=robot.data.default_root_state[low_ids].clone();root[:,:3]+=env.scene.env_origins[low_ids]
    root[:,2]=env.scene.env_origins[low_ids,2]+LOW_ROOT_HEIGHT+.002
    root[:,3:7]=torch.tensor([math.cos(LOW_ROOT_PITCH/2),0.,math.sin(LOW_ROOT_PITCH/2),0.],device=env.device);root[:,7:]=0
    robot.write_root_state_to_sim(root,env_ids=low_ids)
    robot.write_joint_state_to_sim(q,v,env_ids=low_ids)


class PhaseConsistentGroundPickCommand(mdp.GroundPickPhaseCommand):
    """Only reset selection differs; native compute/period/encoding untouched."""
    def reset(self,env_ids):
        result=super().reset(env_ids)
        if env_ids is not None and len(env_ids)>0:
            mask=self._env._gp_low_reset_mask[env_ids]
            self._gp_phase[env_ids]=torch.where(mask,torch.full_like(self._gp_phase[env_ids],LOW_PHASE),self._gp_phase[env_ids])
        return result


@dataclass(kw_only=True)
class PhaseConsistentGroundPickCommandCfg(mdp.GroundPickPhaseCommandCfg):
    class_type:type=PhaseConsistentGroundPickCommand
    def build(self,env):return PhaseConsistentGroundPickCommand(self,env)


def make_history_ground_pick_env_cfg(play=False,stage='nominal'):
    cfg=make_ground_pick_env_cfg(play=play,stage=stage)
    cfg.rewards['mouth_ground_proximity'].func=healthy_mouth_proximity
    cfg.rewards['mouth_perpendicular_to_ground'].func=healthy_mouth_perpendicular
    if not play:
        cfg.events['reset_robot_joints'].func=reset_phase_consistent_low
        params=vars(cfg.commands['twist']).copy();params.pop('class_type',None)
        cfg.commands['twist']=PhaseConsistentGroundPickCommandCfg(**params)
        for term in cfg.curriculum.values():
            for stages in term.params.values():
                if isinstance(stages,list):
                    for item in stages:
                        if isinstance(item,dict)and 'step'in item:
                            assert item['step']%6==0
                            item['step']//=6
    return cfg


def register_history_ground_pick_tasks():
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import MicroduckGroundPickRlCfg
    for stage in ('nominal','measured'):
        task=HISTORY_TASK+('-Measured'if stage=='measured'else'')
        agent=deepcopy(MicroduckGroundPickRlCfg)
        agent.algorithm.class_name='mjlab_microduck.custom_v2.ppo:TerminalCriticPPO';agent.algorithm.num_mini_batches=24
        agent.experiment_name='microduck_v2_p04_groundpick';agent.run_name='health_history_'+stage
        register_mjlab_task(task_id=task,env_cfg=make_history_ground_pick_env_cfg(stage=stage),play_env_cfg=make_history_ground_pick_env_cfg(play=True,stage=stage),rl_cfg=agent,runner_cls=MicroduckOnPolicyRunner)
