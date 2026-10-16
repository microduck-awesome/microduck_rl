"""Independent bounded physical head accuracy bonus for mature V2 policies.

The existing Gaussian reward remains untouched. Commanded axes share this
bonus; at idle all four axes share it. No error clipping or temporal state.
"""
import hashlib
from pathlib import Path
import torch

HEAD_JOINTS=('neck_pitch','head_pitch','head_yaw','head_roll')


def accuracy_scores(error, command, relative_band=.1, minimum_band=.002):
    assert relative_band>0 and minimum_band>0
    band=torch.maximum(command.abs()*relative_band,command.new_full((),minimum_band))
    per_axis=1.0/(1.0+error.abs()/band)
    active=(command.abs()>1e-9).to(error.dtype)
    count=active.sum(-1)
    commanded=(per_axis*active).sum(-1)/count.clamp_min(1)
    score=torch.where(count>0,commanded,per_axis.mean(-1))
    return score,per_axis,band


def physical_head_command_accuracy(env, command_name='head_pose', relative_band=.1, minimum_band=.002):
    robot=env.scene['robot']
    if not hasattr(env,'_v2_accuracy_head_ids'):
        ids,names=robot.find_joints(HEAD_JOINTS,preserve_order=True)
        assert tuple(names)==HEAD_JOINTS
        # V2 has no passive backlash joint: the joint is the physical output.
        assert not any(f'passive_{name}_backlash' in robot.joint_names for name in HEAD_JOINTS)
        env._v2_accuracy_head_ids=ids
    ids=env._v2_accuracy_head_ids
    command=env.command_manager.get_command(command_name)
    error=robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids]-command
    score,per_axis,band=accuracy_scores(error,command,relative_band,minimum_band)
    # Read-only diagnostic access to uncapped physical errors and bounded scores.
    env._v2_head_accuracy_error=error
    env._v2_head_accuracy_per_axis=per_axis
    env._v2_head_accuracy_band=band
    return score


def accuracy_contract(cfg):
    term=cfg.rewards.get('head_command_accuracy')
    if term is None:return None
    assert term.func is physical_head_command_accuracy
    return dict(id='v2-command-axis-accuracy-v1',weight=term.weight,params=term.params,
                active_axis_threshold=1e-9,idle='mean_over_all_four_axes',
                physical_output='V2 joint_pos minus HOME, no passive backlash',
                formula='mean_active(1/(1+abs(error)/max(relative_band*abs(command),minimum_band)))',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def attach_accuracy_oracle(env):
    """Preflight only: compare one native reward computation to physical oracle."""
    cfg=env.cfg.rewards.get('head_command_accuracy')
    if cfg is None:return None
    assert cfg.params==dict(relative_band=.1,minimum_band=.002)
    robot=env.scene['robot'];ids,names=robot.find_joints(HEAD_JOINTS,preserve_order=True)
    assert tuple(names)==HEAD_JOINTS
    index=env.reward_manager.active_terms.index('head_command_accuracy')
    original=env.reward_manager.compute
    record=dict(calls=0,max_abs_delta=torch.zeros((),device=env.device))
    def checked(dt):
        result=original(dt)
        command=env.command_manager.get_command('head_pose')
        error=robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids]-command
        band=torch.maximum(.1*command.abs(),torch.full_like(command,.002))
        scores=band/(band+error.abs())
        mask=(command.abs()>1e-9).to(error.dtype);count=mask.sum(-1)
        expected=torch.where(count>0,(scores*mask).sum(-1)/count.clamp_min(1),scores.mean(-1))
        actual=env.reward_manager._step_reward[:,index]/cfg.weight
        torch._assert_async(torch.isclose(actual,expected,atol=1e-7,rtol=1e-6).all(),
                            'Accuracy cached reward differs from independent physical oracle')
        record['max_abs_delta'].copy_(torch.maximum(record['max_abs_delta'],(actual-expected).abs().max()))
        record['calls']+=1
        return result
    env.reward_manager.compute=checked
    return record
