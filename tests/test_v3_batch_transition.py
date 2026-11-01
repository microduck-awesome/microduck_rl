"""Control-step accounting when V3 changes the number of environments."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_v2_standup import inherited_control_steps, restrict_evaluation_events
from reassess_v3_checkpoint import same
import torch


def test_fresh_parent_uses_original_batch_size():
    info={'v2':{'completed_updates':500}}
    parent={'resume':None,'num_envs':8192,'allocation_updates':500,'agent':{'num_steps_per_env':24}}
    assert inherited_control_steps(info,parent,24)==98_304_000


def test_new_checkpoints_keep_recorded_cumulative_total():
    info={'v2':{'completed_updates':505,'total_environment_steps':98_304_000+5*16384*24}}
    assert inherited_control_steps(info,{'resume':'earlier'},24)==100_270_080


def test_missing_lineage_or_rollout_mismatch_fails_closed():
    info={'v2':{'completed_updates':500}}
    for parent in ({'resume':'previous'},{'resume':None,'num_envs':8192,'allocation_updates':500,'agent':{'num_steps_per_env':16}}):
        try:
            inherited_control_steps(info,parent,24)
        except ValueError:
            pass
        else:
            raise AssertionError('Ambiguous sample count must fail')


def test_evaluation_preserves_ordered_home_reset_and_rejects_missing_or_late_reset():
    names=['reset_base','reset_robot_joints','reset_action_history','set_ground_state',
           'unrelated_randomizer','v2_reset_joint_guards','v3_reset_ground_clearance']
    cfg=SimpleNamespace(events={name:object() for name in names})
    restrict_evaluation_events(cfg)
    assert list(cfg.events)==[name for name in names if name!='unrelated_randomizer']
    for invalid in ([name for name in names if name!='reset_robot_joints'],
                    ['reset_base','set_ground_state','reset_robot_joints','reset_action_history',
                     'v2_reset_joint_guards','v3_reset_ground_clearance']):
        cfg=SimpleNamespace(events={name:object() for name in invalid})
        try:
            restrict_evaluation_events(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing or reordered HOME reset must fail')


def test_reassessment_state_comparison_rejects_numerical_changes():
    first={'actor':{'weight':torch.tensor([1.,2.])},'optimizer':[torch.tensor([.1]),3]}
    second={'actor':{'weight':torch.tensor([1.,2.])},'optimizer':[torch.tensor([.1]),3]}
    assert same(first,second)
    second['actor']['weight'][1]=2.0001
    assert not same(first,second)
