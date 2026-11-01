"""Policy joint limit guard must not shift the independent mouth HOME."""
from types import SimpleNamespace
import torch
from mjlab_microduck.custom_v2.training import guard_reset_joints
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS, HOME_JOINT_POSITIONS


def test_guard_clamps_only_selected_policy_joints_and_worlds():
    names=list(POLICY_JOINTS[:9])+['mouth']+list(POLICY_JOINTS[9:])
    q=torch.tensor([[HOME_JOINT_POSITIONS[n] for n in names]]*2)
    limits=torch.tensor([[[-2.,2.]]*len(names)]*2)
    mouth=names.index('mouth');limits[:,mouth,:]=torch.tensor([0.,.48])
    q[0,names.index('left_hip_yaw')]=2.1
    q[1,names.index('left_hip_yaw')]=2.1
    class Asset:
        data=SimpleNamespace(joint_pos=q,joint_pos_limits=limits)
        def find_joints(self,joint_names,preserve_order=False):
            assert preserve_order
            return [names.index(n) for n in joint_names],list(joint_names)
        def write_joint_state_to_sim(self,position,velocity,joint_ids,env_ids):
            assert torch.count_nonzero(velocity)==0
            self.data.joint_pos[env_ids[:,None],torch.tensor(joint_ids)] = position
    asset=Asset();env=SimpleNamespace(scene={'robot':asset},device='cpu',num_envs=2)
    guard_reset_joints(env,torch.tensor([0]))
    assert q[0,mouth]==0 and q[1,mouth]==0
    torch.testing.assert_close(q[0,names.index('left_hip_yaw')],torch.tensor(1.998))
    torch.testing.assert_close(q[1,names.index('left_hip_yaw')],torch.tensor(2.1))
