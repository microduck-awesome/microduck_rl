"""Explicit mature-policy head exploration cap, shared by every PPO operation.

The learned scalar std parameters and optimizer state remain intact. A fixed,
nonpersistent limit changes the actual Normal in update(), hence sampling,
log-probability, stored distribution parameters, entropy and KL all agree.
Deterministic inference/export remain the upstream identity. This is action
exploration, not sensor noise. Raw std is deliberately distinct from effective.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import torch
from torch.distributions import Normal
from rsl_rl.modules.distribution import GaussianDistribution
from .microduck_v2 import POLICY_JOINTS

HEAD_NAMES=('neck_pitch','head_pitch','head_yaw','head_roll')

class HeadCappedGaussianDistribution(GaussianDistribution):
    def __init__(self, output_dim: int, init_std: float=1., std_type: str='scalar',
                 head_std_cap: float=.01):
        if output_dim != len(POLICY_JOINTS) or std_type != 'scalar':
            raise ValueError('Only the V2 14-axis scalar Gaussian is qualified')
        if not math.isfinite(head_std_cap) or head_std_cap <= 0:
            raise ValueError('Positive finite head exploration cap required')
        super().__init__(output_dim, init_std, std_type)
        self.head_std_cap=float(head_std_cap)
        self.head_indices=tuple(POLICY_JOINTS.index(n) for n in HEAD_NAMES)
        limits=torch.full((output_dim,),float('inf'))
        limits[list(self.head_indices)]=head_std_cap
        self.register_buffer('_head_std_limits',limits,persistent=False)

    def effective_std(self):
        # No lower clamp: invalid inherited/learned raw std remains an error,
        # not an undisclosed extra distribution transform.
        torch._assert_async((torch.isfinite(self.std_param)&(self.std_param>0)).all(),
                            'Nonpositive or nonfinite raw Gaussian std')
        return torch.minimum(self.std_param,self._head_std_limits)

    def update(self, mlp_output):
        self._distribution=Normal(mlp_output,self.effective_std().expand_as(mlp_output))

    @torch.no_grad()
    def exploration_metadata(self):
        effective=self.effective_std()
        legs=[i for i,n in enumerate(POLICY_JOINTS) if n not in HEAD_NAMES]
        assert torch.equal(effective[legs],self.std_param[legs])
        return dict(id='v2-head-effective-std-cap-v1', head_std_cap_rad=self.head_std_cap,
                    policy_joint_order=list(POLICY_JOINTS),head_joint_names=list(HEAD_NAMES),
                    raw_std_parameter_rad=self.std_param.detach().cpu().tolist(),
                    effective_std_rad=effective.detach().cpu().tolist(),
                    leg_std_unchanged=True,raw_parameter_and_optimizer_preserved=True,
                    state_dict_keys=list(self.state_dict()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    semantics='cap actual Normal before sampling, log_prob, entropy, params and KL; deterministic identity')
