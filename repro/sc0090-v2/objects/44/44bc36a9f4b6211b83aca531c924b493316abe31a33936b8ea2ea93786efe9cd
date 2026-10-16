"""GPU sampling of idle, single-axis, paired and combined head requests."""
from dataclasses import dataclass
import torch
from mjlab_microduck.tasks.mdp import UniformPoseCommand, UniformPoseCommandCfg


def sample_head(ranges, n, device, zero_probability, precision_probability=0.):
    bounds = torch.tensor(ranges, device=device)
    values = bounds[:, 0] + torch.rand(n, 4, device=device) * (bounds[:, 1] - bounds[:, 0])
    # Keep endpoints represented as the curriculum widens; random interiors
    # still dominate, rather than learning a small set of fixed button poses.
    endpoint = torch.where(torch.rand(n, 4, device=device) < .5, bounds[:, 0], bounds[:, 1])
    values = torch.where(torch.rand(n, 4, device=device) < .2, endpoint, values)
    if precision_probability > 0:
        small = .005 + .035 * torch.rand(n, 4, device=device)
        small = torch.where(values >= 0, small, -small)
        small = small.clamp(bounds[:, 0], bounds[:, 1])
        precision = torch.rand(n, 1, device=device) < precision_probability
        values = torch.where(precision, small, values)
    u = torch.rand(n, device=device)
    active = (u - zero_probability) / (1 - zero_probability)
    counts = torch.where(active < .5, 1, torch.where(active < .75, 2, 4))
    counts = torch.where(u < zero_probability, 0, counts)
    ranks = torch.rand(n, 4, device=device).argsort(-1).argsort(-1)
    return values * (ranks < counts[:, None])


class BalancedHeadCommand(UniformPoseCommand):
    def _resample_command(self, env_ids):
        self._command[env_ids] = sample_head(self.cfg.ranges, len(env_ids), self.device,
                                             self.cfg.zero_command_prob,
                                             self.cfg.precision_probability)


@dataclass(kw_only=True)
class BalancedHeadCommandCfg(UniformPoseCommandCfg):
    precision_probability: float = 0.

    def build(self, env):
        if len(self.ranges) != 4 or not 0 <= self.zero_command_prob < 1:
            raise ValueError('Four head axes and a nontrivial active sampling probability are required')
        if not 0 <= self.precision_probability <= 1:
            raise ValueError('Precision command probability must be within [0,1]')
        return BalancedHeadCommand(self, env)


def sampling_contract(cfg):
    import hashlib
    from pathlib import Path
    return dict(zero_probability=cfg.zero_command_prob,
        precision_probability=cfg.precision_probability,
        precision_magnitude_range=[.005,.04],ranges=[list(x) for x in cfg.ranges],
        conditional_active_axis_probabilities={'1':.5,'2':.25,'4':.25},
        full_range_endpoint_probability=.2,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
