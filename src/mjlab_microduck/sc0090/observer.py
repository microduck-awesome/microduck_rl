"""Frozen FP32 O0/O1 computation on causal raw register feature histories.

This module evaluates prebuilt features. The stream frontend, history clock,
fault warmup and bounded output query remain the caller's responsibility.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

O1_SHA256 = '312b877d33dbc643036aa99970ebfc491e1659bf1c45cc65dfa06da212260629'


class FrozenObserver(nn.Module):
    def __init__(self, weight_path, mode):
        super().__init__()
        if mode not in {'O0', 'O1'}:
            raise ValueError('Expected an explicit O0/O1 identity')
        if hashlib.sha256(Path(weight_path).read_bytes()).hexdigest() != O1_SHA256:
            raise ValueError('Expected the frozen matched-native100 FP32 observer')
        self.mode = mode
        shapes = {'mapping': (2,), 'mean': (100,), 'std': (100,), 'residual_scale': (2,),
                  'w0': (100, 32), 'b0': (32,), 'w1': (32, 16), 'b1': (16,), 'w2': (16, 2), 'b2': (2,)}
        with np.load(weight_path, allow_pickle=False) as archive:
            for name, shape in shapes.items():
                value = archive[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid frozen observer array {name}')
                self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        if (self.std <= 0).any():
            raise ValueError('Observer normalization must have positive scales')

    @torch.no_grad()
    def forward(self, history, line_q3):
        """Return physical q/v and validity; invalid inputs never yield an actor observation.

        History is (..., 10, 10), oldest to newest at 20ms intervals. Fields are
        nominal position/confirmed goal, raw speed/load, age, enable/P, LS3/LS5
        and raw voltage, exactly as the frozen C model expects.
        """
        if history.shape[-2:] != (10, 10) or line_q3.shape != history.shape[:-2]:
            raise ValueError('Expected ten ten-field history frames and a matching LS3 position')
        valid = torch.isfinite(history).all(dim=(-2, -1)) & torch.isfinite(line_q3)
        valid &= ((history[..., 4] >= 0.) & (history[..., 4] <= .025)).all(dim=-1)
        safe = torch.where(valid[..., None, None], history, 0.).to(dtype=torch.float32)
        q3 = torch.where(valid, line_q3, 0.).to(dtype=torch.float32)
        baseline = torch.stack([self.mapping[0]+self.mapping[1]*q3,
                                self.mapping[1]*safe[..., -1, 7]], dim=-1)
        estimate = baseline
        if self.mode == 'O1':
            x = safe.clone()
            current = self.mapping[0]+self.mapping[1]*safe[..., -1, 0]
            x[..., :2] = self.mapping[0]+self.mapping[1]*x[..., :2]-current[..., None, None]
            x[..., 7:9] *= self.mapping[1]
            x[..., -1, 0] = current
            x = (x.flatten(start_dim=-2)-self.mean)/self.std
            x = torch.relu(x@self.w0+self.b0)
            x = torch.relu(x@self.w1+self.b1)
            estimate = baseline+(x@self.w2+self.b2)*self.residual_scale
        valid &= torch.isfinite(estimate).all(dim=-1)
        return torch.where(valid[..., None], estimate, float('nan')), valid
