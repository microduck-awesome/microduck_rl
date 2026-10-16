"""Run reset prehistory only for resetting lanes, preserving full-lane RNG draws."""
from copy import copy
import torch

STATIC = {'pool','residual','context','duration_us','condition_pools','condition_lengths',
          'condition_settings','condition_mass','condition_phases','condition_voltages',
          'condition_modes','coefficients','load_coefficients','position_residual_coefficients'}


class ResetSubset:
    @torch.inference_mode(False)
    def __init__(self, source, lanes, capacity=None):
        self.source, self.lanes = source, lanes
        self.capacity = len(lanes) if capacity is None else capacity
        rows = torch.empty(self.capacity, dtype=torch.int64, device=source.device)
        rows[:len(lanes)].copy_(lanes)
        rows[len(lanes):].fill_(0)
        self.pipeline = copy(source)
        self.pipeline.stream = copy(source.stream)
        self.fields = []
        for original, target in ((source,self.pipeline),(source.stream,self.pipeline.stream)):
            names = []
            for name,value in vars(original).items():
                if name not in STATIC and isinstance(value,torch.Tensor) and value.ndim and value.shape[0] == source.n:
                    setattr(target,name,value[rows].clone())
                    names.append(name)
            target.n = self.capacity
            self.fields.append((original,target,names))
        self.pipeline.stream._rows = torch.arange(self.capacity,device=source.device)
        self.pipeline._rng_full_count = source.n
        self.pipeline._rng_rows = rows

    def stage(self, lanes):
        """Reuse storage and captured kernels; padded lanes are always masked out."""
        self.lanes = lanes
        rows = self.pipeline._rng_rows
        rows[:len(lanes)].copy_(lanes)
        rows[len(lanes):].fill_(0)
        for original,target,names in self.fields:
            for name in names:
                if name != '_rows':
                    getattr(target,name).copy_(getattr(original,name)[rows])

    def commit(self):
        for original,target,names in self.fields:
            for name in names:
                if name != '_rows':
                    getattr(original,name).index_copy_(0,self.lanes,getattr(target,name)[:len(self.lanes)])
