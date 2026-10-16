"""Experimental fused tensor frontend; admission requires reference parity.

Kept opt-in until numerical and throughput checks pass. Torch/Inductor versions
are pinned by the project. CUDA graph ownership stays with the caller.
"""
import torch
from .stream import CausalStream
from .precision import reference_divide


@torch.library.custom_op('v2_measurement::least_squares', mutates_args=())
def least_squares(sample_t: torch.Tensor, sample_q: torch.Tensor,
                  grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the original FP64 reduction order, including cancellation at zero
    # velocity. All operations remain on device and inside the caller's graph.
    return CausalStream._least_squares(sample_t,sample_q,grid)


@least_squares.register_fake
def _(sample_t, sample_q, grid):
    shape = (sample_q.shape[0],2)
    return sample_q.new_empty(shape),sample_q.new_empty(shape)


_options = {'triton.cudagraphs': False, 'use_fast_math': False}
_tick = torch.compile(CausalStream.tick, fullgraph=True, dynamic=True, options=_options)
_read = torch.compile(CausalStream.read, fullgraph=True, dynamic=True, options=_options)
_query = torch.compile(CausalStream.query, fullgraph=True, dynamic=True, options=_options)


class CompiledCausalStream(CausalStream):
    _least_squares = staticmethod(least_squares)
    _divide = staticmethod(reference_divide)

    def __init__(self, *args, **kwargs):
        import triton
        if triton.knobs.language.default_fp_fusion:
            raise RuntimeError('Exact compiled S0/O0 requires TRITON_DEFAULT_FP_FUSION=0')
        super().__init__(*args, **kwargs)

    @torch.compiler.disable(recursive=False)
    def tick(self, *args, **kwargs):
        with torch.inference_mode(False), torch.no_grad():
            return _tick(self, *args, **kwargs)

    @torch.compiler.disable(recursive=False)
    def read(self, *args, **kwargs):
        with torch.inference_mode(False), torch.no_grad():
            return _read(self, *args, **kwargs)

    def query(self, *args, **kwargs):
        with torch.inference_mode(False), torch.no_grad():
            return _query(self, *args, **kwargs)
