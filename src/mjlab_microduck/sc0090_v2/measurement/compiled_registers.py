"""Opt-in fusion of the measurement generator, with opaque RNG/donor/BLAS ops."""
import torch
import warp as wp
from .registers import RegisterPipeline
from .gpu_donors import choose_donors
from .precision import reference_divide


@torch.library.custom_op('v2_measurement::donors', mutates_args=('starts','offsets'))
def donors(context: torch.Tensor, history: torch.Tensor, pools: torch.Tensor,
           lengths: torch.Tensor, conditions: torch.Tensor, seeds: torch.Tensor,
           q: torch.Tensor, v: torch.Tensor, goal: torch.Tensor, enabled: torch.Tensor,
           mask: torch.Tensor, count: int, starts: torch.Tensor, offsets: torch.Tensor) -> None:
    wp.launch(choose_donors,dim=q.numel(),inputs=[*[wp.from_torch(x) for x in
        (context,history,pools,lengths,conditions,seeds,q,v,goal,enabled,mask)],count,
        wp.from_torch(starts),wp.from_torch(offsets)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(q.device)))


@donors.register_fake
def _(*args):
    return None


@torch.library.custom_op('v2_measurement::calibrated_dot', mutates_args=())
def calibrated_dot(basis: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    # Keep the frozen reference's BLAS reduction order before quantization.
    return basis @ coefficients


@calibrated_dot.register_fake
def _(basis,coefficients):
    return basis.new_empty((basis.shape[0],))


_advance = torch.compile(RegisterPipeline.advance,fullgraph=False,dynamic=True,
                         options={'triton.cudagraphs':False,'use_fast_math':False})


class CompiledRegisterPipeline(RegisterPipeline):
    _calibrated_dot = staticmethod(calibrated_dot)
    _divide = staticmethod(reference_divide)

    @torch.inference_mode(False)
    def advance(self,time_us,q,v,*,mask=None):
        # Keep the private Torch generator and exact full-lane RNG consumption;
        # compiler-generated RNG is not the specified donor sampler.
        from .gpu_donors import make_seeds
        self._compiled_donor_seeds = make_seeds(self)
        return _advance(self,time_us,q,v,mask=mask)
