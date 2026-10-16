"""GPU primitives whose original rounding is part of causal state parity."""
import torch


@torch.library.custom_op('v2_measurement::reference_divide', mutates_args=())
def reference_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator


@reference_divide.register_fake
def _(numerator, denominator):
    return torch.empty_like(numerator)
