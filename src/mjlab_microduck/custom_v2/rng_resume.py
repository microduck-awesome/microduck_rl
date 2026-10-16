"""Restore the trained CUDA random stream when moving a single-device policy.

Device indices come from the recorded parent run, never from the current device.
Other saved streams retain the previous set_rng_state_all behavior. This bounded
experiment requires the same number of visible CUDA devices as the checkpoint.
Native environment reset and private simulator generators are separate state.
"""
import hashlib
from pathlib import Path
import torch


def _digest(state):
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def restore_training_rng(metadata, parent_run, target_device):
    source = torch.device(parent_run['device'])
    target = torch.device(target_device)
    if source.type != 'cuda' or source.index is None:
        raise ValueError('Parent run must record an explicit CUDA device index')
    if target.type != 'cuda' or target.index is None:
        raise ValueError('Target must name an explicit CUDA device index')
    if metadata.get('rng_device') is not None:
        if torch.device(metadata['rng_device']) != source:
            raise ValueError('Checkpoint RNG device conflicts with parent run provenance')
    states = [value.detach().cpu() for value in metadata['cuda_rng']]
    count = torch.cuda.device_count()
    if len(states) != count or not (source.index < count and target.index < count):
        raise ValueError('Saved and visible CUDA device counts must match for this experiment')
    cpu = metadata['torch_rng'].detach().cpu()
    if cpu.dtype != torch.uint8 or any(value.dtype != torch.uint8 for value in states):
        raise ValueError('Expected native byte RNG states')
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(states)
    if source != target:
        torch.cuda.set_rng_state(states[source.index], target)
    assert torch.equal(torch.get_rng_state(), cpu)
    restored = torch.cuda.get_rng_state_all()
    for index, state in enumerate(restored):
        expected = states[source.index] if index == target.index else states[index]
        assert torch.equal(state, expected), f'CUDA RNG restoration mismatch on index {index}'
    return dict(status='PASS', source_device=str(source), target_device=str(target),
        active_stream_mapped=source != target, saved_visible_device_count=len(states),
        cpu_rng_sha256=_digest(cpu), source_active_cuda_rng_sha256=_digest(states[source.index]),
        target_restored_cuda_rng_sha256=_digest(restored[target.index]),
        inactive_streams_restored_bitwise=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Native torch CPU/CUDA random streams; not a full simulator-state replay')
