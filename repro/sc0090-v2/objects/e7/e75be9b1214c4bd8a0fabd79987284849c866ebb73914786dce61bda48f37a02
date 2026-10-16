"""Explicit mature-policy normalization freeze, using RSL's existing `until`."""
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization

RECIPE = {'id': 'mature-parent-normalizer-freeze-v1', 'targets': ['actor','critic'],
          'semantics': 'retain parent mean/variance/std/count exactly; stop future statistic updates'}

def freeze_normalizers(algorithm):
    snapshots = {}
    counts = {}
    for name in RECIPE['targets']:
        norm = getattr(algorithm,name).obs_normalizer
        if type(norm) is not EmpiricalNormalization:
            raise TypeError('Freeze requires the validated empirical normalizer')
        count = int(norm.count)
        if count <= 0:
            raise ValueError('Freeze requires an already observed parent normalizer')
        snapshots[name] = {k:v.detach().clone() for k,v in norm.state_dict().items()}
        norm.until = count
        counts[name] = count
    assert_frozen_normalizers(algorithm,snapshots)
    return {**RECIPE,'until_counts':counts}, snapshots

def assert_frozen_normalizers(algorithm,snapshots):
    for name,expected in snapshots.items():
        norm = getattr(algorithm,name).obs_normalizer
        if norm.until != int(expected['count']):
            raise AssertionError('Normalizer freeze schedule changed')
        actual = norm.state_dict()
        if not all(torch.equal(actual[k],v) for k,v in expected.items()):
            raise AssertionError('Frozen '+name+' normalization state changed')
