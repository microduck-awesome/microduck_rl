"""Explicit one-time learning-rate transition after exact mature-state restore."""
import hashlib,math
from pathlib import Path

def apply_initial_learning_rate(algorithm, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive finite transition learning rate required')
    before=float(algorithm.learning_rate)
    groups_before=[float(g['lr']) for g in algorithm.optimizer.param_groups]
    if any(x!=before for x in groups_before):raise ValueError('Algorithm and Adam learning rates already disagree')
    algorithm.learning_rate=float(learning_rate)
    for group in algorithm.optimizer.param_groups:group['lr']=float(learning_rate)
    return dict(id='explicit-mature-initial-lr-v1',before_algorithm_lr=before,
        before_group_lrs=groups_before,after_initial_lr=float(learning_rate),
        applies='once, only after exact parent restore, before the first new rollout',
        retained='actor/critic/normalizer parameters and Adam moments; subsequent adaptive schedule',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
