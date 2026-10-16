"""Fixed-shape device-only search of the unchanged empirical donor objective.

Each active lane draws the same number of uniform candidates as the reference.
Inactive lanes do no search. No nonzero(), host mask, or N x 1024 x 4 gather.
RNG is explicitly versioned: per-lane Torch seed + Warp uniform candidates.
"""
import torch
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': False})
wp.init()
SAMPLER_VERSION = 'v2-device-donor-1024-v1'


@wp.func
def context_cost(context: wp.array2d(dtype=wp.float32),
                 history: wp.array3d(dtype=wp.float32), lane: int, donor: int,
                 q: float, v: float, goal: float, enabled: float):
    score = wp.abs(context[donor, 0]-q)/0.25 + wp.abs(context[donor, 1]-v)/0.3
    score += 20.0*wp.abs(context[donor, 2]-enabled) + wp.abs(context[donor, 3]-goal)/0.1
    for j in range(3):
        back = 4
        if j == 1:
            back = 9
        elif j == 2:
            back = 18
        old = donor-back
        score += wp.abs((context[old, 0]-context[donor, 0])-(history[lane, 19-back, 0]-q))/0.1
        score += wp.abs(context[old, 1]-history[lane, 19-back, 1])/0.3
        score += wp.abs((context[old, 3]-context[donor, 3])-(history[lane, 19-back, 3]-goal))/0.1
        score += 20.0*wp.abs(context[old, 2]-history[lane, 19-back, 2])
    return score


@wp.kernel
def choose_donors(context: wp.array2d(dtype=wp.float32),
                  history: wp.array3d(dtype=wp.float32),
                  pools: wp.array2d(dtype=wp.int64), lengths: wp.array(dtype=wp.int64),
                  conditions: wp.array(dtype=wp.int64), seeds: wp.array(dtype=wp.int32),
                  q: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.float32),
                  goal: wp.array(dtype=wp.float32), enabled: wp.array(dtype=wp.float32),
                  mask: wp.array(dtype=wp.bool), candidate_count: int,
                  starts: wp.array(dtype=wp.int64), offsets: wp.array(dtype=wp.int64)):
    lane = wp.tid()
    if mask[lane]:
        rng = wp.rand_init(seeds[lane])
        condition = int(conditions[lane])
        length = int(lengths[condition])
        best = float(1.0e30)
        selected = int(0)
        for j in range(candidate_count):
            column = wp.min(int(wp.randf(rng)*float(length)), length-1)
            donor = int(pools[condition, column])
            score = context_cost(context, history, lane, donor, q[lane], v[lane], goal[lane], enabled[lane])
            if score < best:
                best = score
                selected = donor
        starts[lane] = wp.int64(selected)
        offsets[lane] = wp.int64(0)


def make_seeds(pipeline):
    seeds = torch.randint(0, 2**31-1, (getattr(pipeline, '_rng_full_count', pipeline.n),), dtype=torch.int32,
                          generator=pipeline.rng, device=pipeline.device)
    if hasattr(pipeline, '_rng_rows'):
        seeds = seeds[pipeline._rng_rows]
    return seeds


def select(pipeline, q, v, goal, enabled, mask):
    seeds = getattr(pipeline, '_compiled_donor_seeds', None)
    if seeds is not None:
        from .compiled_registers import donors
        donors(pipeline.context,pipeline.physical_context,pipeline.condition_pools,
               pipeline.condition_lengths,pipeline.condition_index,seeds,q,v,goal,enabled,
               mask,pipeline.condition_candidate_count,pipeline.donor_start,pipeline.donor_offset)
        return
    seeds = make_seeds(pipeline)
    wp.launch(choose_donors, dim=pipeline.n, inputs=[
        wp.from_torch(pipeline.context), wp.from_torch(pipeline.physical_context),
        wp.from_torch(pipeline.condition_pools), wp.from_torch(pipeline.condition_lengths),
        wp.from_torch(pipeline.condition_index), wp.from_torch(seeds),
        wp.from_torch(q), wp.from_torch(v), wp.from_torch(goal), wp.from_torch(enabled),
        wp.from_torch(mask), pipeline.condition_candidate_count,
        wp.from_torch(pipeline.donor_start), wp.from_torch(pipeline.donor_offset)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(q.device)))
