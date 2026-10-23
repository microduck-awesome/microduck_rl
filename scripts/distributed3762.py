"""Global observation and advantage statistics for split baseline batches."""
import torch

def install_global_statistics(alg):
    world = torch.distributed.get_world_size()
    for model in (alg.actor, alg.critic):
        normalizer = model.obs_normalizer
        original = normalizer.update
        def update(x, original=original):
            batches = [torch.empty_like(x) for _ in range(world)]
            torch.distributed.all_gather(batches, x.contiguous())
            # Rollout runs under inference_mode; persistent buffers must remain
            # ordinary tensors so the next learn() can load/broadcast them.
            with torch.inference_mode(False), torch.no_grad():
                original(torch.cat(batches, dim=0))
        normalizer.update = update
    original_returns = alg.compute_returns
    def returns(obs):
        # Obtain unnormalized GAE, then use the original global-batch convention.
        flag = alg.normalize_advantage_per_mini_batch
        alg.normalize_advantage_per_mini_batch = True
        try:
            original_returns(obs)
        finally:
            alg.normalize_advantage_per_mini_batch = flag
        if not flag:
            a = alg.storage.advantages
            stats = torch.stack((a.double().sum(), a.double().square().sum(),
                                 torch.tensor(a.numel(), device=a.device, dtype=torch.float64)))
            torch.distributed.all_reduce(stats)
            mean = stats[0] / stats[2]
            std = ((stats[1] - stats[2]*mean.square()) / (stats[2]-1)).clamp_min(0).sqrt()
            alg.storage.advantages = (a - mean.to(a.dtype)) / (std.to(a.dtype) + 1e-8)
    alg.compute_returns = returns
