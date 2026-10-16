"""Opt-in, update-preserving diagnostics for the pinned feedforward PPO.

Install on a loaded runner with ``attach_ppo_diagnostics(runner, jsonl_path)``.
Legacy trainers/checkpoints and site-packages stay unchanged. Unsupported custom
objectives, distributed learners, RND, symmetry and recurrent models fail closed.

The optimization arithmetic below follows rsl_rl PPO.update, copyright
(c) 2021-2026 ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause. The upstream
method fingerprint and paired CPU/CUDA tests guard that limited implementation.
License text: licenses/rsl_rl-BSD-3-Clause.txt in the repository root.
"""
from __future__ import annotations

from functools import partial
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution

UPSTREAM_UPDATE_SHA256 = '507b4a88853dc6eda17c7c2e79bd081430c97ec8ca2e2efc24572729c94907b0'
CONTRACT = {
    'schema': 1,
    'kl': 'KL(old || current), original rollout samples, before each minibatch optimizer step',
    'kl_p95': 'pooled sample-visits over all epochs; not p95 of minibatch means',
    'clip_fraction': 'sample-visits with ratio outside [1-clip_param, 1+clip_param]',
    'gradients': 'actor/critic separate L2 norms; original clipping calls retained',
    'gradient_clip_fraction': 'minibatches with pre-clip norm > max_grad_norm',
    'explained_variance': '1-Var(returns-values)/Var(returns), population variance, fixed rollout',
    'ev_invalid': 'return variance <= 1e-12: valid=0 and EV tag omitted',
    'statistics_only': True,
}


@torch.no_grad()
def explained_variance(target, prediction, *, variance_epsilon=1e-12):
    """Return EV (or None), target variance and error MSE without clipping EV."""
    y = target.detach().reshape(-1).double()
    p = prediction.detach().reshape(-1).double()
    if y.shape != p.shape or y.numel() < 2:
        raise ValueError('EV requires matching arrays with at least two samples')
    if not bool(torch.isfinite(y).all() and torch.isfinite(p).all()):
        raise FloatingPointError('Nonfinite value diagnostics')
    variance = y.var(correction=0)
    residual = y - p
    mse = residual.square().mean()
    ev = None if float(variance) <= variance_epsilon else float(1 - residual.var(correction=0) / variance)
    return ev, float(variance), float(mse)


@torch.no_grad()
def gradient_norm(parameters):
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        raise ValueError('Expected gradients on the monitored model')
    return torch.stack(norms).norm(2)


@torch.no_grad()
def policy_statistics(kl_visits, ratio_visits, clip_param):
    kl = torch.cat([v.detach().reshape(-1) for v in kl_visits])
    ratio = torch.cat([v.detach().reshape(-1) for v in ratio_visits])
    if kl.shape != ratio.shape or not kl.numel():
        raise ValueError('Expected aligned KL and likelihood-ratio samples')
    if not bool(torch.isfinite(kl).all() and torch.isfinite(ratio).all()):
        raise FloatingPointError('Nonfinite PPO diagnostic samples')
    outside = (ratio < 1 - clip_param) | (ratio > 1 + clip_param)
    return {
        'PPO/kl_mean': float(kl.double().mean()),
        'PPO/kl_p95': float(torch.quantile(kl.float(), .95)),
        'PPO/kl_max': float(kl.max()),
        'PPO/kl_minibatch_mean_max': max(float(v.double().mean()) for v in kl_visits),
        'PPO/clip_fraction': float(outside.double().mean()),
        'PPO/sample_visits': float(kl.numel()),
    }


@torch.no_grad()
def value_predictions(algorithm, chunk_size=16384):
    # MLPModel.forward does not update normalizers or draw random numbers for a
    # critic without an output distribution. Do not call update_normalization.
    observations = algorithm.storage.observations.reshape(-1)
    return torch.cat([algorithm.critic(observations[i:i+chunk_size]).detach()
                      for i in range(0, observations.batch_size[0], chunk_size)])


class Diagnostics:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.original_update = algorithm.update
        self.last = {}
        self.updates = 0
        self.closed = False
        self.restore_logger = None
        self.record_actor_groups = False
        self.actor_group_clipping = False

    def close(self):
        if not self.closed:
            self.algorithm.update = self.original_update
            if self.restore_logger is not None:
                self.restore_logger()
            self.closed = True


def enable_ppo_diagnostics(algorithm):
    """Attach to standard PPO/TerminalCriticPPO; preserve optimizer and models."""
    if getattr(algorithm.update, '__func__', None) is not PPO.update:
        raise ValueError('Diagnostics requires the unmodified upstream PPO.update')
    if hashlib.sha256(inspect.getsource(PPO.update).encode()).hexdigest() != UPSTREAM_UPDATE_SHA256:
        raise ValueError('Upstream PPO.update changed; review diagnostics and repeat parity tests')
    from mjlab_microduck.custom_v2.head_exploration import HeadCappedGaussianDistribution
    if (algorithm.is_multi_gpu or algorithm.rnd or algorithm.symmetry
            or type(algorithm.actor) is not MLPModel or type(algorithm.critic) is not MLPModel
            or type(algorithm.actor.distribution) not in (GaussianDistribution,HeadCappedGaussianDistribution)
            or algorithm.critic.distribution is not None):
        raise ValueError('Diagnostics supports single-device Gaussian MLP PPO without auxiliary objectives')
    if algorithm.actor.is_recurrent or algorithm.critic.is_recurrent:
        raise ValueError('Recurrent diagnostic sample weighting is not implemented')
    if {id(p) for p in algorithm.actor.parameters()} & {id(p) for p in algorithm.critic.parameters()}:
        raise ValueError('Shared actor/critic parameters require separate clipping semantics')
    monitor = Diagnostics(algorithm)
    algorithm.update = partial(_update, algorithm, monitor)
    return monitor


def _update(self, monitor):
    """Same supported PPO operations/order/RNG draws; detached observations only."""
    if self.storage.step != self.storage.num_transitions_per_env:
        raise ValueError('Diagnostics requires one complete rollout')
    ev_rollout, target_variance, mse_rollout = explained_variance(self.storage.returns, self.storage.values)
    ev_before, _, mse_before = explained_variance(self.storage.returns, value_predictions(self))
    means = dict(value=0., surrogate=0., entropy=0.)
    kl_visits, ratio_visits = [], []
    first_distribution_delta = None
    gradients = {key: [] for key in ('actor_pre', 'actor_post', 'critic_pre', 'critic_post')}
    groups = actor_gradient_groups(self.actor) if monitor.record_actor_groups else {}
    group_metrics = {name: {key: [] for key in ('pre', 'post', 'step', 'relative_step')} for name in groups}
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
        original_batch_size = batch.observations.batch_size[0]
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        # Keep stochastic_output=True, including upstream's otherwise-unused RNG draw.
        self.actor(batch.observations, masks=batch.masks,
                   hidden_state=batch.hidden_states[0], stochastic_output=True)
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)
        values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        if first_distribution_delta is None:
            first_distribution_delta=tuple(float((old-new).detach().abs().max()) for old,new in zip(batch.old_distribution_params,distribution_params,strict=True))
        entropy = self.actor.output_entropy[:original_batch_size]
        with torch.inference_mode():
            kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
            kl_mean = torch.mean(kl)
        if self.desired_kl is not None and self.schedule == 'adaptive':
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
        surrogate = -torch.squeeze(batch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        if self.use_clipped_value_loss:
            value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - batch.returns).pow(2)
            value_losses_clipped = (value_clipped - batch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (batch.returns - values).pow(2).mean()
        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        before = {}
        for name, parameters in groups.items():
            group_metrics[name]['pre'].append(gradient_norm(parameters).detach())
            before[name] = [p.detach().clone() for p in parameters]
        # Default retains exactly the upstream clipping call. The opt-in arm
        # clips disjoint hidden/output/std groups independently, each once.
        if monitor.actor_group_clipping:
            if not groups:
                raise ValueError('Group clipping requires explicit complete actor groups')
            actor_pre = gradient_norm(self.actor.parameters())
            for parameters in groups.values():
                nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        else:
            actor_pre = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_pre = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        gradients['actor_pre'].append(actor_pre.detach())
        gradients['critic_pre'].append(critic_pre.detach())
        gradients['actor_post'].append(gradient_norm(self.actor.parameters()))
        gradients['critic_post'].append(gradient_norm(self.critic.parameters()))
        for name, parameters in groups.items():
            group_metrics[name]['post'].append(gradient_norm(parameters).detach())
        self.optimizer.step()
        with torch.no_grad():
            for name, parameters in groups.items():
                step = torch.stack([(p-old).norm() for p, old in zip(parameters, before[name], strict=True)]).norm()
                scale = torch.stack([p.norm() for p in before[name]]).norm().clamp_min(1e-12)
                group_metrics[name]['step'].append(step)
                group_metrics[name]['relative_step'].append(step / scale)
        means['value'] += value_loss.item()
        means['surrogate'] += surrogate_loss.item()
        means['entropy'] += entropy.mean().item()
        kl_visits.append(kl.detach())
        ratio_visits.append(ratio.detach())
    count = self.num_learning_epochs * self.num_mini_batches
    assert len(kl_visits) == count
    means = {key: value / count for key, value in means.items()}
    ev_after, _, mse_after = explained_variance(self.storage.returns, value_predictions(self))
    metrics = policy_statistics(kl_visits, ratio_visits, self.clip_param)
    metrics['PPO/kl_first_minibatch_mean'] = float(kl_visits[0].double().mean())
    metrics['PPO/kl_first_minibatch_max'] = float(kl_visits[0].max())
    metrics['PPO/mean_first_minibatch_max_abs_recompute_delta'] = first_distribution_delta[0]
    metrics['PPO/std_first_minibatch_max_abs_recompute_delta'] = first_distribution_delta[1]
    metrics['PPO/ratio_first_minibatch_max_abs_error_from_one'] = float((ratio_visits[0]-1).abs().max())
    metrics['PPO/learning_rate_after_update'] = float(self.learning_rate)
    for model in ('actor', 'critic'):
        for phase in ('pre', 'post'):
            values = torch.stack(gradients[f'{model}_{phase}'])
            metrics[f'Grad/{model}_norm_{phase}_mean'] = float(values.double().mean())
            metrics[f'Grad/{model}_norm_{phase}_max'] = float(values.max())
        metrics[f'Grad/{model}_clip_fraction'] = float((torch.stack(gradients[f'{model}_pre']) > self.max_grad_norm).double().mean())
    metrics['Grad/max_norm_limit'] = float(self.max_grad_norm)
    if groups:
        metrics['Grad/group_clipping_enabled'] = float(monitor.actor_group_clipping)
        for name, items in group_metrics.items():
            for key, values in items.items():
                metrics[f'Grad/{name}_{key}_mean'] = float(torch.stack(values).double().mean())
            metrics[f'Grad/{name}_retained_fraction_mean'] = float((torch.stack(items['post']) / torch.stack(items['pre']).clamp_min(1e-12)).double().mean())
    metrics['Critic/return_variance'] = target_variance
    metrics['Critic/explained_variance_valid'] = float(ev_before is not None)
    for phase, ev, mse in (('rollout', ev_rollout, mse_rollout), ('before', ev_before, mse_before), ('after', ev_after, mse_after)):
        if ev is not None:
            metrics[f'Critic/explained_variance_{phase}'] = ev
        metrics[f'Critic/mse_{phase}'] = mse
    metrics['PPO/minibatches'] = float(count)
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Nonfinite PPO diagnostics')
    self.storage.clear()
    monitor.updates += 1
    monitor.last = metrics
    # Do not mislabel telemetry as Loss/* or add it to existing loss checks.
    return means


def actor_gradient_groups(actor):
    """Partition every actor parameter once, with no reparameterization."""
    linear = [layer for layer in actor.mlp.modules() if isinstance(layer, nn.Linear)]
    if len(linear) < 2:
        raise ValueError('Actor needs a hidden trunk and a linear output layer')
    output = list(linear[-1].parameters())
    std = list(actor.distribution.parameters())
    excluded = {id(p) for p in output + std}
    hidden = [p for p in actor.parameters() if id(p) not in excluded]
    groups = dict(hidden=hidden, output=output, std=std)
    flat = [p for ps in groups.values() for p in ps]
    if (not all(groups.values()) or len({id(p) for p in flat}) != len(flat)
            or {id(p) for p in flat} != {id(p) for p in actor.parameters()}):
        raise ValueError('Actor gradient partition is incomplete or overlapping')
    return groups


def attach_ppo_diagnostics(runner, jsonl_path, *, record_actor_groups=False,
                           actor_group_clipping=False):
    """Record proper TensorBoard groups plus a durable per-update JSONL sidecar."""
    path = Path(jsonl_path)
    if path.exists():
        raise FileExistsError('Use a new segment directory for diagnostics: ' + str(path))
    monitor = enable_ppo_diagnostics(runner.alg)
    monitor.record_actor_groups = record_actor_groups or actor_group_clipping
    monitor.actor_group_clipping = actor_group_clipping
    original_log = runner.logger.log
    signature = inspect.signature(original_log)

    def log(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        it = int(bound.arguments['it'])
        if not monitor.last:
            raise RuntimeError('Logger reached before a diagnostics update')
        original_log(*args, **kwargs)
        if runner.logger.writer is not None:
            for name, value in monitor.last.items():
                runner.logger.writer.add_scalar(name, value, it)
            runner.logger.writer.add_scalar('Telemetry/new_updates', monitor.updates, it)
            record = dict(schema=1, legacy_step=it, new_update=monitor.updates, metrics=monitor.last)
            with path.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
    runner.logger.log = log
    monitor.restore_logger = lambda: setattr(runner.logger, 'log', original_log)
    return monitor
