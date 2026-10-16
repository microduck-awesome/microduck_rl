"""PPO bootstrapping from explicit, pre-reset successor critic observations."""

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel


TERMINAL_CRITIC_KEY = 'sc0090_terminal_critic_v1'
BOOTSTRAP_CONTRACT = {'schema': 1, 'state': 'no_reset_successor_critic_v1',
                      'normalization': 'before_step_update', 'terminal_precedence': True}


def require_collectively(algorithm, valid):
    flag = torch.tensor(int(valid), device=algorithm.device, dtype=torch.int32)
    if algorithm.is_multi_gpu:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if not bool(flag):
        raise RuntimeError('Invalid or missing terminal critic observations; refusing PPO collection on all ranks')


class TerminalCriticPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_bootstrap_samples = 0
        if self.rnd or type(self.actor) is not MLPModel or type(self.critic) is not MLPModel:
            raise ValueError('Terminal critic PPO requires feedforward MLPs without RND')
        if tuple(self.critic.obs_groups) != ('critic',):
            raise ValueError('Terminal critic PPO requires a separate critic observation group')

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        algorithm._terminal_env = env.unwrapped
        return algorithm

    def act(self, obs):
        # Inference/export use actor.forward directly; they do not need a second
        # simulation or a different play observation layout. Collection installs
        # the producer before its first actual training action.
        env = self._terminal_env
        if not env.cfg.is_finite_horizon and not hasattr(env, '_sc0090_terminal_critic_recorder'):
            from .terminal import SC0090TerminalCriticRecorder
            SC0090TerminalCriticRecorder.install(env)
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        if self._terminal_env.cfg.is_finite_horizon:
            # Finite-horizon tasks do not bootstrap at their true time boundary.
            remaining = {key: value for key, value in extras.items() if key != 'time_outs'}
            return super().process_env_step(obs, rewards, dones, remaining)
        record = extras.get(TERMINAL_CRITIC_KEY)
        ids = record.get('env_ids') if isinstance(record, dict) else None
        terminal = record.get('critic') if isinstance(record, dict) else None
        mask = extras.get('time_outs')
        mask_valid = (isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                      and mask.device == rewards.device and mask.shape == rewards.shape)
        expected = mask.nonzero(as_tuple=False).flatten() if mask_valid else None
        valid = (
            mask_valid and isinstance(ids, torch.Tensor) and ids.dtype == torch.long
            and ids.device == rewards.device and ids.ndim == 1
            and isinstance(terminal, torch.Tensor) and terminal.device == rewards.device
            and terminal.dtype == obs['critic'].dtype
            and terminal.shape == (len(ids), self.critic.obs_dim)
            and type(record.get('step')) is int
            and record.get('step') == self._terminal_env.common_step_counter
            and torch.equal(ids, expected)
            and bool(dones[ids].all())
        )
        # The same collective is reached on every rank, including ranks without
        # a timeout. A bad local record must not leave peers in later PPO calls.
        require_collectively(self, valid)
        self.terminal_bootstrap_samples += len(ids)
        adjusted = rewards
        if len(ids):
            # Evaluate BEFORE the once-per-step normalizer update in the base
            # method, using the same normalization snapshot as V(s_t).
            with torch.no_grad():
                terminal_obs = TensorDict({'critic': terminal}, batch_size=[len(ids)])
                value = self.critic(terminal_obs).detach().squeeze(-1)
            adjusted = rewards.clone()
            adjusted[ids] += self.gamma * value
        # Do not replace transition.values: it is still V(s_t), also used by
        # value clipping. Suppress upstream's old V(s_t) timeout addition only.
        remaining = {key: value for key, value in extras.items() if key != 'time_outs'}
        return super().process_env_step(obs, adjusted, dones, remaining)

    def save(self):
        saved = super().save()
        saved['terminal_critic_bootstrap'] = BOOTSTRAP_CONTRACT.copy()
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        # Explicitly selecting this learner permits migration of old checkpoints
        # without changing weights, optimizer, normalizers, LR or stage clocks.
        contract = loaded_dict.get('terminal_critic_bootstrap')
        if contract is not None and contract != BOOTSTRAP_CONTRACT:
            raise ValueError('Unknown terminal critic bootstrap checkpoint contract')
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get('optimizer'):
            # Upstream restores Adam's groups but leaves the adaptive schedule's
            # Python LR at its constructor value. Its next minibatch would then
            # overwrite the resumed LR with that unrelated value.
            rates = {float(group['lr']) for group in self.optimizer.param_groups}
            if len(rates) != 1:
                raise ValueError('Expected one shared PPO learning rate')
            self.learning_rate = rates.pop()
        return result
