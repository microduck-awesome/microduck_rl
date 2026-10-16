"""Stagger only the first timeout deadline; preserve true physical episode age."""
from pathlib import Path
import hashlib
import torch

SEED_BASE = 2026101507
RECIPE = dict(id='first-timeout-deadline-phase-v1', interval='uniform integer 1..max_episode_length',
    physical_episode_age='unchanged', after_reset='native full episode length',
    rng='dedicated device Generator; seed base + inherited completed updates',
    scope='once per process after exact mature restore and native initial reset')

def reset_timeout_deadlines(env, env_ids):
    if not hasattr(env, '_first_timeout_deadline'):
        env._first_timeout_deadline = torch.full_like(env.episode_length_buf, env.max_episode_length)
        env._first_timeout_active = torch.zeros_like(env.episode_length_buf, dtype=torch.bool)
    selected = slice(None) if env_ids is None else env_ids
    env._first_timeout_deadline[selected] = env.max_episode_length
    env._first_timeout_active[selected] = False


def phased_time_out(env):
    return env.episode_length_buf >= env._first_timeout_deadline


def activate_first_timeout_phase(env, completed_updates):
    assert not getattr(env, '_first_timeout_phase_armed', False), 'Activate only once per process'
    assert bool((env.episode_length_buf == 0).all()), 'Requires genuinely fresh native episodes'
    assert bool((env._first_timeout_deadline == env.max_episode_length).all())
    before = dict(age=env.episode_length_buf.clone(), qpos=env.sim.data.qpos.clone(),
        qvel=env.sim.data.qvel.clone(), origins=env.scene.env_origins.clone(),
        **{'obs_'+key:value.clone() for key,value in env.obs_buf.items()})
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    seed = SEED_BASE + int(completed_updates)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    deadlines = torch.randint(1, int(env.max_episode_length)+1, env.episode_length_buf.shape,
        device=env.device, dtype=env.episode_length_buf.dtype, generator=generator)
    env._first_timeout_deadline.copy_(deadlines)
    env._first_timeout_active.fill_(True)
    env._first_timeout_phase_armed = True
    actual = dict(age=env.episode_length_buf,qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,
        origins=env.scene.env_origins,**{'obs_'+key:value for key,value in env.obs_buf.items()})
    assert all(torch.equal(actual[key],value) for key,value in before.items()), 'Activation changed actual state/observations'
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng:
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    digest=lambda x:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return dict(status='PASS',recipe=RECIPE,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=seed,inherited_completed_updates=int(completed_updates),initial_native_age_zero=True,
        initial_state_and_observation_bitwise_unchanged=True,global_cpu_cuda_rng_bitwise_unchanged=True,
        captured_state_sha256={key:digest(value) for key,value in before.items()},
        deadline_sha256=digest(deadlines),deadline_histogram=torch.bincount(deadlines.long(),minlength=int(env.max_episode_length)+1).cpu().tolist())
