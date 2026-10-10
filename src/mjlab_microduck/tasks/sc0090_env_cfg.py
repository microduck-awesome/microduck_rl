"""Independent SC0090 bench, based on the single-axis XL330 task structure.

All actuator parameters, register units and action calibration belong to v5
SC0090. No whole-robot observation or controller conventions are reused.
"""
import json
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from . import mdp
from mjlab_microduck.sc0090.actions import ActionConfig
from mjlab_microduck.sc0090.action_term import SC0090ActionCfg
from mjlab_microduck.sc0090.backend import make_entity_cfg


WORKSPACE = Path(__file__).resolve().parents[4]
STUDY = WORKSPACE/'sc0090-datasets/v5-testbench-sim2real-20261011'
M6_PATH = WORKSPACE/'sc0090-datasets/v5-model-study/models/m6-seed1/model.json'
MAIN_SETTINGS = (.005, 15., 30., 11.1, .0008243333333333334, .0852197885, .005624596222247351, 2., 1.)
MAIN_MASS = .1003


def action_contract():
    weight = STUDY/'observer/qualification/O1-native100-seed42.npz'
    import numpy as np
    with np.load(weight, allow_pickle=False) as archive:
        mapping = archive['mapping']
    return ActionConfig(506., MAIN_SETTINGS[6], float(mapping[0]), float(mapping[1]),
                        .8, -.8, .8, 150, 850, 2.)


def make_sc0090_env_cfg(S='S1', O='O1', play=False, *, R='R0'):
    if R not in ['R0','R1']:raise ValueError('Expected explicit R0/R1 identity')
    randomization=str(STUDY/'measurement/R1-frozen-v1') if R=='R1' else None
    events={'reset': EventTermCfg(func=mdp.sc0090_reset, mode='reset')}
    if R=='R1':events['randomize']=EventTermCfg(func=mdp.sc0090_randomize,mode='reset')
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=1, terrain=None, entities={'bench': make_entity_cfg(
            str(M6_PATH), MAIN_SETTINGS, MAIN_MASS, joint_limits=(-1.6, 1.6))}),
        observations={
            'actor': ObservationGroupCfg(terms={'input': ObservationTermCfg(func=mdp.sc0090_actor)},
                concatenate_terms=True, enable_corruption=False),
            'critic': ObservationGroupCfg(terms={'input': ObservationTermCfg(func=mdp.sc0090_critic)},
                concatenate_terms=True, enable_corruption=False)},
        actions={'position': SC0090ActionCfg(entity_name='bench', contract=action_contract(),
            fit_path=str(STUDY/'measurement/fit-joint-residual'),
            weight_path=str(STUDY/'observer/qualification/O1-native100-seed42.npz'), S=S, O=O,
            randomization_path=randomization)},
        commands={'target': mdp.SC0090TargetsCfg(trajectory_seed=91012 if play else 91011)},
        events=events,
        rewards={
            'tracking': RewardTermCfg(func=mdp.sc0090_tracking, weight=3.),
            'action_rate': RewardTermCfg(func=mdp.sc0090_action_rate, weight=-.01),
            'velocity': RewardTermCfg(func=mdp.sc0090_velocity_cost, weight=-.001)},
        terminations={'time_out': TerminationTermCfg(func=base_mdp.time_out, time_out=True),
                      'fault': TerminationTermCfg(func=mdp.sc0090_fault)},
        sim=SimulationCfg(nconmax=1, njmax=1, mujoco=MujocoCfg(
            timestep=.005, integrator='euler', gravity=(0., 0., -9.80665))),
        decimation=4, episode_length_s=8.,
    )


def make_sc0090_rl_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128, 64), activation='elu', obs_normalization=False,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .5, 'std_type': 'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation='elu', obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=.005),
        experiment_name='sc0090-sim2real', logger='tensorboard', upload_model=False,
        num_steps_per_env=24, max_iterations=500, save_interval=100)
