"""The comparison instrument must reproduce the existing Pages execution."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_official_gait import Probe
from sc0090_server import Demo, RL_REPO, default_models, rehearsal


def test_probe_matches_pages_through_startup_and_walking():
    walking, recovery = default_models()
    probe = Probe(dict(
        policy=str(walking), scene=str(RL_REPO / rehearsal.MICRODUCK_XML),
        motor='sc0090', kp=rehearsal.BAM_KP_FW, vin=12., vin_min=10.,
        supply_resistance_ohm=.1, action_scale=1., velocity_obs_lag=1,
        head_alpha=1., legs_alpha=1.))
    demo = Demo(walking, recovery)
    demo.mode = 'walking'
    for _ in range(150):  # Three seconds includes the measured walking window.
        probe.step(.1)
        demo.step([.1, 0., 0.])
        np.testing.assert_array_equal(probe.data.qpos, demo.data.qpos)
        np.testing.assert_array_equal(probe.data.qvel, demo.data.qvel)
        np.testing.assert_array_equal(probe.policy.last_action, demo.policy.last_action)
        before = probe.policy.get_observations()
        probe.frame()
        probe.frame()
        np.testing.assert_array_equal(before, probe.policy.get_observations())
