"""Known synthetic motion checks the units used to judge training candidates."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_gait import segments, summarize


def reference_trace():
    dt = .02
    t = np.arange(500) * dt
    # One stride per second per foot; half-cycle phase offset. The finite
    # window includes 9 complete left touchdowns and 10 right touchdowns.
    phase = np.arange(500)[:, None] + np.array([0, 25])[None, :]
    contact = phase % 50 < 30
    foot_position = np.zeros((len(t), 2, 3))
    foot_position[:, :, 0] = .1 * t[:, None]
    omega = np.sin(2 * np.pi * 5 * t)[:, None] * np.array([3., 4., 0.])
    return dt, dict(foot_normal_force=contact.astype(float), foot_position=foot_position,
                   forward=np.tile([1., 0., 0.], (len(t), 1)),
                   velocity=np.tile([.1, 0.], (len(t), 1)),
                   head_angular_velocity=omega, torso_angular_velocity=np.zeros_like(omega),
                   neck_velocity=np.zeros((len(t), 4)), neck_action=np.zeros((len(t), 4)),
                   head_relative_position=np.zeros((len(t), 3)))


def test_physical_units_and_complete_strides():
    dt, trace = reference_trace()
    result = summarize(trace, dt)
    assert result['touchdowns_per_s'] == pytest.approx(1.9)
    assert result['median_same_foot_stride_m'] == pytest.approx(.1)
    assert result['median_flight_s'] == pytest.approx(.4)
    assert result['head_angular_speed_rms_rad_s'] == pytest.approx(5 / np.sqrt(2))
    assert result['head_angular_velocity_peak_hz'] == pytest.approx(5)
    assert result['head_angular_velocity_power_above_6hz_fraction'] < 1e-20


def test_window_edges_and_stationary_motion():
    assert segments(np.array([True, False, True, True, False, True])) == [(2, 4)]
    dt, trace = reference_trace()
    trace['foot_normal_force'][:] = 1
    trace['head_angular_velocity'][:] = 0
    result = summarize(trace, dt)
    assert result['touchdowns_per_s'] == 0
    assert result['median_same_foot_stride_m'] is None
    assert result['median_flight_s'] is None
    assert result['head_angular_velocity_peak_hz'] is None
    assert result['head_angular_speed_rms_rad_s'] == 0
