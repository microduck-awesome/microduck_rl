"""Control isolation, reset history, deployment contract and live CPU rollouts."""
import importlib.util
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sc0090_server', ROOT / 'scripts/sc0090_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def message(seq, client='a', **values):
    return dict(client=client, seq=seq, twist=[.1, 0, 0], **values)


def test_reordered_commands_and_timeout_cannot_restore_motion():
    box = server.Mailbox()
    assert box.submit(message(2), 1.)
    assert not box.submit(message(1), 1.1)
    assert box.consume(1.2)[0]['twist'] == [.1, 0, 0]
    assert box.consume(1.5)[0]['twist'] == [0, 0, 0]
    assert not box.submit(message(1), 1.6)


def test_browser_lease_and_expired_event():
    box = server.Mailbox()
    assert box.submit(message(1, event='push'), 1.)
    assert not box.submit(message(1, client='b'), 1.1)
    assert box.consume(1.5)[1] == []
    assert box.submit(message(1, client='b', event='supine'), 1.6)
    assert box.consume(1.7)[1] == ['supine']
    assert box.consume(1.8)[1] == []


@pytest.mark.parametrize('twist', [[float('nan'), 0, 0], [float('inf'),0,0], [0,0], ['bad',0,0]])
def test_invalid_input_never_reaches_simulation(twist):
    box = server.Mailbox()
    with pytest.raises((ValueError, TypeError)):
        box.submit(dict(client='a', seq=1, twist=twist), 1.)
    assert box.owner is None


def test_concurrent_posts_keep_latest_sequence():
    box = server.Mailbox()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda seq: box.submit(message(seq), 1.), range(100)))
    assert box.seq == 99
    assert not box.submit(message(98, event='push'), 1.1)
    assert box.consume(1.2)[1] == []


def test_checkpoint_selection_is_queued_once():
    box = server.Mailbox()
    assert box.submit(message(1, load={'slot':'walking','id':'known-id'}),1.)
    assert box.consume(1.1)[1] == [{'load':{'slot':'walking','id':'known-id'}}]
    assert box.consume(1.2)[1] == []
    with pytest.raises(ValueError):
        box.submit(message(2, load={'slot':'other','id':'bad'}),1.2)


def test_exploration_lease_timeout_and_new_browser_stop_autonomy():
    options=dict(id='run', seed=1, duration=2, recovery_timeout=12)
    box = server.Mailbox()
    box.submit(message(1, explore=options), 1.)
    assert box.consume(1.1)[0]['explore'] == options
    assert box.consume(1.5)[0]['explore'] is None
    box.submit(message(1, client='b'), 1.6)
    assert box.consume(1.7)[0]['explore'] is None


@pytest.fixture(scope='module')
def demo():
    if not (ROOT / 'models/manifest.json').is_file():
        pytest.skip('Export local policies before running the CPU rollout tests')
    paths = server.default_models()
    return server.Demo(*paths)


def test_nominal_training_physics_and_joint_contract(demo):
    import mjlab.tasks  # Register plugins before importing robot constants.
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    from mjlab_microduck.actuator.sc0090 import SC0090_MAX_SPEED_RPM
    import re
    assert demo.model.opt.timestep == .005
    assert demo.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert (demo.model.opt.iterations, demo.model.opt.ls_iterations) == (10,20)
    assert SC0090_MAX_SPEED_RPM == 80
    assert demo.motor.model.actuator.max_speed_rad_s == pytest.approx(80*2*np.pi/60)
    for index, name in enumerate(demo.joint_names):
        reference = next(value for pattern,value in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern,name))
        assert demo.policy.default_pose[index] == np.float32(reference)


def test_reset_clears_policy_and_motor_history(demo):
    demo.reset('supine')
    for _ in range(30):
        demo.step([0,0,0])
    demo.reset('supine')
    first = []
    for _ in range(15):
        demo.step([0,0,0]); first.append(demo.data.qpos.copy())
    demo.reset('supine')
    second = []
    for _ in range(15):
        demo.step([0,0,0]); second.append(demo.data.qpos.copy())
    np.testing.assert_array_equal(first, second)


def test_switch_preserves_last_action_and_zeroes_recovery_twist(demo):
    demo.reset('standing')
    demo.mode = 'walking'
    demo.step([.03,0,0])
    assert demo.policy.command[0] == np.float32(.03)  # No low-speed switch deadzone.
    old = demo.policy.last_action.copy()
    demo.mode = 'recovery'
    captured = []
    original = demo.policy.infer
    def infer():
        captured.append(demo.policy.get_observations().copy())
        return original()
    demo.policy.infer = infer
    try:
        demo.step([.3,0,0])
    finally:
        demo.policy.infer = original
    np.testing.assert_array_equal(captured[0][34:48], old)
    np.testing.assert_array_equal(captured[0][48:51], 0)


@pytest.mark.parametrize('pose', server.POSES)
def test_live_recovery_spawns_are_finite_and_stand(demo, pose):
    demo.reset(pose)
    demo.mode = 'auto'
    for _ in range(400):
        demo.step([0,0,0])
    state = demo.status()
    assert state['height'] >= .105 and state['tilt'] <= 20
    assert state['active'] == 'walking'
    if pose != 'standing':
        assert .5 <= state['rise_time'] <= 8.


def test_nonfinite_action_stops_before_physics(demo):
    demo.reset('standing')
    original = demo.policy.infer
    demo.policy.infer = lambda: np.full(14, np.nan)
    try:
        with pytest.raises(FloatingPointError, match='action'):
            demo.step([0,0,0])
        assert demo.data.time == 0.
    finally:
        demo.policy.infer = original


def test_push_stays_horizontal_on_a_fallen_body(demo):
    demo.reset('left_side')
    va = int(demo.model.joint('trunk_base_freejoint').dofadr[0])
    before = demo.data.qvel[va:va+3].copy()
    demo.push()
    delta = demo.data.qvel[va:va+3] - before
    assert delta[2] == 0
    assert np.linalg.norm(delta[:2]) == pytest.approx(.2)


def test_rejected_policy_load_preserves_current_model_and_state(demo, tmp_path):
    demo.reset('standing')
    demo.step([.1,0,0])
    old_session = demo.sessions['walking']
    old_state = demo.data.qpos.copy()
    bad = tmp_path / 'bad.onnx'
    bad.write_bytes(b'not an ONNX model')
    with pytest.raises(Exception):
        demo.load_policy('walking',bad)
    assert demo.sessions['walking'] is old_session
    np.testing.assert_array_equal(demo.data.qpos,old_state)


def test_live_exploration_sequence_uses_real_recovery_and_bounded_commands(demo):
    planner = server.Exploration()
    planner.sync(dict(id='live',seed=1,duration=2,recovery_timeout=12))
    demo.reset('standing'); demo.mode = 'auto'
    seen, resets = set(), set()
    for _ in range(4500):
        twist,event = planner.step(demo.status(), server.CONTROL_DT, .1, .6)
        if planner.action:
            seen.add(planner.action)
        if event in server.POSES:
            resets.add(event); demo.reset(event)
        elif event == 'push':
            demo.push()
        assert planner.active, planner.status()
        demo.step(twist)
        if planner.count >= 17:
            break
    assert len(seen) == 16 and resets == set(server.POSES)-{'standing'}
    assert len(planner.history) <= 6
