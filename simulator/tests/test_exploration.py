from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from exploration import Exploration, MOVES, RECOVERIES, validate_request


def request(**values):
    return dict(id='run-1', seed=1, duration=2, recovery_timeout=3, **values)


def state(**values):
    return dict(dict(paused=False, error=None, recovering=False, stable=True, time=10.), **values)


def cycle(seed):
    planner = Exploration()
    options = request()
    options.update(seed=seed)
    planner.sync(options)
    actions, commands = [], []
    for _ in range(2000):
        # Arbitrary MuJoCo time resets must not alter scheduler timing.
        twist, event = planner.step(state(time=0), .1, .3, 1.)
        commands.append(twist)
        if planner.count > len(actions):
            actions.append((planner.action, event))
        if len(actions) == len(MOVES) + 1 + len(RECOVERIES):
            break
    return actions, commands


def test_seed_replays_complete_action_bag_and_speed_bounds():
    first = cycle(1)
    assert first == cycle(1)
    assert first[0] != cycle(2)[0]
    assert {action for action,event in first[0]} == set(MOVES) | set(RECOVERIES) | {'push'}
    for vx,vy,yaw in first[1]:
        assert abs(vx) <= .3 and abs(vy) <= .15 and abs(yaw) <= 1.
    # Stale browser tabs cannot disable the mandatory recovery segments.
    assert validate_request(request(recoveries=False)) == request()


def test_fall_waits_for_standing_before_next_action_and_does_not_reset():
    planner = Exploration(); planner.sync(request())
    assert planner.step(state(recovering=True,stable=False), .02, .1, .6) == ([0.,0.,0.],None)
    for _ in range(10):
        assert planner.step(state(recovering=True,stable=False), .02, .1, .6) == ([0.,0.,0.],None)
    assert planner.count == 0 and planner.waiting
    planner.step(state(), .02, .1, .6)
    assert not planner.waiting and planner.active
    planner.step(state(), .02, .1, .6)
    assert planner.count == 1


def test_timeout_keeps_failure_latched_until_explicit_restart():
    planner = Exploration(); options = request(); planner.sync(options)
    for _ in range(5):
        twist,event = planner.step(state(recovering=True,stable=False), 1., .1, .6)
        assert not any(twist) and event is None
    assert not planner.active and planner.failed and planner.history[-1]['outcome'] == '未在时限内稳定'
    planner.sync(options)
    assert not planner.active
    planner.sync(None)
    assert planner.failed
    planner.sync(options)
    assert planner.active and not planner.failed


def test_pause_and_stop_do_not_advance_or_replay_pose_events():
    planner = Exploration(); options = request(); planner.sync(options)
    planner.step(state(paused=True), 20., .1, .6)
    assert planner.count == 0 and planner.elapsed == 0
    planner.step(state(), .02, .1, .6)
    planner.stop('manual')
    count = planner.count
    planner.sync(options)
    assert planner.step(state(), 20., .1, .6) == ([0.,0.,0.],None)
    assert planner.count == count


def test_command_duration_covers_the_complete_physics_interval():
    planner = Exploration(); planner.sync(request())
    for _ in range(100):
        twist,event = planner.step(state(), .02, .1, .6)
        assert any(twist) and event is None and planner.count == 1
    assert planner.elapsed == pytest.approx(2.)
    assert planner.step(state(), .02, .1, .6) == ([0.,0.,0.],None)
    assert planner.history[-1]['outcome'] == '指令段结束'


@pytest.mark.parametrize('key,value', [('seed',-1),('seed',True),('duration',float('nan')),
                                      ('recovery_timeout',float('inf')),('seed',2**32),('id','')])
def test_invalid_schedule_is_rejected(key,value):
    options=request(); options[key]=value
    with pytest.raises(ValueError):
        validate_request(options)
