"""Seeded demo command scheduling; called only by the simulation owner thread.

Durations use simulation time. No policy output, motor parameter or physics state
is changed here: explicit pose resets use the same events as the manual buttons.
"""
import random

MOVES = ('forward', 'backward', 'left', 'right', 'turn_left', 'turn_right',
         'curve_left', 'curve_right', 'slow_forward', 'idle')
RECOVERIES = ('sitting', 'prone', 'supine', 'left_side', 'right_side')
LABELS = dict(forward='前进', backward='后退', left='向左侧移', right='向右侧移',
              turn_left='向左转', turn_right='向右转', curve_left='向左绕弯',
              curve_right='向右绕弯', slow_forward='低速前进', idle='停步站立',
              push='侧向扰动', sitting='坐姿', prone='俯卧', supine='仰卧',
              left_side='左侧卧', right_side='右侧卧')


def validate_request(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Invalid exploration request')
    if not isinstance(value.get('id'), str) or not 1 <= len(value['id']) <= 80:
        raise ValueError('Invalid exploration ID')
    if type(value.get('seed')) is not int or not 0 <= value['seed'] < 2**32:
        raise ValueError('Exploration seed must be a uint32')
    for key, minimum, maximum in (('duration', 2, 10), ('recovery_timeout', 2, 60)):
        number = value.get(key)
        if type(number) not in (float, int) or not minimum <= number <= maximum:
            raise ValueError(f'Invalid exploration {key}')
    return {key: value[key] for key in ('id', 'seed', 'duration', 'recovery_timeout')}


class Exploration:
    def __init__(self):
        self.request = None
        self.active = False
        self.label = '手动控制'
        self.action = None
        self.history = []
        self.count = 0
        self.elapsed = 0.
        self.waiting = False
        self.failed = False

    def sync(self, request):
        if request == self.request:
            return  # A failed run stays stopped until an explicit new request.
        self.request = dict(request) if request is not None else None
        if request is None:
            if self.active:
                self.stop('自由探索已停止')
            return
        self.rng = random.Random(request['seed'])
        self.bag = []
        self.active, self.failed = True, False
        self.action, self.waiting = None, False
        self.count, self.elapsed, self.history = 0, 0., []
        self.label = '准备开始'

    def stop(self, reason, failed=False):
        self.active, self.failed = False, failed
        self.label = reason

    def finish(self, outcome):
        self.history.append({'number': self.count, 'action': self.action,
                             'label': self.label, 'outcome': outcome})
        self.history = self.history[-6:]
        self.action, self.waiting, self.elapsed = None, False, 0.

    def step(self, state, dt, speed, turn):
        if not self.active or state['paused']:
            return [0., 0., 0.], None
        if state['error']:
            self.stop('数值异常，探索已停止', failed=True)
            return [0., 0., 0.], None
        if self.waiting:
            self.elapsed += dt
            if not state['recovering'] and state['stable']:
                self.finish('稳定站立')
            elif self.elapsed >= self.request['recovery_timeout']:
                reason = f'{self.label}：等待 {self.request["recovery_timeout"]:g} 秒仍未稳定，已停止'
                self.finish('未在时限内稳定')
                self.stop(reason, failed=True)
            return [0., 0., 0.], None
        if state['recovering']:
            self.waiting, self.elapsed = True, 0.
            self.label = '意外跌倒，等待模型起身'
            return [0., 0., 0.], None
        if self.action is None:
            if not self.bag:
                self.bag = list(MOVES) + ['push'] + list(RECOVERIES)
                self.rng.shuffle(self.bag)
            self.action = self.bag.pop()
            self.count += 1
            self.elapsed = 0.
            self.label = LABELS[self.action]
            if self.action in RECOVERIES or self.action == 'push':
                self.waiting = True
                self.label = ('重置为' + self.label + '，等待模型起身' if self.action in RECOVERIES
                              else '侧向扰动，等待稳定')
                return [0., 0., 0.], self.action
            self.fraction = self.rng.uniform(.35, 1.)
        if self.elapsed + 1e-9 >= self.request['duration']:
            self.finish('指令段结束')
            return [0., 0., 0.], None
        self.elapsed += dt
        # Speed sliders remain the upper bounds; these are command changes,
        # never joint-action interpolation or motor/physics modifications.
        speed *= self.fraction
        turn *= self.fraction
        twists = dict(forward=[speed,0,0], backward=[-speed,0,0],
                      left=[0,min(speed,.15),0], right=[0,-min(speed,.15),0],
                      turn_left=[0,0,turn], turn_right=[0,0,-turn],
                      curve_left=[speed,0,turn/2], curve_right=[speed,0,-turn/2],
                      slow_forward=[min(speed,.03),0,0], idle=[0,0,0])
        return twists[self.action], None

    def status(self):
        return dict(active=self.active, failed=self.failed, label=self.label, action=self.action,
                    number=self.count, elapsed=self.elapsed, history=list(self.history),
                    request_id=self.request['id'] if self.request else None)
