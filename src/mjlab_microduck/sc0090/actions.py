"""One explicit physical-angle/count/action contract for training and runners."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActionConfig:
    zero_count: float
    radians_per_count: float
    calibration_bias: float
    calibration_scale: float
    action_scale_rad: float
    minimum_rad: float
    maximum_rad: float
    minimum_count: int
    maximum_count: int
    maximum_rate_rad_s: float
    policy_dt_s: float = .02

    def __post_init__(self):
        if not np.isfinite(list(self.__dict__.values())).all():
            raise ValueError('Nonfinite action contract')
        if (self.radians_per_count <= 0 or self.calibration_scale <= 0 or self.action_scale_rad <= 0
                or self.minimum_rad >= self.maximum_rad or self.minimum_count >= self.maximum_count
                or self.maximum_rate_rad_s <= 0 or self.policy_dt_s <= 0):
            raise ValueError('Invalid action coordinate/bounds/rate contract')

    def physical_from_count(self, count):
        nominal = (count-self.zero_count)*self.radians_per_count
        return self.calibration_bias+self.calibration_scale*nominal

    def count_from_physical(self, q):
        nominal = (q-self.calibration_bias)/self.calibration_scale
        return int(np.rint(nominal/self.radians_per_count+self.zero_count))


class ActionController:
    """Counts are quantized and slew-limited before the confirmed target is used.

    last_raw_action is exactly the actor's previous unfiltered scalar. Applied
    goal/count are separate telemetry. No EMA or deployment-only smoothing.
    The bounds below must be frozen from training support and bench clearance.
    """

    def __init__(self, config: ActionConfig):
        self.config = config
        self.last_count = None
        self.last_raw_action = 0.0

    def reset(self, confirmed_count: int):
        c = self.config
        if not c.minimum_count <= confirmed_count <= c.maximum_count:
            raise ValueError('Reset needs a valid confirmed goal count')
        self.last_count = int(confirmed_count)
        self.last_raw_action = 0.0

    def step(self, raw_action: float):
        if self.last_count is None or not math.isfinite(raw_action):
            raise ValueError('Action history uninitialized or actor output nonfinite')
        c = self.config
        requested = c.action_scale_rad*raw_action
        physical = float(np.clip(requested, c.minimum_rad, c.maximum_rad))
        physical_min_count = math.ceil((c.minimum_rad-c.calibration_bias)/c.calibration_scale/c.radians_per_count+c.zero_count)
        physical_max_count = math.floor((c.maximum_rad-c.calibration_bias)/c.calibration_scale/c.radians_per_count+c.zero_count)
        wanted = int(np.clip(c.count_from_physical(physical), max(c.minimum_count, physical_min_count),
                             min(c.maximum_count, physical_max_count)))
        maximum_delta = math.floor(c.maximum_rate_rad_s*c.policy_dt_s/(c.calibration_scale*c.radians_per_count))
        count = int(np.clip(wanted, self.last_count-maximum_delta, self.last_count+maximum_delta))
        self.last_raw_action, self.last_count = float(raw_action), count
        nominal = (count-c.zero_count)*c.radians_per_count
        return {'raw_action': float(raw_action), 'goal_count': count, 'goal_nominal_rad': nominal,
                'goal_physical_rad': float(c.physical_from_count(count)),
                'position_limited': physical != requested, 'rate_limited': count != wanted}
