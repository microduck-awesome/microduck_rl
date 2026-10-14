"""Fifteen independent S0/O0 streams with one supply per assembled robot.

R1 load labels select empirical sensor donors only. CAD bodies and frozen M6
coefficients are never replaced by the identification pendulum's mass/inertia.
"""

import torch

from mjlab_microduck.sc0090.registers import RegisterPipeline
from .artifacts import TrainingArchive


class WholeRobotFeedback:
    def __init__(self, archive_path, num_envs, *, device, seed=42, stage="nominal", joints=15):
        if stage not in {"nominal", "measured"}:
            raise ValueError("Expected nominal or measured R1 stage")
        self.archive = TrainingArchive.load(archive_path)
        self.num_envs, self.joints, self.device = num_envs, joints, device
        self.stage = stage
        self.pipeline = RegisterPipeline(
            self.archive.fit_path,
            self.archive.weight_path,
            num_envs * joints,
            "S0",
            "O0",
            device=device,
            seed=seed,
            randomization_path=self.archive.randomization_path,
        )
        # Synthetic register origin, not a hardware calibration. Both directions
        # use this origin; each real joint must supply its own signed count map.
        self.pipeline.stream.zero = 512.0
        self.pipeline.stream.q_limits = (-3.0, 3.0)
        self.clock = torch.full((num_envs,), 1_280_000, dtype=torch.int64, device=device)
        self.voltage = torch.full((num_envs,), 11.1, device=device)
        self.rng = torch.Generator(device=device).manual_seed(seed + 731)
        self.reset_attempts = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.last_q = torch.zeros((num_envs, joints), device=device)
        self.last_v = torch.zeros_like(self.last_q)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _lane_ids(self, env_ids):
        return (
            env_ids[:, None] * self.joints + torch.arange(self.joints, device=self.device)
        ).flatten()

    def _time(self):
        return self.clock.repeat_interleave(self.joints)

    def counts(self, physical_target):
        stream = self.pipeline.stream
        mapping = stream.model.mapping
        # The CAD range can exceed the finite native register range (head yaw).
        # These are deployment count guards, not motor speed/torque parameters.
        count = (physical_target - mapping[0]) / mapping[1] / stream.scale + stream.zero
        return torch.round(count.clamp(*stream.count_limits))

    def physical_target(self, counts):
        stream = self.pipeline.stream
        return (
            stream.model.mapping[0]
            + stream.model.mapping[1] * (counts - stream.zero) * stream.scale
        )

    def physical_goal_for_motor(self, counts, feedback_bias, feedback_scale):
        """Translate the completed native goal to the BAM robot-coordinate API."""
        nominal = (counts - self.pipeline.stream.zero) * self.pipeline.stream.scale
        return (nominal - feedback_bias) / feedback_scale

    def _conditions(self, ids):
        p = self.pipeline
        voltages = torch.tensor(
            [11.1] if self.stage == "nominal" else [10.2, 11.1, 12.4], device=self.device
        )
        selected = torch.randint(len(voltages), (len(ids),), generator=self.rng, device=self.device)
        self.voltage[ids] = voltages[selected]
        lanes = self._lane_ids(ids)
        wanted = self.voltage[ids].repeat_interleave(self.joints)
        # Uniform among measured donor groups at the robot's shared voltage.
        matching = p.condition_voltages[None] == wanted[:, None]
        scores = torch.rand(matching.shape, generator=self.rng, device=self.device)
        conditions = scores.masked_fill(~matching, -1).argmax(-1)
        p.condition_index[lanes] = conditions
        p.voltage[lanes] = wanted
        p.voltage_mode[lanes] = p.condition_modes[conditions]
        phase = torch.randint(256, (len(lanes),), generator=self.rng, device=self.device)
        p.episode_phase[lanes] = p.condition_phases[conditions, phase]

    def reset(self, ids, pose):
        """Warm measured streams during a synthetic held-pose reset prehistory.

        As in the source bench task, this is a reset initial condition, outside
        the scored rollout. It does not pretend to settle a free-standing robot.
        Every actor joint value still comes through S0/O0, including at reset.
        """
        self._conditions(ids)
        p = self.pipeline
        mask = torch.zeros(p.n, dtype=torch.bool, device=self.device)
        mask[self._lane_ids(ids)] = True
        q = pose.flatten().clone()
        zeros = torch.zeros_like(q)
        goal = self.counts(pose).flatten()
        nominal_goal = (goal - p.stream.zero) * p.stream.scale
        remaining = mask.clone()
        for _ in range(8):
            lanes = remaining.nonzero().flatten()
            if not len(lanes):
                break
            p.reset(lanes)
            p.physics[lanes, :, 0] = q[lanes, None]
            p.physics[lanes, :, 2] = q[lanes, None]
            p.play_state[lanes] = q[lanes]
            p.physical_context[lanes, :, 0] = q[lanes, None]
            p.physical_context[lanes, :, 2] = 1.0
            p.physical_context[lanes, :, 3] = nominal_goal[lanes, None]
            p.stream.command_values[lanes, -1, 0] = goal[lanes]
            self.reset_attempts[ids] += remaining.reshape(self.num_envs, self.joints)[ids].any(-1)
            for elapsed in range(0, 280_001, 5000):
                time = 1_000_000 + elapsed
                if 0 < elapsed < 280_000 and elapsed % 20_000 == 0:
                    p.stream.command(time, goal, mask=remaining)
                p.advance(time, q, zeros, mask=remaining)
            _, valid = p.query(1_280_000)
            remaining &= ~valid
        if remaining.any():
            raise RuntimeError("S0/O0 held-pose warmup failed; no ideal-state fallback")
        self.clock[ids] = 1_280_000
        self.refresh()

    def command(self, counts):
        keep = self.pipeline.command(self._time(), counts.flatten())
        # A reset already seeded this same instant with its held target. Replace
        # that seed before simulation starts; later commands are strictly ordered.
        same = self.pipeline.stream.command_t[:, -1] == self._time()
        seed = same & (self._time() == 1_280_000)
        if seed.any():
            self.pipeline.stream.command_values[seed, -1, 0] = counts.flatten()[seed]
        if not (keep | seed).all():
            raise RuntimeError("Invalid or non-monotonic simulated joint command")

    def advance(self, q, v):
        self.pipeline.advance(self._time(), q.flatten(), v.flatten())
        self.clock += 5000
        self.refresh()

    def refresh(self):
        estimate, valid = self.pipeline.query(self._time())
        estimate = estimate.reshape(self.num_envs, self.joints, 2)
        self.ready = valid.reshape(self.num_envs, self.joints).all(-1)
        # Invalid lanes are retained only until the environment's fault
        # termination resets them. They never receive privileged q/v substitutes.
        good = valid.reshape(self.num_envs, self.joints)
        self.last_q = torch.where(good, estimate[..., 0], self.last_q)
        self.last_v = torch.where(good, estimate[..., 1], self.last_v)
