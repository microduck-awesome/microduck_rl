"""Batched causal register frontend, 100Hz features and completed-output query.

Port of the isolated FP32 firmware stream. Native registers, request/end/available
times and completed command/PID evidence are inputs; no physical truth is accepted.
"""

import math

import torch

from mjlab_microduck.sc0090.observer import FrozenObserver


class CausalStream:
    @staticmethod
    def _divide(numerator, denominator):
        return numerator / denominator

    def __init__(self, num_envs, weight_path, mode, *, device='cpu', zero=506.,
                 radians_per_count=.005624596222247351, count_limits=(20, 1001), q_limits=(-2., 2.),
                 spike_rejection=True, o0_local_history=False):
        self.n = num_envs
        self.device = device
        self.zero = zero
        self.scale = radians_per_count
        self.count_limits = count_limits
        # Command safety margins do not make native readback counts invalid.
        # The SC0090 present-position register is an unsigned 10-bit value.
        self.raw_count_limits = (0, 1023)
        self.q_limits = q_limits
        self.spike_rejection = spike_rejection
        if o0_local_history and mode != 'O0':
            raise ValueError('O1 requires its full feature history')
        self.o0_local_history = o0_local_history
        if num_envs < 1 or not (.0001 <= abs(radians_per_count) <= .02) or not (0 <= zero <= 1023):
            raise ValueError('Invalid stream count calibration')
        if not (20 <= count_limits[0] < count_limits[1] <= 1001) or not q_limits[0] < q_limits[1]:
            raise ValueError('Invalid stream physical/count bounds')
        self.model = FrozenObserver(weight_path, mode).to(device)
        z = lambda *shape, dtype=torch.float32: torch.zeros((num_envs, *shape), dtype=dtype, device=device)
        self.gate_t = z(3, dtype=torch.int64)
        self.gate_q = z(3)
        self.gate_count = z(dtype=torch.int64)
        self.gate_last = z(dtype=torch.int64)
        self.gate_replaced = z(dtype=torch.bool)
        self.gate_initialized = z(dtype=torch.bool)
        # Native raw ring: end, available, q, speed, load, voltage, status, sequence.
        self.raw_t = z(8, dtype=torch.int64)
        self.raw_available = z(8, dtype=torch.int64)
        self.raw_q = z(8)
        self.raw_registers = z(8, 3)
        self.raw_status = z(8, dtype=torch.int64)
        self.raw_sequence = z(8, dtype=torch.int64)
        self.sequence = z(dtype=torch.int64)
        self.raw_count = z(dtype=torch.int64)
        self.source_fault = z(dtype=torch.int64)
        self.command_t = z(8, dtype=torch.int64)
        self.command_values = z(8, 3)
        self.pid_t = z(2, dtype=torch.int64)
        self.pid_values = z(2, 4)
        self.sample_t = z(5, dtype=torch.int64)
        self.sample_q = z(5)
        self.samples = z(dtype=torch.int64)
        self.history = z(19, 10)
        self.frames = z(dtype=torch.int64)
        self.last_tick = z(dtype=torch.int64)
        self.seen_sequence = z(dtype=torch.int64)
        self.seen_fault = z(dtype=torch.int64)
        self.rewarm_until = z(dtype=torch.int64)
        self.output_t = z(8, dtype=torch.int64)
        self.output_available = z(8, dtype=torch.int64)
        self.output_values = z(8, 2)
        self.output_count = z(dtype=torch.int64)
        self.front_status = z(dtype=torch.int64)
        self.reason = z(dtype=torch.int64)
        self.last_features = z(10, 10)
        self.line_q3 = z()
        self.resets = z(dtype=torch.int64)
        self._rows = torch.arange(num_envs, device=device)

    def _tensor(self, value, dtype=torch.int64):
        value = torch.as_tensor(value, dtype=dtype, device=self.device)
        return value.expand(self.n) if value.ndim == 0 else value

    def _append(self, buffer, value, mask):
        # All buffers are chronological; partial worlds update independently.
        shifted = torch.cat([buffer[:, 1:], value.unsqueeze(1)], dim=1)
        buffer.copy_(torch.where(mask.reshape(self.n, *([1]*(buffer.ndim-1))), shifted, buffer))

    def _clear_derived(self, mask, *, preserve_fault=False):
        for buffer in [self.sample_t, self.sample_q, self.samples, self.history, self.frames,
                       self.last_tick, self.seen_sequence, self.output_t, self.output_available,
                       self.output_values, self.output_count, self.last_features, self.line_q3]:
            buffer.masked_fill_(mask.reshape(self.n, *([1]*(buffer.ndim-1))), 0)
        if not preserve_fault:
            self.seen_fault.masked_fill_(mask, 0)
            self.rewarm_until.masked_fill_(mask, 0)
        self.resets += mask

    def reset(self, env_ids=None):
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[slice(None) if env_ids is None else env_ids] = True
        self._clear_derived(mask)
        for buffer in [self.gate_t, self.gate_q, self.gate_count, self.gate_last,
                       self.gate_replaced, self.gate_initialized, self.raw_t, self.raw_available, self.raw_q,
                       self.raw_registers, self.raw_status, self.raw_sequence, self.sequence,
                       self.raw_count, self.source_fault, self.command_t, self.command_values,
                       self.pid_t, self.pid_values, self.front_status, self.reason]:
            buffer[mask] = 0

    def command(self, time_us, goal_count, enabled=True, confirmed=True, *, mask=None):
        time = self._tensor(time_us)
        goal = self._tensor(goal_count, torch.float32)
        enabled = self._tensor(enabled, torch.float32)
        confirmed = self._tensor(confirmed, torch.float32)
        keep = (time > 0) & (time > self.command_t[:, -1]) & (goal >= 0) & (goal <= 1023)
        if mask is not None:
            keep &= mask
        self._append(self.command_t, time, keep)
        self._append(self.command_values, torch.stack([goal, enabled, confirmed], -1), keep)
        return keep

    def pid(self, time_us, kp, kd=30, ki=0, valid=True, *, mask=None):
        time = self._tensor(time_us)
        p, d, i, good = [self._tensor(v, torch.float32) for v in [kp, kd, ki, valid]]
        keep = (time > 0) & (time > self.pid_t[:, -1])
        if mask is not None:
            keep &= mask
        self._append(self.pid_t, time, keep)
        self._append(self.pid_values, torch.stack([p, d, i, good], -1), keep)
        bad = keep & ((good == 0) | (p < 5) | (p > 30) | (d != 30) | (i != 0))
        self.source_fault.copy_(torch.where(bad, torch.maximum(self.source_fault, time), self.source_fault))
        return keep

    def read(self, start_us, end_us, available_us, count, speed, load, voltage, valid=True, *, mask=None):
        start, end, available = [self._tensor(t) for t in [start_us, end_us, available_us]]
        count, speed, load, voltage = [self._tensor(v, torch.float32) for v in [count, speed, load, voltage]]
        valid = self._tensor(valid, torch.bool)
        active = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.gate_initialized |= active
        ordered = (end > self.gate_last) & (end > 0)
        q = (count-self.zero)*abs(self.scale)
        # A successful completed transaction is not invalid merely because
        # it took longer than a physics step. Availability, freshness and
        # causal query support are checked against their actual timestamps.
        good = valid & (start > 0) & (start <= end)
        good &= torch.isfinite(q) & (count >= self.raw_count_limits[0]) & (count <= self.raw_count_limits[1])
        dt = torch.where(self.gate_last > 0, end-self.gate_last, 0)
        gap = dt > 15000
        t, h = self.gate_t, self.gate_q
        slope = torch.stack([self._divide(h[:, 1]-h[:, 0],(t[:, 1]-t[:, 0]).clamp_min(1).float()*1e-6),
                             self._divide(h[:, 2]-h[:, 1],(t[:, 2]-t[:, 1]).clamp_min(1).float()*1e-6),
                             self._divide(h[:, 2]-h[:, 0],(t[:, 2]-t[:, 0]).clamp_min(1).float()*1e-6)], -1).median(-1).values
        prediction = h[:, 2]+slope*dt.float()*1e-6
        threshold = math.radians(8.)+2.*slope.abs()*dt.float()*1e-6
        anomaly = (self.gate_count == 3) & ~gap & ((q-prediction).abs() > threshold)
        anomaly &= (q-h[:, 2]).abs() > math.radians(8.)
        if not self.spike_rejection:
            # Explicit diagnostic ablation: preserve native validity, bounds,
            # timing, LS/query and all S0 samples. Only disable the predictor's
            # replacement/rejection of otherwise valid raw position samples.
            anomaly = torch.zeros_like(anomaly)
        persistent = anomaly & self.gate_replaced
        replaced = anomaly & ~persistent
        processed = torch.where(replaced, prediction, q)
        status = torch.where(replaced, 1, 0)
        status = torch.where(persistent, 4, status)
        # Completion jitter alone cannot invalidate a successful read.
        # Reset the local spike predictor on a large interval; query age and
        # explicit read failures determine whether causal features are usable.
        status = torch.where(~good, -1, status)
        status = torch.where(~ordered, -4, status)
        accepted = active & ordered & good
        self.gate_count.copy_(torch.where(active & ordered & (~good | gap | persistent), 0, self.gate_count))
        self._append(self.gate_t, end, accepted)
        self._append(self.gate_q, processed, accepted)
        self.gate_count.copy_(torch.where(accepted, (self.gate_count+1).clamp_max(3), self.gate_count))
        self.gate_replaced.copy_(torch.where(active & ordered, good & replaced, self.gate_replaced))
        self.gate_last.copy_(torch.where(active & ordered, end, self.gate_last))
        keep = active & (end > 0) & (end > self.raw_t[:, -1]) & (available >= end)
        keep &= available >= self.raw_available[:, -1]
        self.sequence += keep
        for buffer, value in [(self.raw_t, end), (self.raw_available, available),
                              (self.raw_q, torch.where((status == 0) | (status == 1), processed, 0.)),
                              (self.raw_registers, torch.stack([speed, load, voltage], -1)),
                              (self.raw_status, status), (self.raw_sequence, self.sequence)]:
            self._append(buffer, value, keep)
        self.raw_count.copy_(torch.where(keep, (self.raw_count+1).clamp_max(8), self.raw_count))
        bad = active & (~keep | ((status != 0) & (status != 1)))
        self.source_fault.copy_(torch.where(bad, torch.maximum(self.source_fault, available), self.source_fault))
        self.front_status.copy_(torch.where(active, torch.where(keep, status, -99), self.front_status))
        return processed, self.front_status

    def _last(self, times, cutoff):
        index = torch.arange(times.shape[1], device=self.device).expand(self.n, -1)
        return torch.where((times > 0) & (times <= cutoff[:, None]), index, -1).max(-1).values

    def _select(self, buffer, index):
        return buffer[self._rows, index.clamp_min(0)]

    @staticmethod
    def _least_squares(sample_t, sample_q, grid):
        slopes, positions = [], []
        for n in [3, 5]:
            relative = (sample_t[:, -n:]-sample_t[:, -n, None]).double()*1e-6
            centered = relative-relative.mean(-1, keepdim=True)
            values = sample_q[:, -n:].double()
            slope = (centered*values).sum(-1)/centered.square().sum(-1).clamp_min(1e-12)
            line = values.mean(-1)+slope*((grid-sample_t[:, -n]).double()*1e-6-relative.mean(-1))
            slopes.append(slope.float()); positions.append(line.float())
        return torch.stack(slopes,-1), torch.stack(positions,-1)

    def tick(self, grid_us, entered_us=None, completed_us=None, *, mask=None):
        grid = self._tensor(grid_us)
        wall = grid if entered_us is None else self._tensor(entered_us)
        completed = wall+1000 if completed_us is None else self._tensor(completed_us)
        active = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        irregular = (self.last_tick > 0) & (grid-self.last_tick != 10000)
        fault = (self.source_fault > 0) & (self.source_fault <= wall) & (self.source_fault != self.seen_fault)
        oldest = self.raw_sequence[:, 0]
        eviction = (self.seen_sequence > 0) & (oldest > self.seen_sequence+1)
        irregular &= active
        fault &= active
        eviction &= active
        reset = irregular | fault | eviction
        self._clear_derived(reset, preserve_fault=True)
        rewarm_us = 0 if self.o0_local_history else 220000
        self.rewarm_until.copy_(torch.where(irregular | eviction, wall+rewarm_us, self.rewarm_until))
        self.seen_fault.copy_(torch.where(fault, self.source_fault, self.seen_fault))
        self.rewarm_until.copy_(torch.where(fault, self.source_fault+rewarm_us, self.rewarm_until))
        self.last_tick.copy_(torch.where(active, grid, self.last_tick))
        read_seen = self._last(self.raw_available, wall)
        self.seen_sequence.copy_(torch.where(active, torch.maximum(self.seen_sequence, torch.where(read_seen >= 0,
            self._select(self.raw_sequence, read_seen), 0)), self.seen_sequence))
        rp, cp, pp, current_pp = [self._last(t, cutoff) for t, cutoff in
            [(self.raw_available, grid), (self.command_t, grid), (self.pid_t, grid), (self.pid_t, wall)]]
        command = self._select(self.command_values, cp)
        pid = self._select(self.pid_values, pp)
        current_pid = self._select(self.pid_values, current_pp)
        read_t = self._select(self.raw_t, rp)
        read_available = self._select(self.raw_available, rp)
        read_status = self._select(self.raw_status, rp)
        read_q = self._select(self.raw_q, rp)
        registers = self._select(self.raw_registers, rp)
        command_t = self._select(self.command_t, cp)
        current_bad = (current_pp >= 0) & ((current_pid[:, 3] == 0) | (current_pid[:, 0] < 5) |
            (current_pid[:, 0] > 30) | (current_pid[:, 1] != 30) | (current_pid[:, 2] != 0))
        pid_missing = (pp < 0) | (pid[:, 3] == 0)
        pid_bad = (pid[:, 0] < 5) | (pid[:, 0] > 30) | (pid[:, 1] != 30) | (pid[:, 2] != 0)
        command_missing = (cp < 0) | (grid-command_t > 25000) | (command[:, 0] > 1023)
        unconfirmed = command[:, 2] == 0
        read_missing = (rp < 0) | (grid-read_t < 0) | (grid-read_t > 25000)
        front_fault = ((read_status != 0) & (read_status != 1)) | ((self.seen_fault > 0) & (read_available <= self.seen_fault))
        contract = (grid <= 0) | (grid > wall) | ~self.gate_initialized
        reason = torch.ones(self.n, dtype=torch.int64, device=self.device)
        # Later assignments represent earlier rejection branches in the C source.
        for bad, value in [(front_fault, 13), (read_missing, 5), (unconfirmed, 8),
                           (command_missing, 7), (pid_bad, 4), (pid_missing, 3),
                           (current_bad, 4), (current_bad & (current_pid[:, 3] == 0), 3), (contract, 10)]:
            reason = torch.where(bad, value, reason)
        clear = active & (contract | pid_missing | pid_bad | current_bad | command_missing | unconfirmed | read_missing)
        self._clear_derived(clear)
        good = active & ~clear & ~front_fault
        self._append(self.sample_t, read_t, good)
        self._append(self.sample_q, read_q, good)
        self.samples.copy_(torch.where(good, (self.samples+1).clamp_max(5), self.samples))
        have_ls = good & (self.samples == 5)
        slopes, positions = self._least_squares(self.sample_t,self.sample_q,grid)
        self.line_q3.copy_(torch.where(active, positions[:,0], self.line_q3))
        frame = torch.stack([read_q, (command[:, 0]-self.zero)*abs(self.scale), registers[:, 0], registers[:, 1],
            (grid-read_t).float()*1e-6, command[:, 1], pid[:, 0], slopes[:,0], slopes[:,1], registers[:, 2]], -1)
        self._append(self.history, frame, have_ls)
        self.frames.copy_(torch.where(have_ls, (self.frames+1).clamp_max(19), self.frames))
        self.last_features.copy_(torch.where(active[:, None, None], self.history[:, ::2], self.last_features))
        output, numeric = self.model(self.last_features, self.line_q3)
        # O0 uses only the present LS3 estimate and derivative. It does not use
        # the 19-frame O1 network history. Keep the existing five-sample LS
        # preparation and every packet, command, timestamp and numeric guard.
        history_ready = torch.ones_like(have_ls) if self.o0_local_history else self.frames == 19
        candidate = have_ls & history_ready & (grid >= self.rewarm_until)
        numeric &= (completed >= wall) & (completed >= grid) & (output[:, 1].abs() <= 40)
        numeric &= (output[:, 0] >= self.q_limits[0]-.02) & (output[:, 0] <= self.q_limits[1]+.02)
        invalid_output = candidate & ~numeric
        reason = torch.where(invalid_output, 11, reason)
        self._clear_derived(invalid_output)
        ready = candidate & numeric
        self._append(self.output_t, grid, ready)
        self._append(self.output_available, completed, ready)
        self._append(self.output_values, output, ready)
        self.output_count.copy_(torch.where(ready, (self.output_count+1).clamp_max(8), self.output_count))
        reason = torch.where(ready, 2, reason)
        self.reason.copy_(torch.where(active, reason, self.reason))
        return output, ready

    def query(self, query_us, wall_us):
        query, wall = self._tensor(query_us), self._tensor(wall_us)
        eligible = (self.output_t > 0) & (self.output_available <= wall[:, None])
        index = torch.arange(8, device=self.device).expand(self.n, -1)
        newest = torch.where(eligible, index, -1).max(-1).values
        left = torch.where(eligible & (self.output_t <= query[:, None]), index, -1).max(-1).values
        right = torch.where(eligible & (self.output_t >= query[:, None]), index, 8).min(-1).values
        rt = self._select(self.output_t, right.clamp_max(7))
        lt = self._select(self.output_t, left)
        newest_t = self._select(self.output_t, newest)
        newest_available = self._select(self.output_available, newest)
        span = rt-lt
        valid = (newest >= 0) & (left >= 0) & (right < 8) & (query > 0)
        valid &= (wall >= newest_available) & (wall-newest_available <= 50000)
        valid &= (newest_t >= query) & (newest_t-query <= 50000) & (span >= 0) & (span <= 25000)
        alpha = self._divide((query-lt).float(),span.clamp_min(1).float())
        lv = self._select(self.output_values, left)
        rv = self._select(self.output_values, right.clamp_max(7))
        output = lv+(rv-lv)*alpha[:, None]
        valid &= torch.isfinite(output).all(-1)
        return torch.where(valid[:, None], output, float('nan')), valid
