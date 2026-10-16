"""Training-sealed full register/event generator for free physical simulation.

Physical q/v are private inputs to S. O receives quantized nominal position,
learned delayed/filtered raw speed, empirical raw load, native voltage bytes and
completed transaction times. Correlated residual blocks come from training only.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .stream import CausalStream


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RegisterPipeline:
    @staticmethod
    def _divide(numerator, denominator):
        return numerator / denominator

    @staticmethod
    def _calibrated_dot(basis, coefficients):
        return basis @ coefficients

    def __init__(self, fit_path, weight_path, num_envs, S, O, *, group='L10_M070', kp=15,
                 voltage=11.1, device='cpu', seed=42, random_phase=False,randomization_path=None,
                 stream_class=CausalStream):
        if S not in {'S0', 'S1'}:
            raise ValueError('Expected S0/S1 identity')
        self.n, self.device, self.S = num_envs, device, S
        root = Path(fit_path)
        self.model = json.loads((root/'model.json').read_text())
        library = root/'residual-library.npz'
        if file_digest(library) != self.model['residual_library_sha256']:
            raise ValueError('Frozen training residual library changed')
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['training_records'] != 848 or manifest['development_read']:
            raise ValueError('S must use the original training-only calibration')
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.stream = stream_class(num_envs, weight_path, O, device=device)
        self.group, self.kp, self.voltage = group, kp, voltage
        self.random_phase = random_phase
        self.randomization=None
        if randomization_path is not None:
            randomization_path=Path(randomization_path)
            identity=json.loads((randomization_path/'manifest.json').read_text())
            if identity['development_read'] or identity['training_records']!=848 or file_digest(randomization_path/'model.json')!=identity['model_sha256']:
                raise ValueError('R1 operating conditions must be frozen from training only')
            self.randomization=json.loads((randomization_path/'model.json').read_text())
            if not self.randomization['fit_parameters_immutable'] or self.randomization['primary_D']!='sc0090-v5-m6-seed1':
                raise ValueError('R1 must preserve the frozen v5 M6 parameters')
            self.condition_rng=torch.Generator(device=device).manual_seed(seed+91019)
            self.phase_rng=torch.Generator(device=device).manual_seed(seed+91020)
        self.voltage_mode = next(r['mode'] for r in self.model['voltage_register'] if r['condition'] == [voltage, kp, group])
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
        with np.load(library, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            offsets, good = archive['offsets'], archive['valid']
            matching = [i for i, r in enumerate(metadata) if r['group'] == group and r['kp'] == kp and r['voltage_v'] == voltage]
            candidates = []
            for i in matching:
                a, b = offsets[i:i+2]
                starts = np.arange(a, b-31)
                keep = (np.convolve(good[a:b].astype(int), np.ones(32, int), 'valid') == 32) & (starts >= a+18)
                candidates.extend(starts[keep])
            if not candidates:
                raise ValueError('No matched continuous training donor blocks for the requested condition')
            self.pool = tensor(candidates, torch.int64)
            self.residual = tensor(archive['residuals'])
            self.context = tensor(archive['contexts'])
            self.duration_us = tensor((archive['duration_ns']+999)//1000, torch.int64)
            if self.randomization is not None:
                conditions=self.randomization['conditions'];pools=[]
                for condition in conditions:
                    pool=[]
                    for i,r in enumerate(metadata):
                        if (r['group'],r['kp'],r['voltage_v'])!=(condition['group'],kp,condition['voltage_v']):continue
                        a,b=offsets[i:i+2];starts=np.arange(a,b-31)
                        keep=(np.convolve(good[a:b].astype(int),np.ones(32,int),'valid')==32)&(starts>=a+18)
                        pool.extend(starts[keep])
                    if not pool:raise ValueError('R1 condition has no matched continuous training blocks')
                    pools.append(pool)
                width=max(map(len,pools));padded=np.zeros((len(pools),width),dtype=np.int64)
                for i,pool in enumerate(pools):padded[i,:len(pool)]=pool
                self.condition_pools=tensor(padded,torch.int64);self.condition_lengths=tensor(list(map(len,pools)),torch.int64)
                self.condition_candidate_count = min(1024, min(map(len, pools)))
                initial=next(i for i,r in enumerate(conditions) if (r['group'],r['voltage_v'])==(group,voltage))
                self.condition_index=torch.full((num_envs,),initial,dtype=torch.int64,device=device)
                self.condition_settings=tensor([r['settings'] for r in conditions]);self.condition_mass=tensor([r['mass_kg'] for r in conditions])
                self.condition_phases=tensor([r['phase_us'] for r in conditions],torch.int64)
                phases=torch.randint(256,(num_envs,),generator=self.phase_rng,device=device)
                self.episode_phase=self.condition_phases[self.condition_index,phases].clone()
                modes=[next(r['mode'] for r in self.model['voltage_register'] if r['condition']==[c['voltage_v'],kp,c['group']]) for c in conditions]
                self.condition_voltages=tensor([r['voltage_v'] for r in conditions]);self.condition_modes=tensor(modes)
                self.voltage=self.condition_voltages[self.condition_index].clone();self.voltage_mode=self.condition_modes[self.condition_index].clone()
        self.coefficients = tensor(self.model[S]['coefficients'])
        self.load_coefficients = tensor(self.model['load_coefficients'])
        self.position_residual_coefficients = tensor(self.model.get('position_residual_coefficients', np.zeros((13, 2))))[:, int(S == 'S1')]
        self.play_width = math.radians(self.model[S]['width_deg'])
        self.lag_us = int(self.model[S]['readout_lag_ms']*1000)
        self.speed_tau = self.model['native_speed']['tau_s']
        self.speed_lag_us = int(self.model['native_speed']['lag_s']*1e6)
        self.speed_slope, self.speed_bias = self.model['native_speed']['slope_bias']
        z = lambda *shape, dtype=torch.float32: torch.zeros((num_envs, *shape), dtype=dtype, device=device)
        self.physics_t = z(64, dtype=torch.int64)
        self.physics = z(64, 3)
        self.play_state = z()
        self.physical_context = z(19, 4)
        self.donor_start = z(dtype=torch.int64)
        self.donor_offset = z(dtype=torch.int64)
        self.next_start = z(dtype=torch.int64)
        self.pending_start = z(dtype=torch.int64)
        self.pending_end = z(dtype=torch.int64)
        self.pending_donor = z(dtype=torch.int64)
        self.pending = z(dtype=torch.bool)
        self.last_speed_time = z(dtype=torch.int64)
        self.speed_filtered = z()
        self.clock = z(dtype=torch.int64)
        self.event_count = z(dtype=torch.int64)
        self.last_raw = z(7)
        self.last_read_t = z(3, dtype=torch.int64)
        self.reset()

    def reset(self, env_ids=None, *, epoch_us=1_000_000):
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[slice(None) if env_ids is None else env_ids] = True
        self.stream.reset(env_ids)
        # The episode starts after a controlled stationary hold at physical zero.
        # These private S histories describe that hold, never actor observations.
        self.physics_t[mask] = epoch_us+torch.arange(-64, 0, device=self.device)*5000
        self.physics[mask] = 0.
        self.play_state[mask] = 0.
        self.physical_context[mask] = 0.
        self.donor_offset[mask] = 32
        self.pending[mask] = False
        self.pending_start[mask] = self.pending_end[mask] = 0
        self.clock[mask] = epoch_us-5000
        self.last_speed_time[mask] = 0
        self.speed_filtered[mask] = 0.
        self.event_count[mask] = 0
        if self.randomization is not None:
            phase=self.episode_phase
        else:
            phase = torch.randint(0, 10000, (self.n,), generator=self.rng, device=self.device) if self.random_phase else torch.full((self.n,), 2000, device=self.device)
        self.next_start[mask] = epoch_us+phase[mask]
        self.last_raw[mask] = 0.
        self.last_read_t[mask] = 0
        zero_goal = int(round(self.stream.zero+(0.-float(self.stream.model.mapping[0])) /
                              float(self.stream.model.mapping[1])/self.stream.scale))
        self.stream.command(epoch_us, zero_goal, True, True, mask=mask)
        self.stream.pid(epoch_us, self.kp, mask=mask)

    def command(self, time_us, count, enabled=True, confirmed=True):
        return self.stream.command(time_us, count, enabled, confirmed)

    def sample_conditions(self,env_ids):
        if self.randomization is None:raise ValueError('R1 condition sampling was not configured')
        sampled=torch.randint(len(self.condition_lengths),(len(env_ids),),generator=self.condition_rng,device=self.device)
        self.condition_index[env_ids]=sampled
        self.voltage[env_ids]=self.condition_voltages[sampled]
        self.voltage_mode[env_ids]=self.condition_modes[sampled]
        phases=torch.randint(256,(len(env_ids),),generator=self.phase_rng,device=self.device)
        self.episode_phase[env_ids]=self.condition_phases[sampled,phases]
        return self.condition_settings[sampled],self.condition_mass[sampled]

    def _interpolate(self, queries, column):
        indices = torch.arange(64, device=self.device).expand(self.n, -1)
        left = torch.where(self.physics_t <= queries[:, None], indices, -1).max(-1).values
        right = torch.where(self.physics_t >= queries[:, None], indices, 64).min(-1).values
        rows = self.stream._rows
        li, ri = left.clamp_min(0), right.clamp_max(63)
        lt, rt = self.physics_t[rows, li], self.physics_t[rows, ri]
        alpha = self._divide((queries-lt).float(),(rt-lt).clamp_min(1).float())
        values = self.physics[rows, li, column] + alpha*(self.physics[rows, ri, column]-self.physics[rows, li, column])
        return values, (left >= 0) & (right < 64) & (queries <= self.clock)

    def _choose_blocks(self, q, v, goal, enabled, mask):
        # Every selected lane still draws 1024 independent uniform candidates
        # and minimizes the unchanged context cost. The old path gathered and
        # scored all N lanes even when only a few needed a new residual block
        # (particularly expensive during partial-reset prehistory). Compact on
        # device; never transfer sample states or candidate scores to the CPU.
        # This changes RNG consumption, not the conditional sampling law. New
        # V2 runs use this version from their first sample, with a recorded seed.
        rows = mask.nonzero().flatten()
        q, v, goal, enabled = (x[rows] for x in (q, v, goal, enabled))
        if self.randomization is None:
            count = min(1024, len(self.pool))
            random = torch.randint(len(self.pool), (len(rows), count), generator=self.rng, device=self.device)
            subset = self.pool[random]
        else:
            count=self.condition_candidate_count
            conditions=self.condition_index[rows]
            lengths=self.condition_lengths[conditions]
            random=(torch.rand((len(rows),count),generator=self.rng,device=self.device)*lengths[:,None]).long()
            subset=self.condition_pools[conditions[:,None],random]
        donor = self.context[subset]
        score = (donor[..., 0]-q[:, None]).abs()/.25+(donor[..., 1]-v[:, None]).abs()/.3
        score += 20*(donor[..., 2]-enabled[:, None]).abs()+(donor[..., 3]-goal[:, None]).abs()/.1
        for back in [4, 9, 18]:
            old = self.context[subset-back]
            target = self.physical_context[rows, -back]
            score += ((old[..., 0]-donor[..., 0])-(target[:, 0]-q)[:, None]).abs()/.1
            score += (old[..., 1]-target[:, 1, None]).abs()/.3
            score += ((old[..., 3]-donor[..., 3])-(target[:, 3]-goal)[:, None]).abs()/.1
            score += 20*(old[..., 2]-target[:, 2, None]).abs()
        selected = subset[torch.arange(len(rows), device=self.device), score.argmin(-1)]
        self.donor_start[rows] = selected
        self.donor_offset[rows] = 0

    def advance(self, time_us, q, v, *, mask=None):
        time = self.stream._tensor(time_us)
        q, v = [self.stream._tensor(value, torch.float32) for value in [q, v]]
        active = time > self.clock
        if mask is not None:
            active &= mask
        self.clock.copy_(torch.where(active, time, self.clock))
        played = torch.minimum(q+self.play_width/2, torch.maximum(q-self.play_width/2, self.play_state))
        self.play_state.copy_(torch.where(active, played, self.play_state))
        self.stream._append(self.physics_t, time, active)
        self.stream._append(self.physics, torch.stack([q, v, played], -1), active)
        request = active & ~self.pending & (self.next_start <= time+5000)
        # Select an exogenous training block before its read completes, using
        # only already simulated state and confirmed past commands.
        ci = self.stream._last(self.stream.command_t, time)
        command = self.stream._select(self.stream.command_values, ci)
        goal = (command[:, 0]-self.stream.zero)*self.stream.scale
        enabled = command[:, 1]
        choose = request & (self.donor_offset >= 32)
        if q.is_cuda and self.randomization is not None:
            from .gpu_donors import select
            select(self, q, v, goal, enabled, choose)
        elif choose.any():
            self._choose_blocks(q, v, goal, enabled, choose)
        donor_index = self.donor_start+self.donor_offset.clamp_max(31)
        self.pending_donor.copy_(torch.where(request, donor_index, self.pending_donor))
        self.pending_start.copy_(torch.where(request, self.next_start, self.pending_start))
        self.pending_end.copy_(torch.where(request, self.next_start+self.duration_us[donor_index], self.pending_end))
        self.pending |= request
        complete = active & self.pending & (self.pending_end <= time)
        end, start, residual = self.pending_end, self.pending_start, self.residual[self.pending_donor]
        truth_q, q_good = self._interpolate(end, 0)
        truth_v, v_good = self._interpolate(end, 1)
        mid = start+(end-start)//2
        sensor_q, position_good = self._interpolate(mid-self.lag_us, 2)
        delayed_v, speed_good = self._interpolate(end-self.speed_lag_us, 1)
        ci = self.stream._last(self.stream.command_t, end)
        command = self.stream._select(self.stream.command_values, ci)
        goal = (command[:, 0]-self.stream.zero)*self.stream.scale
        enabled = command[:, 1]
        alpha = torch.exp(-(end-self.last_speed_time).float()*1e-6/self.speed_tau)
        filtered = torch.where(self.last_speed_time > 0, alpha*self.speed_filtered+(1-alpha)*delayed_v, delayed_v)
        self.speed_filtered.copy_(torch.where(complete, filtered, self.speed_filtered))
        self.last_speed_time.copy_(torch.where(complete, end, self.last_speed_time))
        speed = self.speed_slope*filtered+self.speed_bias+residual[:, 2]
        error = goal-(self.model['load_coordinate_feedback'][0]+self.model['load_coordinate_feedback'][1]*truth_q)
        demand, sine = self.kp*error, torch.sin(truth_q)
        one = torch.ones_like(q)
        load_basis = torch.stack([one, enabled, truth_q, truth_v, truth_v.abs(), sine,
            demand, torch.tanh(demand), torch.tanh(demand).abs(), demand*enabled,
            truth_v*enabled, sine*enabled, one*self.voltage, one*self.kp/15.], -1)
        load = self._calibrated_dot(load_basis,self.load_coefficients)+residual[:, 3]
        register_q = self.coefficients[0]+sensor_q*(self.coefficients[1]+sensor_q*(self.coefficients[2]+sensor_q*self.coefficients[3]))
        load_scaled, speed_scaled = load/1000., speed/1000.
        basis = torch.stack([one, truth_q, truth_v, goal-truth_q, enabled, load_scaled, speed_scaled,
            load_scaled*sine, load_scaled*truth_q, torch.tanh(self.kp*(goal-truth_q)),
            truth_v.abs(), one*self.kp/15., one*self.voltage/11.1], -1)
        register_q += self._calibrated_dot(basis,self.position_residual_coefficients)+residual[:, int(self.S == 'S1')]
        count = torch.round(register_q/self.stream.scale+self.stream.zero)
        speed = torch.round(speed/self.model['speed_quantum_raw'])*self.model['speed_quantum_raw']
        load = torch.round(load.clamp(-1023, 1023))
        voltage = torch.round((self.voltage_mode+residual[:, 4]).clamp(0, 255))
        valid = q_good & v_good & position_good & speed_good & (ci >= 0)
        valid &= torch.isfinite(count) & torch.isfinite(speed) & torch.isfinite(load)
        # End is the recorded physical proxy; availability is when this
        # discrete simulator has actually constructed/filtered the registers.
        self.stream.read(start, end, time, count, speed, load, voltage, valid, mask=complete)
        raw = torch.stack([count, speed, load, voltage, valid.float(),
                           self.pending_donor.float(), self.stream.front_status.float()], -1)
        self.last_raw.copy_(torch.where(complete[:, None], raw, self.last_raw))
        self.last_read_t.copy_(torch.where(complete[:, None], torch.stack([start, end, time], -1), self.last_read_t))
        self.event_count += complete
        self.stream._append(self.physical_context, torch.stack([truth_q, truth_v, enabled, goal], -1), complete)
        self.donor_offset += complete
        self.next_start.copy_(torch.where(complete, start+10000, self.next_start))
        self.pending &= ~complete
        tick = active & (time % 10000 == 0)
        self.stream.tick(time, time, time+1000, mask=tick)

    def query(self, control_time_us):
        time = self.stream._tensor(control_time_us)
        return self.stream.query(time-20000, time)
