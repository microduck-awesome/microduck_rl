"""Read-only fault attribution for full bench P04 evaluation, off the training path."""
import numpy as np
import torch


class FeedbackFaultRecorder:
    def __init__(self, feedback):
        self.feedback = feedback
        p, s = feedback.pipeline, feedback.pipeline.stream
        self.stream = s
        n, j, device = feedback.num_envs, feedback.joints, feedback.device
        self.shape = n, j
        self.previous_source = s.source_fault.clone()
        self.previous_tick = s.last_tick.clone()
        self.source_events = torch.zeros_like(s.source_fault)
        self.numeric_events = torch.zeros_like(s.source_fault)
        self.last_numeric_time = torch.zeros_like(s.source_fault)
        self.last_source_status = torch.zeros_like(s.source_fault)
        self.last_source_raw = torch.zeros_like(p.last_raw)
        self.last_source_times = torch.zeros_like(p.last_read_t)
        self.bad_controls = torch.zeros_like(s.source_fault)
        self.first_step = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.first = {name: torch.zeros((n,j),dtype=torch.int64,device=device) for name in
            ('invalid_lanes','reason','source_status','source_time_us','numeric_time_us',
             'frames','output_count','rewarm_until_us')}
        self.first['raw'] = torch.zeros((n,j,7),device=device)
        self.first['read_times_us'] = torch.zeros((n,j,3),dtype=torch.int64,device=device)
        self.original = feedback.advance

        def advance(q, v):
            self.original(q,v)
            # These operations only read the stream and write our own buffers.
            # They do not enter the model CUDA graph or advance its private RNG.
            source = s.source_fault > self.previous_source
            self.source_events += source
            self.last_source_status.copy_(torch.where(source,s.front_status,self.last_source_status))
            self.last_source_raw.copy_(torch.where(source[:,None],p.last_raw,self.last_source_raw))
            self.last_source_times.copy_(torch.where(source[:,None],p.last_read_t,self.last_source_times))
            numeric = (s.last_tick != self.previous_tick) & (s.reason == 11)
            self.numeric_events += numeric
            self.last_numeric_time.copy_(torch.where(numeric,s.last_tick,self.last_numeric_time))
            self.previous_source.copy_(s.source_fault)
            self.previous_tick.copy_(s.last_tick)
        feedback.advance = advance

    def control_boundary(self, step):
        f, s = self.feedback, self.stream
        # query() is pure: it interpolates existing completed outputs and does
        # not append state, issue a read, run the observer or draw randomness.
        _, valid = f.pipeline.query(f._time())
        valid = valid.reshape(self.shape)
        torch._assert_async((valid.all(-1)==f.ready).all(),'Fault diagnostic/query parity')
        self.bad_controls += (~valid).flatten()
        fresh = ~f.ready & (self.first_step < 0)
        self.first_step.copy_(torch.where(fresh,step,self.first_step))
        values = dict(invalid_lanes=(~valid).long(),reason=s.reason,
            source_status=self.last_source_status,source_time_us=s.source_fault,
            numeric_time_us=self.last_numeric_time,frames=s.frames,
            output_count=s.output_count,rewarm_until_us=s.rewarm_until,
            raw=self.last_source_raw,read_times_us=self.last_source_times)
        for name, target in self.first.items():
            value=values[name].reshape(target.shape)
            target.copy_(torch.where(fresh.reshape(-1,*([1]*(target.ndim-1))),value,target))

    def save(self, path, bucket):
        np.savez_compressed(path,first_control_step=self.first_step.cpu().numpy(),
            bucket=bucket.cpu().numpy(),
            source_events=self.source_events.reshape(self.shape).cpu().numpy(),
            numeric_events=self.numeric_events.reshape(self.shape).cpu().numpy(),
            invalid_control_counts=self.bad_controls.reshape(self.shape).cpu().numpy(),
            **{name:value.cpu().numpy() for name,value in self.first.items()})

    def close(self):
        self.feedback.advance=self.original
