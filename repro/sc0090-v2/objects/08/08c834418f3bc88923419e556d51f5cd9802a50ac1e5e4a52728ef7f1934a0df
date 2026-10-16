"""Actual applied head-command exposure, accumulated on the training GPU."""
import torch


class CommandExposure:
    def __init__(self, env):
        self.env = env
        self.classes = torch.tensor([0,1,2,4], device=env.device)
        self.combinations = torch.zeros(4, device=env.device, dtype=torch.int64)
        self.positive = torch.zeros(4, device=env.device, dtype=torch.int64)
        self.negative = torch.zeros(4, device=env.device, dtype=torch.int64)
        self.range_quarters = torch.zeros(4,4, device=env.device, dtype=torch.int64)
        self.steps = 0
        self.ranges = None

    def record(self):
        term = self.env.command_manager.get_term('head_pose')
        command = term.command
        ranges = tuple(term.cfg.ranges)
        if ranges != self.ranges:
            self.ranges = ranges
            self.scale = torch.tensor([max(abs(a),abs(b)) for a,b in ranges],device=self.env.device).clamp_min(1e-9)
        active = command != 0
        count = active.sum(-1)
        self.combinations += (count[:,None] == self.classes).sum(0)
        self.positive += (command > 0).sum(0)
        self.negative += (command < 0).sum(0)
        bucket = (command.abs()/self.scale*4).long().clamp_max(3)
        self.range_quarters += ((bucket[:,:,None] == torch.arange(4,device=self.env.device))
                               & active[:,:,None]).sum(0)
        self.steps += self.env.num_envs

    def report(self):
        return dict(environment_control_steps=self.steps, combinations=self.combinations.tolist(),
            combination_axes=[0,1,2,4], positive=self.positive.tolist(),negative=self.negative.tolist(),
            range_quarters=self.range_quarters.tolist(),current_ranges=self.ranges)
