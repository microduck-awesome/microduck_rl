"""Replay the exact held-pose reset prehistory with a persistent CUDA graph.

Two real ticks initialize kernels; no invisible warmup ticks or RNG draws.
The same graph handles all 57 timestamps, commands and retry masks. Compact
workspaces are bucketed by capacity so changing reset counts cannot recapture
the normal rollout graph, or force a new reset graph on every control step.
"""
import torch


class ResetPrehistory:
    @torch.inference_mode(False)
    def __init__(self, pipeline, chunk_ticks=1):
        if chunk_ticks not in (1, 19):
            raise ValueError('Reset graph chunk must be 1 or 19 ticks')
        self.pipeline = pipeline
        self.chunk_ticks = chunk_ticks
        self.q = torch.zeros(pipeline.n, device=pipeline.device)
        self.v = torch.zeros_like(self.q)
        self.goal = torch.zeros_like(self.q)
        self.confirmed = torch.ones_like(self.q)
        self.mask = torch.zeros(pipeline.n, device=pipeline.device, dtype=torch.bool)
        self.time = torch.zeros((), device=pipeline.device, dtype=torch.int64)
        self.graphs = {}
        self.eager_calls = self.captures = self.replays = 0
        self.tick_replays = 0

    @torch.inference_mode(False)
    def _tick(self):
        command_mask = self.mask & (self.time > 1_000_000) & (self.time < 1_280_000)
        command_mask &= self.time % 20_000 == 0
        self.pipeline.stream.command(self.time, self.goal, self.confirmed, self.confirmed,
                                     mask=command_mask)
        self.pipeline.advance(self.time, self.q, self.v, mask=self.mask)
        self.time.add_(5000)

    def run(self, q, goal, mask):
        self.q.copy_(q)
        self.goal.copy_(goal)
        self.mask.copy_(mask)
        self.time.fill_(1_000_000)
        remaining = 57
        while remaining:
            if self.eager_calls < 2:
                self._tick()
                self.eager_calls += 1
                remaining -= 1
                continue
            count = self.chunk_ticks if remaining >= self.chunk_ticks else 1
            graph = self.graphs.get(count)
            if graph is None:
                with torch.inference_mode(False), torch.no_grad():
                    graph = torch.cuda.CUDAGraph()
                    graph.register_generator_state(self.pipeline.rng)
                    with torch.cuda.graph(graph):
                        for _ in range(count):
                            self._tick()
                self.graphs[count] = graph
                self.captures += 1
            graph.replay()
            self.replays += 1
            self.tick_replays += count
            remaining -= count
