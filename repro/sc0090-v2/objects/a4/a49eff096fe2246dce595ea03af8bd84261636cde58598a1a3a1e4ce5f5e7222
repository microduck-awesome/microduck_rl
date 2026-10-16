"""One CUDA graph for a complete 5 ms S0/O0 measurement tick and query."""
import torch


class FeedbackGraph:
    def __init__(self, feedback):
        self.owner = feedback
        self.graph = None
        self.eager_calls = 0
        self.replays = 0
        self.capture_count = 0

    def advance(self, q, v):
        owner = self.owner
        # Eager initialization and capture must see the same ordinary storage.
        # Incoming advanced-indexed physics views can be inference tensors;
        # switching their tensor kind at capture would trigger compilation.
        if not hasattr(self, 'q'):
            with torch.inference_mode(False):
                self.q, self.v = torch.empty_like(q), torch.empty_like(v)
        self.q.copy_(q)
        self.v.copy_(v)
        if not q.is_cuda or not torch.is_inference_mode_enabled() or self.eager_calls < 2:
            owner._advance_eager(self.q, self.v)
            self.eager_calls += 1
            return
        if self.graph is None:
            # Mutable stream state is ordinary storage; capture bookkeeping must
            # also stay outside inference mode so future learner captures work.
            with torch.inference_mode(False), torch.no_grad():
                graph = torch.cuda.CUDAGraph()
                graph.register_generator_state(owner.pipeline.rng)
                with torch.cuda.graph(graph):
                    owner._advance_eager(self.q, self.v)
            self.graph = graph
            self.capture_count += 1
        self.graph.replay()
        self.replays += 1
