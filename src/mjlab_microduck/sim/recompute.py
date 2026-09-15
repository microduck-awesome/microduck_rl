"""Reuse the pinned mjlab constant-recompute pipeline without changing physics."""

import mujoco_warp as mjwarp
import warp as wp
from mjlab.managers.event_manager import RecomputeLevel


def prepare_recompute_storage(sim):
    """Keep reset-derived statistics independent across randomized worlds.

    mjlab expands top-level fields declared by the startup event. The nested
    meaninertia statistic needs separate handling: the pinned Warp recompute
    kernel writes it per world and the solver reads it for convergence scaling.
    A singleton with stride zero would make all worlds race on one value.
    """
    current = sim.wp_model.stat.meaninertia
    if sim.num_envs > 1 and (current.shape[0] != sim.num_envs or current.strides[0] == 0):
        with wp.ScopedDevice(sim.wp_device):
            sim.wp_model.stat.meaninertia = wp.full(
                sim.num_envs, sim.mj_model.stat.meaninertia, dtype=float)
        sim._model_bridge.clear_cache()
        sim.create_graph()


class RecomputeGraphCache:
    """Instance-local CUDA graphs, invalidated alongside mjlab's own graphs.

    Domain randomization updates array contents in place. Replaying the same
    computation therefore reads the new masses/inertias on every reset. Model
    field expansion replaces arrays and calls create_graph; invalidate here as
    well so a captured pointer can never refer to an old model allocation.
    """

    def __init__(self, sim):
        self.sim = sim
        self.graphs = {}
        self._original_recompute = sim.recompute_constants
        self._original_create_graph = sim.create_graph

    def create_graph(self):
        self.graphs.clear()
        return self._original_create_graph()

    def recompute(self, level):
        if (not self.sim.use_cuda_graph
                or level not in (RecomputeLevel.set_const_0, RecomputeLevel.set_const)):
            return self._original_recompute(level)
        with wp.ScopedDevice(self.sim.wp_device):
            if level == RecomputeLevel.set_const:
                # This pinned Warp version reads body_gravcomp back to the
                # CPU to update the Python ngravcomp count. Preserve that
                # behavior outside capture, then graph the expensive pipeline.
                mjwarp.set_const_fixed(self.sim.wp_model, self.sim.wp_data)
            graph = self.graphs.get(level)
            if graph is None:
                # Capture records operations without running them. Launch once
                # below, including on the first call: no duplicate DR update.
                with wp.ScopedCapture() as capture:
                    mjwarp.set_const_0(self.sim.wp_model, self.sim.wp_data)
                graph = self.graphs[level] = capture.graph
            wp.capture_launch(graph)


def install_recompute_graph_cache(sim):
    """Install once on this simulation; leave CPU/non-graph execution intact."""
    cache = getattr(sim, "_microduck_recompute_cache", None)
    if cache is None and sim.use_cuda_graph:
        cache = RecomputeGraphCache(sim)
        sim._microduck_recompute_cache = cache
        sim.recompute_constants = cache.recompute
        sim.create_graph = cache.create_graph
    return cache
