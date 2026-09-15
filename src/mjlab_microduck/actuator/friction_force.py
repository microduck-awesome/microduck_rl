"""Extract the same BAM friction forces without scattering inactive zeros."""

import mujoco
import torch
import warp as wp


_FRICTION_DOF = wp.constant(int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))


@wp.kernel(enable_backward=False)
def _extract_friction(
    types: wp.array2d(dtype=wp.int32),
    ids: wp.array2d(dtype=wp.int32),
    forces: wp.array2d(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float32),
    valid_ids: wp.array(dtype=wp.int32),
):
    world, row = wp.tid()
    if row < counts[world] and types[world, row] == _FRICTION_DOF:
        dof = ids[world, row]
        if 0 <= dof and dof < output.shape[1]:
            wp.atomic_add(output, world, dof, forces[world, row])
        else:
            wp.atomic_min(valid_ids, 0, 0)


def friction_force(types, ids, forces, counts, nv):
    """Keep the original mask and addition; omit only inactive zero entries.

    MuJoCo creates at most one DOF-friction row for each DOF. Atomic addition
    also preserves the upstream behavior for repeated indices. Stale rows past
    nefc (including invalid IDs and NaN forces) must never reach the output.
    Caller selects this implementation only for the supported CUDA dtypes.
    """
    result = torch.zeros((types.shape[0], nv), device=forces.device, dtype=forces.dtype)
    valid_ids = torch.ones(1, device=forces.device, dtype=torch.int32)
    stream = wp.stream_from_torch(torch.cuda.current_stream(forces.device))
    wp.launch(_extract_friction, dim=types.shape,
              inputs=[wp.from_torch(types), wp.from_torch(ids), wp.from_torch(forces),
                      wp.from_torch(counts), wp.from_torch(result), wp.from_torch(valid_ids)], stream=stream)
    # Match scatter_add's asynchronous device-side index guard. Never permit
    # a corrupt active constraint ID to turn into an out-of-bounds write.
    torch._assert_async(valid_ids, "SC0090 friction constraint DOF index out of bounds")
    return result
