"""Compare an actual V2 ONNX and command trajectories with the compiled firmware core.

This operates on local files and a host shared library, never on a device.
Run after the firmware host build; ONNX keeps its observation normalizer.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from mjlab_microduck.sc0090_v2.feedback import WholeRobotFeedback


class Joint(ct.Structure):
    """Portable md_joint_config_t ABI; model loading is separately checked by the C verifier."""

    _fields_ = (
        [("id", ct.c_uint8), ("bus", ct.c_uint8), ("calibrated", ct.c_bool)]
        + [
            (name, ct.c_float)
            for name in (
                "zero_count",
                "radians_per_count",
                "min_rad",
                "max_rad",
                "max_rate",
            )
        ]
        + [("min_count", ct.c_uint16), ("max_count", ct.c_uint16)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.firmware.resolve() / "esp32p4/tools"))
    from convert_policy import export
    from verify_policy import Policy, verify
    from v2_contract import RUNTIME, JOINTS

    args.report.parent.mkdir(parents=True, exist_ok=True)
    binary = args.report.with_suffix(".mdp")
    manifest = export(args.onnx, binary)
    if manifest.get("runtime") != RUNTIME:
        raise ValueError("This validation requires a V2 export")
    lib = ct.CDLL(str(args.library.resolve()))
    lib.md_policy_load.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(Policy)]
    lib.md_policy_load.restype = ct.c_bool
    lib.md_policy_run.argtypes = [ct.POINTER(Policy)] + [ct.c_void_p] * 3
    lib.md_policy_run.restype = ct.c_bool
    lib.md_policy_run_packed.argtypes = [ct.POINTER(Policy)] + [ct.c_void_p] * 8
    lib.md_policy_run_packed.restype = ct.c_bool
    inference = verify(lib, args.onnx, binary)
    lib.md_v2_joint_target.argtypes = (
        [ct.POINTER(Joint), ct.c_uint] + [ct.c_float] * 4 + [ct.POINTER(ct.c_uint16)]
    )
    lib.md_v2_joint_target.restype = ct.c_bool
    lib.md_v2_target_angle.argtypes = [ct.POINTER(Joint), ct.c_uint16]
    lib.md_v2_target_angle.restype = ct.c_float

    root = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/custom_v2"
    model = mujoco.MjModel.from_xml_path(str(root / "assets/robot_walk.xml"))
    names = JOINTS.split(",")
    names.insert(9, "mouth")
    limits = torch.tensor(
        np.array([model.joint(name).range for name in names]), dtype=torch.float32
    )
    maps = [
        Joint(
            j % 5 + 1,
            j // 5,
            True,
            512,
            0.005624596222247351,
            *limits[j].tolist(),
            0.25,
            20,
            1001,
        )
        for j in range(15)
    ]
    torch.set_num_threads(2)
    feedback = WholeRobotFeedback(root / "training-assets", 1, device="cpu")
    generator = torch.Generator().manual_seed(20261015)
    targets = torch.randn((500, 15), generator=generator) * 3
    maximum_angle_error = 0.0
    samples = 0
    for rate in (None, 0.5, 5.0):
        previous = feedback.physical_target(feedback.counts(torch.zeros(1, 15)))
        native_previous = previous[0].tolist()
        for desired in targets:
            wanted = desired.clamp(limits[:, 0], limits[:, 1])[None]
            if rate is not None:
                wanted = previous + (wanted - previous).clamp(-rate * 0.02, rate * 0.02)
            wanted = wanted.clamp(limits[:, 0], limits[:, 1])
            expected = feedback.counts(wanted)
            previous = feedback.physical_target(expected)
            for joint in range(15):
                count = ct.c_uint16()
                assert lib.md_v2_joint_target(
                    ct.byref(maps[joint]),
                    joint,
                    float(desired[joint]),
                    native_previous[joint],
                    rate or 0,
                    0.02,
                    ct.byref(count),
                )
                assert count.value == int(expected[0, joint]), (
                    rate,
                    samples,
                    joint,
                    count.value,
                    int(expected[0, joint]),
                )
                native_previous[joint] = lib.md_v2_target_angle(
                    ct.byref(maps[joint]), count.value
                )
                maximum_angle_error = max(
                    maximum_angle_error,
                    abs(native_previous[joint] - float(previous[0, joint])),
                )
                samples += 1
    assert maximum_angle_error < 1e-6
    result = dict(
        passed=True,
        hardware_validated=False,
        inference=inference,
        runtime=RUNTIME,
        training_contract_sha256=manifest["training_contract_sha256"],
        command_samples=samples,
        native_count_mismatches=0,
        decoded_target_max_error_rad=maximum_angle_error,
        tested_slew_rad_s=[None, 0.5, 5.0],
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        limitation="Host numeric and simulated command parity only; no physical timing or walking validation.",
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
