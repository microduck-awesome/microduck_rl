"""Validate frozen V2 ONNX semantics and the portable MDP version-two extension.

The contract hash records provenance; it is not motion approval or authentication.
Calibration and the resident model identity remain device-owned gates.
"""

import hashlib
import json
import math
from pathlib import Path
import struct

RUNTIME = "microduck-v2-sc0090-native-o0-rise-v1"
RUNTIME_ID = 0x834335F5
WALKING_RUNTIME = "microduck-v2-sc0090-native-o0-walking-v1"
WALKING_RUNTIME_ID = 0x11D720E4
RUNTIMES = {RUNTIME: RUNTIME_ID, WALKING_RUNTIME: WALKING_RUNTIME_ID}
JOINTS = (
    "left_hip_yaw,left_hip_roll,left_hip_pitch,left_knee,left_ankle,"
    "neck_pitch,head_pitch,head_yaw,head_roll,"
    "right_hip_yaw,right_hip_roll,right_hip_pitch,right_knee,right_ankle"
)


def unique(pairs):
    """Return a new object from ordered JSON pairs, rejecting duplicate keys.

    @param pairs Borrowed ordered key/value pairs from json.loads.
    @return New dictionary; raises ValueError before overwriting a duplicate.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate V2 contract key: {key}")
        result[key] = value
    return result


def number(value, low, high):
    """Check one finite non-boolean scalar against inclusive contract bounds.

    @param value Candidate numeric scalar.
    @param low Inclusive lower bound in the scalar's units.
    @param high Inclusive upper bound in the scalar's units.
    @return Original scalar or ValueError; no input mutation.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("invalid V2 action scale or command slew")
    return value


def deployment(metadata, source_sha256):
    """Bind an exact frozen V2 candidate to the shared physical experiment profile.

    Original training metadata and weights remain immutable. In particular, the baseline
    candidate is labelled as a transfer experiment, never rewritten as P04 training.
    Unknown candidates require a reviewed task/command profile, not just a matching shape.
    """
    candidate = None
    for name in ("v2-rise-candidates.json", "v2-walking-candidates.json"):
        registry = json.loads((Path(__file__).parent / "contracts" / name).read_text())
        candidate = next(
            (c for c in registry["candidates"] if c["onnx_sha256"] == source_sha256), None
        )
        if candidate is not None:
            break
    if candidate is None or metadata != candidate["source_metadata"]:
        raise ValueError("V2 requires an exact released candidate and its original metadata")
    if registry["deployment"]["runtime"] not in RUNTIMES:
        raise ValueError("unsupported V2 deployment profile")
    contract = dict(
        registry["deployment"],
        source_sha256=source_sha256,
        source_runtime=metadata["microduck_runtime"],
        experiment=candidate["experiment"],
        candidate_id=candidate["id"],
        original_training_contract_sha256=metadata.get("microduck_training_contract_sha256"),
    )
    return contract


def extension(metadata, source_sha256=None):
    """Emit a V2 deployment extension only for verified original ONNX bytes/metadata.

    @param metadata Unique original ONNX string metadata, never mutated.
    @param source_sha256 SHA256 of the same immutable bytes parsed and converted by the caller.
    @return 44 little-endian bytes, or None for legacy alpha; unknown V2 semantics raise.
    """
    runtime = metadata.get("microduck_runtime", "microduck-alpha")
    if runtime == "microduck-alpha":
        if any(key.startswith("microduck_training_contract") for key in metadata):
            raise ValueError("training contract requires an explicit supported runtime")
        return None
    value = deployment(metadata, source_sha256)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).digest()
    return struct.pack("<Iff32s", RUNTIMES[value["runtime"]], 1.0, 0.0, digest)


def info(data):
    """Decode a supported V2 MLP extension without interpreting coefficients.

    @param data Borrowed complete model bytes; full payload validation belongs to the caller.
    @return Copied runtime/target/provenance dictionary; raises ValueError for invalid framing.
    """
    if len(data) < 128 or data[:8] != b"MDPOL001" or struct.unpack_from("<I", data, 8)[0] != 2:
        raise ValueError("require MDPOL001 version two")
    runtime, scale, rate, digest = struct.unpack_from("<Iff32s", data, 84)
    if runtime not in RUNTIMES.values() or not any(digest):
        raise ValueError("invalid V2 runtime identity")
    number(scale, 1, 1)
    number(rate, 0, 0)
    return dict(
        runtime=next(name for name, identity in RUNTIMES.items() if identity == runtime),
        policy_runtime_id=runtime,
        action_scale=scale,
        command_rate_rad_s=rate or None,
        action_ema=False,
        deployment_contract_sha256=digest.hex(),
    )
