"""Validate the exact P04 S0/O0/R1 archive before constructing sensor training.

An absent archive is an error, never permission to substitute ideal state or
independent Gaussian noise. Importing stock tasks does not load this archive.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


# Paths and hashes from common3000/software-acceptance/computer-software-admission.json.
FILES = {
    "measurement/fit-joint-residual/model.json": "ac648d005c1f51619bcc4774e7aab3a88038e1f7cbb86f8b8cb8824600b3b598",
    "measurement/fit-joint-residual/residual-library.npz": "821e0b927ed6a09854ec4b62c78b0ecf021df706125bd2f376fbc545e38c50cb",
    "measurement/fit-joint-residual/manifest.json": "e8f78e9f289ea69cb2a21d94093977c419498f128f70f7190453af41fb7b0cb5",
    "measurement/R1-frozen-v1/model.json": "0bee6ab37fbf9b7b6b26fe1149508e3f76034f0afbfec229d5c8e99fe24d3eda",
    "measurement/R1-frozen-v1/manifest.json": "9924f41b31da05a337030b37d7e9dc3aeaca4b688c62c6525bf04b153f334f3b",
    "observer/qualification/O1-native100-seed42.npz": "312b877d33dbc643036aa99970ebfc491e1659bf1c45cc65dfa06da212260629",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class TrainingArchive:
    root: Path
    paths: dict[str, Path]
    measurement: dict
    randomization: dict
    observer_mapping: tuple[float, float]

    @classmethod
    def load(cls, root):
        root = Path(root).expanduser().resolve()
        release = (root / "training-assets/index.json").is_file()
        paths = {
            name: (
                root
                / (
                    "software-acceptance/observer/O1-native100-seed42.npz"
                    if name.startswith("observer/")
                    else "training-assets/" + name
                )
                if release
                else root / name
            )
            for name in FILES
        }
        missing = [name for name in FILES if not paths[name].is_file()]
        if missing:
            raise FileNotFoundError(
                "S0/O0/R1 training requires the original P04 research archive. "
                "Missing under " + str(root) + ": " + ", ".join(missing)
            )
        for name, expected in FILES.items():
            if digest(paths[name]) != expected:
                raise ValueError("P04 artifact SHA256 mismatch: " + name)
        measurement = json.loads(paths["measurement/fit-joint-residual/model.json"].read_text())
        randomization = json.loads(paths["measurement/R1-frozen-v1/model.json"].read_text())
        if (
            measurement["schema"] != "sc0090-causal-native-register-model-v1"
            or measurement["S0"]["width_deg"] != 0
            or measurement["residual_block_frames"] != 32
            or measurement["load_is_torque"]
        ):
            raise ValueError("Unexpected S0 measurement contract")
        conditions = randomization["conditions"]
        if (
            randomization["schema"] != "sc0090-R1-v1"
            or not randomization["fit_parameters_immutable"]
            or len(conditions) != 36
            or {row["voltage_v"] for row in conditions} != {10.2, 11.1, 12.4}
            or any((row["P"], row["D"], row["I"]) != (15, 30, 0) for row in conditions)
        ):
            raise ValueError("Unexpected frozen R1 operating conditions")
        with np.load(
            paths["observer/qualification/O1-native100-seed42.npz"], allow_pickle=False
        ) as weights:
            mapping = weights["mapping"].reshape(-1)
            if mapping.shape != (2,) or not np.isfinite(mapping).all() or mapping[1] <= 0:
                raise ValueError("Invalid frozen O0 affine mapping")
        return cls(root, paths, measurement, randomization, tuple(map(float, mapping)))

    @property
    def residual_path(self):
        return self.paths["measurement/fit-joint-residual/residual-library.npz"]

    @property
    def fit_path(self):
        return self.paths["measurement/fit-joint-residual/model.json"].parent

    @property
    def randomization_path(self):
        return self.paths["measurement/R1-frozen-v1/model.json"].parent

    @property
    def weight_path(self):
        return self.paths["observer/qualification/O1-native100-seed42.npz"]

    def evidence(self):
        return {
            "identity": "P04-S0-O0-R1",
            "files": dict(FILES),
            "neural_observer": False,
            "motor_coefficients_changed": False,
            "bench_pendulum_loads_applied_to_robot": False,
        }
