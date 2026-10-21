#!/usr/bin/env python3
"""Strict exporter for the upstream 61 -> 512 -> 256 -> 128 -> 14 alpha MLP.

The observation normalizer remains part of the model. No graph simplification,
quantization, ignored nodes, or changes to the source ONNX are allowed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort
from v2_contract import extension, info, unique, deployment


def export(source: Path, destination: Path, *, source_bytes: bytes | None = None):
    """Validate and export one supported normalized policy graph.

    Requires exact finite 61-to-14 float contract, Sub/Div plus four Dense/three ELU nodes, graph
    connectivity, unit Gemm attributes and consumed constants. Require finite real ORT zero-input
    inference before writing either output; emits little-endian coefficients/count/CRC/sourceSHA
    without quantization or graph simplification.

    @param source Self-contained ONNX Path; external tensor files are rejected.
    @param destination Output converted model Path and adjacent manifest base.
    @param source_bytes Optional immutable ONNX snapshot; otherwise read source once. Parsing,
        source identity and ORT inference use these same bytes, including LSTM delegation.
    @return Manifest dictionary; writes MDPOL001 or delegates explicit-state LSTM conversion.
    """
    source_bytes = source.read_bytes() if source_bytes is None else source_bytes
    model = onnx.load_model_from_string(source_bytes)
    source_metadata = unique((p.key, p.value) for p in model.metadata_props)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    contract = extension(source_metadata, source_digest)
    tensors = [
        *model.graph.initializer,
        *(
            a.t
            for n in model.graph.node
            for a in n.attribute
            if a.type == onnx.AttributeProto.TENSOR
        ),
    ]
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in tensors):
        raise ValueError("require a self-contained ONNX without external weight files")
    onnx.checker.check_model(model)
    graph = model.graph
    if len(graph.input) == 3:
        from convert_recurrent_policy import export as export_recurrent

        return export_recurrent(source, destination, source_bytes=source_bytes)
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("exactly one input and output required")
    for value, shape in [(graph.input[0], [1, 61]), (graph.output[0], [1, 14])]:
        tt = value.type.tensor_type
        if tt.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tt.shape.dim] != shape:
            raise ValueError(f"unsupported tensor contract: {value.name}")
    if [n.op_type for n in graph.node] != [
        "Sub",
        "Div",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
    ]:
        raise ValueError("unsupported graph: require normalizer + four Dense / three ELU")
    constants = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    used = set()

    def constant(name, shape):
        """Validate and mark one consumed float32 model initializer.

        Adds the name to used only after validation; final used-set equality prevents silently
        accepting unused tensors.

        @param name Exact ONNX initializer name.
        @param shape Required tensor dimensions as a list.
        @return Borrowed initializer array or ValueError on missing/shape/dtype/nonfinite data.
        """
        arr = constants.get(name)
        if (
            arr is None
            or list(arr.shape) != shape
            or arr.dtype != np.float32
            or not np.isfinite(arr).all()
        ):
            raise ValueError(f"invalid constant {name}, expected float32 {shape}")
        used.add(name)
        return arr

    tensor = graph.input[0].name
    coeffs = []
    dims = [61]
    for index, node in enumerate(graph.node):
        if node.domain not in ("", "ai.onnx") or len(node.output) != 1 or node.input[0] != tensor:
            raise ValueError(f"nonlinear graph topology at node {index}")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type in ("Sub", "Div"):
            if attrs or len(node.input) != 2:
                raise ValueError("unsupported normalizer attributes")
            arr = constant(node.input[1], [1, 61])
            if node.op_type == "Div" and not (arr > 0).all():
                raise ValueError("normalizer denominator must be positive")
            coeffs.append(arr.ravel())
        elif node.op_type == "Gemm":
            expected = {"alpha": 1.0, "beta": 1.0, "transB": 1}
            if (
                len(node.input) != 3
                or any(k not in expected or v != expected[k] for k, v in attrs.items())
                or attrs.get("transB", 0) != 1
            ):
                raise ValueError("require nontransposed A and transposed B with unit alpha/beta")
            weight = constants.get(node.input[1])
            if weight is None or weight.ndim != 2:
                raise ValueError("missing dense weight matrix")
            width, previous = weight.shape
            if previous != dims[-1] or not 1 <= width <= 512:
                raise ValueError("invalid dense shape")
            weight = constant(node.input[1], [width, previous])
            bias = constant(node.input[2], [width])
            coeffs.extend([bias, weight.ravel()])
            dims.append(width)
        elif node.op_type == "Elu":
            if len(node.input) != 1 or any(k != "alpha" or v != 1.0 for k, v in attrs.items()):
                raise ValueError("require ELU alpha=1")
        tensor = node.output[0]
    if tensor != graph.output[0].name or dims[-1] != 14 or used != set(constants):
        raise ValueError("unconsumed output or constants")
    data = np.concatenate(coeffs).astype("<f4").tobytes()
    digest = hashlib.sha256(source_bytes).digest()
    header = struct.pack(
        "<8s6I5I32s44s",
        b"MDPOL001",
        2 if contract else 1,
        61,
        14,
        4,
        len(data) // 4,
        zlib.crc32(data),
        *dims,
        digest,
        contract or bytes(44),
    )
    assert len(header) == 128
    session = ort.InferenceSession(source_bytes, providers=["CPUExecutionProvider"])
    selftest = session.run(None, {graph.input[0].name: np.zeros((1, 61), np.float32)})[0]
    if selftest.dtype != np.float32 or selftest.shape != (1, 14) or not np.isfinite(selftest).all():
        raise ValueError("source ONNX failed finite zero-input execution")
    manifest = {
        "format": "MDPOL001",
        "converter_version": 1,
        "source_file": source.name,
        "source_sha256": digest.hex(),
        "source_repository": "https://github.com/pollen-robotics/microduck",
        "license": "Apache-2.0",
        "binary_sha256": hashlib.sha256(header + data).hexdigest(),
        "dims": dims,
        "bytes": len(header + data),
        "normalizer_included": True,
        "observation": "alpha-61-mouth-index-9-excluded",
        "selftest_zero_output": selftest[0].tolist(),
    }
    if contract:
        manifest.update(
            info(header),
            converter_version=3,
            source_metadata=source_metadata,
            source_runtime=source_metadata["microduck_runtime"],
            deployment_contract=deployment(source_metadata, source_digest),
            license="See original microduck_rl LICENSE-MODELS.md",
            source_repository="https://github.com/microduck-awesome/microduck_rl",
            observation="v2-cad-home-61-mouth-index-9-excluded",
        )
    # Validate the complete reference/metadata before replacing any caller output.
    metadata = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + data)
    destination.with_suffix(".json").write_text(metadata)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination), indent=2))
