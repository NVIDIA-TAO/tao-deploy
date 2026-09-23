# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the Sparse4D multi-input engine and a hash-bound provenance sidecar."""

import json
from pathlib import Path

import numpy as np
import onnx

from nvidia_tao_deploy.cv.sparse4d.contract import OUTPUTS, input_shapes, sha256, validate_names
from nvidia_tao_deploy.cv.sparse4d.plugin_check import verify_plugin
from nvidia_tao_deploy.cv.sparse4d.runtime import load_plugin, trt_logger, trt_modules


def build_engine(cfg):
    """Validate ABI, run MSDA conformance, then serialize a fixed deployment profile."""
    spec = cfg.gen_trt_engine
    target = Path(spec.save_engine)
    sidecar = Path(str(target) + ".json")
    if not spec.save_engine or target.exists() or sidecar.exists():
        raise ValueError("save_engine must name a new engine and sidecar; existing artifacts are never overwritten")
    if spec.data_type not in ("fp32", "fp16"):
        raise ValueError("Sparse4D gen_trt_engine supports fp32 and fp16, not INT8")
    if not np.isfinite(spec.workspace_size) or spec.workspace_size <= 0:
        raise ValueError("workspace_size must be positive GiB")
    shapes = input_shapes(cfg.model)
    if (cfg.model.num_queries, cfg.model.num_temp_instances, cfg.model.embed_dims) != (900, 600, 256):
        raise ValueError("Supported production layout: 900 queries, 600 cached queries, 256 channels")
    model = onnx.load(spec.onnx_file)
    onnx.checker.check_model(model)
    validate_names([value.name for value in model.graph.input])
    if {value.name for value in model.graph.output} != set(OUTPUTS):
        raise ValueError("Expected the six-stage Sparse4D export's twelve named outputs")
    msda_count = sum(n.op_type == "MSDA" and n.domain == "nv" for n in model.graph.node)
    if msda_count != 6:
        raise ValueError("Expected six nv::MSDA nodes in the Sparse4D export")
    del model
    plugin_handle = load_plugin(cfg.plugin)
    trt, _ = trt_modules()
    logger = trt_logger(trt)
    trt.init_libnvinfer_plugins(logger, "")
    conformance = verify_plugin()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(spec.onnx_file).resolve())):
        raise ValueError("ONNX parsing failed: " +
                         "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(spec.workspace_size * (1 << 30)))
    config.clear_flag(trt.BuilderFlag.TF32)
    if spec.data_type == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    actual_shapes = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = shapes[tensor.name]
        dtype = trt.bool if tensor.name == "interval_mask" else trt.float32
        if tensor.dtype != dtype or len(tensor.shape) != len(shape):
            raise ValueError(f"Unsupported dtype/rank for {tensor.name}")
        if any(d >= 0 and d != expected for d, expected in zip(tensor.shape, shape)):
            raise ValueError(f"Configured shape {shape} disagrees with ONNX {tensor.name}: {tensor.shape}")
        # TensorRT's Python API returns None on success and raises on invalid shapes.
        profile.set_shape(tensor.name, shape, shape, shape)
        actual_shapes[tensor.name] = list(shape)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise ValueError("TensorRT engine build failed")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(bytes(serialized))
    metadata = {"version": 1, "model_name": "sparse4d", "tensorrt": trt.__version__,
                "engine_sha256": sha256(target), "onnx_sha256": sha256(spec.onnx_file),
                "plugin_sha256": cfg.plugin.sha256.lower(), "plugin_conformance": conformance,
                "precision": spec.data_type, "input_shapes": actual_shapes,
                "model": {key: getattr(cfg.model, key) for key in cfg.model.__dataclass_fields__}
                if hasattr(cfg.model, "__dataclass_fields__") else dict(cfg.model)}
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    # Retain the CDLL through parsing, building and serialization.
    del plugin_handle
    return metadata
