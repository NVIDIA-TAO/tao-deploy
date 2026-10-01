# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable MSDA bounds/nonfinite conformance gate (FP32 and FP16)."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from nvidia_tao_deploy.cv.sparse4d.runtime import EngineRunner, trt_logger, trt_modules


def probe_inputs(dtype):
    """Use production kernel dimensions, with wholly valid or invalid queries."""
    points = [(0.5, 0.5), (0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1),
              (-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1),
              (np.nan, 0.5), (0.5, np.nan), (np.inf, 0.5),
              (0.5, np.inf), (-np.inf, 0.5), (0.5, -np.inf), (np.nan, np.nan)]
    locations = np.full((1, 900, 13, 6, 2), 0.5, dtype)
    for index, point in enumerate(points):
        locations[0, index] = point
    return {"features": np.ones((1, 24, 256), dtype),
            "shapes": np.ones((6, 4, 2), np.int32),
            "starts": np.arange(24, dtype=np.int32).reshape(6, 4),
            "locations": locations,
            "weights": np.ones((1, 900, 13, 6, 4, 8), dtype)}


def verify_plugin():
    """Reject a library that does not skip all invalid locations, before model build.

    This tests the registered implementation, not the name of the selected file.
    The small graph intentionally has NO location sanitization so the plugin's
    own accepted-only bounds check is exercised. No model or dataset is needed.
    """
    trt, _ = trt_modules()
    logger = trt_logger(trt)
    results = {}
    for dtype, tensor_type in ((np.float32, TensorProto.FLOAT), (np.float16, TensorProto.FLOAT16)):
        arrays = probe_inputs(dtype)
        inputs = [helper.make_tensor_value_info(
            n, TensorProto.INT32 if a.dtype == np.int32 else tensor_type, a.shape)
            for n, a in arrays.items()]
        output = helper.make_tensor_value_info("output", tensor_type, [1, 900, 256])
        graph = helper.make_graph([helper.make_node("MSDA", list(arrays), ["output"], domain="nv")],
                                  "sparse4d_msda_conformance", inputs, [output])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("nv", 1)], ir_version=8)
        onnx.checker.check_model(model)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(model.SerializeToString()):
            raise ValueError("MSDA conformance graph rejected: " +
                             "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
        if dtype == np.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise ValueError("Could not build MSDA conformance engine")
        with EngineRunner(serialized) as runner:
            actual = runner(arrays)["output"]
        expected = np.full((1, 900, 256), 312, dtype)
        expected[:, 1:16] = 0
        if not np.array_equal(actual, expected):
            raise ValueError(f"MSDA {np.dtype(dtype)} plugin failed bounds/NaN/Inf conformance; "
                             "use the hardened accepted-only (0 < x,y < 1) implementation")
        results[str(np.dtype(dtype))] = "passed"
    return results
