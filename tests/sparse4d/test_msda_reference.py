# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent bilinear oracle and opt-in tests for the source-built MSDA plugin."""

import os
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.sparse4d


def bilinear_reference(arrays):
    """Sum per-camera/scale/point samples, with zero padding and eight groups."""
    features = arrays["features"][0].astype(np.float64)
    locations = arrays["locations"][0].astype(np.float64)
    weights = arrays["weights"][0].astype(np.float64)
    queries, points, cameras, _ = locations.shape
    result = np.zeros((queries, features.shape[-1]), np.float64)
    for camera in range(cameras):
        for level, (height, width) in enumerate(arrays["shapes"][camera]):
            start = int(arrays["starts"][camera, level])
            for point in range(points):
                x, y = locations[:, point, camera].T
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (x < 1) & (y > 0) & (y < 1)
                px = np.where(valid, x, 0) * width - 0.5
                py = np.where(valid, y, 0) * height - 0.5
                left, top = np.floor(px).astype(int), np.floor(py).astype(int)
                sample = np.zeros_like(result)
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    ix, iy = left + dx, top + dy
                    inside = valid & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
                    factor = (1 - np.abs(px - ix)) * (1 - np.abs(py - iy))
                    sample[inside] += features[start + iy[inside] * width + ix[inside]] * factor[inside, None]
                grouped = np.repeat(weights[:, point, camera, level], features.shape[-1] // 8, axis=-1)
                result += sample * grouped
    return result[None]


def test_bilinear_reference():
    """Check the oracle against hand-computed center, padded-edge, and NaN samples."""
    arrays = {
        "features": np.repeat(np.array([[[1], [2], [3], [4]]]), 256, axis=-1),
        "shapes": np.array([[[2, 2]]]), "starts": np.array([[0]]),
        "locations": np.array([[[[[0.5, 0.5]]], [[[0.125, 0.125]]], [[[np.nan, 0.5]]]]]),
        "weights": np.ones((1, 3, 1, 1, 1, 8)),
    }
    expected = np.repeat(np.array([[[2.5], [0.5625], [0]]]), 256, axis=-1)
    np.testing.assert_array_equal(bilinear_reference(arrays), expected)


def run_plugin(arrays):
    """Build, serialize, reload, and execute a graph with exactly one MSDA node."""
    import onnx
    from onnx import helper
    from nvidia_tao_deploy.cv.sparse4d.runtime import EngineRunner, trt_logger, trt_modules

    inputs = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape)
              for name, value in arrays.items()]
    output = helper.make_tensor_value_info("output", inputs[0].type.tensor_type.elem_type, [1, 900, 256])
    graph = helper.make_graph([helper.make_node("MSDA", list(arrays), ["output"], domain="nv")],
                              "msda_reference", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("nv", 1)],
                              ir_version=8)
    onnx.checker.check_model(model)
    trt, _ = trt_modules()
    builder = trt.Builder(trt_logger(trt))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger(trt))
    assert parser.parse(model.SerializeToString()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if arrays["features"].dtype == np.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    with EngineRunner(engine) as runner:
        return runner(arrays)["output"]


@pytest.mark.skipif(not os.getenv("SPARSE4D_PLUGIN_PATH"), reason="Requires explicit plugin path/hash and a GPU")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("cameras", [4, 6])
def test_msda_bilinear_reference(dtype, index_dtype, cameras):
    """Exercise nonuniform features, padding, camera order, groups, and index types."""
    from nvidia_tao_deploy.cv.sparse4d.runtime import load_plugin

    handle = load_plugin(SimpleNamespace(path=os.environ["SPARSE4D_PLUGIN_PATH"],
                                         sha256=os.environ["SPARSE4D_PLUGIN_SHA256"]))
    rng = np.random.default_rng(29)
    shapes = np.tile([[4, 4], [2, 4], [2, 2], [1, 1]], (cameras, 1, 1)).astype(index_dtype)
    sizes = np.prod(shapes, axis=-1).ravel()
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).reshape(cameras, 4).astype(index_dtype)
    arrays = {
        "features": rng.uniform(-1, 1, (1, int(sizes.sum()), 256)).astype(dtype),
        "shapes": shapes, "starts": starts,
        # Dyadic coordinates keep location rounding out of the FP16 comparison.
        "locations": (rng.integers(1, 16, (1, 900, 13, cameras, 2)) / 16).astype(dtype),
        "weights": (rng.uniform(0, 1, (1, 900, 13, cameras, 4, 8)) / (13 * cameras * 4)).astype(dtype),
    }
    arrays["locations"][:, :4] = np.array([0, 1, np.nan, np.inf], dtype=dtype).reshape(1, 4, 1, 1, 1)
    actual = run_plugin(arrays)
    expected = bilinear_reference(arrays)
    np.testing.assert_allclose(actual, expected, rtol=0.01 if dtype == np.float16 else 2e-5,
                               atol=3e-4 if dtype == np.float16 else 2e-6)
    np.testing.assert_array_equal(actual[:, :4], 0)
    del handle
