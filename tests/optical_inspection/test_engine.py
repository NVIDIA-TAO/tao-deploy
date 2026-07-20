# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet Engine Building and Inferencer Unit Tests."""

import pytest
import os

import numpy as np
from nvidia_tao_deploy.engine.builder import EngineBuilder
import tensorrt as trt

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.optical_inspection.inferencer import OpticalInspectionInferencer

model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/optical_inspection"
TARGET_WIDTH = 100
TARGET_HEIGHT = 400
OUTPUT_SHAPE = (5,)


def load_engine(engine_path):
    """Check load engine.

    Args:
        engine_path(str): Path to the engine.

    Return:
        engine: Return TRT engine.
    """
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(trt_logger)
    trt.init_libnvinfer_plugins(trt_logger, namespace="")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


@pytest.mark.optical_inpspection
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "oi_model.onnx")])
@pytest.mark.parametrize("model_key", ["nvidia_tao"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1])
def test_fp_engine(model_path, model_key, data_type, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        tmp_onnx_file = model_path
        file_format = "onnx"

    output_engine_path = f"/tmp/ocrnet.{data_type}.engine"

    builder = EngineBuilder(min_batch_size=1,
                            opt_batch_size=1,
                            max_batch_size=1)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    trt_infer = OpticalInspectionInferencer(engine_path=output_engine_path,
                                            batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_images = []
    dummy_images1 = []
    for _ in range(batch_size):
        dummy_image = np.random.randint(low=0, high=255, size=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)
        dummy_images1.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)
    dummy_images1 = np.stack(dummy_images1, axis=0)

    results = trt_infer.infer([dummy_images, dummy_images1])
    assert results[0].shape == (batch_size,) + OUTPUT_SHAPE, "Incorrect output shape"
    assert results[1].shape == (batch_size,) + OUTPUT_SHAPE, "Incorrect output shape"

    engine = load_engine(output_engine_path)
    assert type(engine) == trt.tensorrt.ICudaEngine, (
        "CUDA engine type failed."
    )
    del engine
