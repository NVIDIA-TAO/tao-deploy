# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet-Segmentation Engine Building and Inferencer Unit Tests."""

import pytest
import os

import numpy as np
from PIL import Image
from nvidia_tao_deploy.engine.builder import EngineBuilder
import tensorrt as trt

from nvidia_tao_deploy.cv.visual_changenet.segmentation.inferencer import ChangeNetInferencer as ChangeNetSegmentInferencer


model_path_segment = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/changenet/segment-levir-cd/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
INPUT_SHAPE = 256
NUM_CLASSES = 2
OUTPUT_SHAPE = (2, INPUT_SHAPE, INPUT_SHAPE)


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


@pytest.mark.visual_changenet
@pytest.mark.parametrize("model_path", [os.path.join(model_path_segment, "changenet_segment.onnx")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/test_1_2.png")])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("data_type", ["fp32"])
def test_fp_engine(model_path, data_type, batch_size, img_path):
    # Engine building part

    tmp_onnx_file = model_path
    file_format = "onnx"

    output_engine_path = f"/tmp/changnet.{data_type}.engine"

    builder = EngineBuilder(min_batch_size=1,
                            opt_batch_size=2,
                            max_batch_size=4,
                            batch_size=batch_size)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(
        output_engine_path,
        data_type,
    )

    assert os.path.exists(output_engine_path), "Engine was not generated"

    trt_infer = ChangeNetSegmentInferencer(
                engine_path=output_engine_path,
                batch_size=batch_size,
                n_class=NUM_CLASSES,
                mode='predict'
            )

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape
    c1, h1, w1 = trt_infer.input_tensors[1].shape

    assert h == INPUT_SHAPE, "Incorrect height size for input 1"
    assert w == INPUT_SHAPE, "Incorrect width size for input 1"
    assert c == 3, "Incorrect channel size for input 1"

    assert h1 == INPUT_SHAPE, "Incorrect height size for input 2"
    assert w1 == INPUT_SHAPE, "Incorrect width size for input 2"
    assert c1 == 3, "Incorrect channel size for input 2"

    # Load dummy images
    dummy_images = []
    dummy_images1 = []
    for _ in range(batch_size):
        dummy_image = Image.open(img_path).resize((INPUT_SHAPE, INPUT_SHAPE))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)
        dummy_images1.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)
    dummy_images1 = np.stack(dummy_images1, axis=0)

    results = trt_infer.infer([dummy_images, dummy_images1])
    assert results.shape == (batch_size,) + OUTPUT_SHAPE, "Shape of prediction is incorrect"

    engine = load_engine(output_engine_path)
    assert type(engine) == trt.tensorrt.ICudaEngine, (
        "CUDA engine type failed."
    )
    del engine
