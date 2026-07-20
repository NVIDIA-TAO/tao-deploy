# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet-Classification Engine Building and Inferencer Unit Tests."""

import pytest
import os

import numpy as np
from PIL import Image
from nvidia_tao_deploy.engine.builder import EngineBuilder
import tensorrt as trt
from tqdm import tqdm

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.visual_changenet.classification.inferencer import ChangeNetInferencer as ChangeNetClassifyInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/changenet/classify/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
OUTPUT_SHAPE = (1, 1)


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
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "changenet-classify_1light.onnx")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/C48@1_SolderLight.jpg")])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("difference_module", ['euclidean'])
@pytest.mark.parametrize("data_type", ["fp32"])
def test_fp_engine(model_path, data_type, batch_size, img_path, difference_module):
    # Engine building part

    tmp_onnx_file = model_path
    file_format = "onnx"

    output_engine_path = f"/tmp/changnet.{data_type}.engine"

    builder = EngineBuilder(min_batch_size=batch_size,
                            opt_batch_size=batch_size,
                            max_batch_size=batch_size,
                            batch_size=batch_size)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(
        output_engine_path,
        data_type,
    )

    assert os.path.exists(output_engine_path), "Engine was not generated"

    trt_infer = ChangeNetClassifyInferencer(
            engine_path=output_engine_path,
            batch_size=batch_size,
            diff_module=difference_module
        )

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape
    c1, h1, w1 = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size for input 1"
    assert w == TARGET_WIDTH, "Incorrect width size for input 1"
    assert c == 3, "Incorrect channel size for input 1"

    assert h1 == TARGET_HEIGHT, "Incorrect height size for input 2"
    assert w1 == TARGET_WIDTH, "Incorrect width size for input 2"
    assert c1 == 3, "Incorrect channel size for input 2"

    # Load dummy image
    dummy_images = []
    dummy_images1 = []
    for _ in range(batch_size):
        dummy_image = Image.open(img_path).resize((TARGET_HEIGHT, TARGET_WIDTH))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)
        dummy_images1.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)
    dummy_images1 = np.stack(dummy_images1, axis=0)

    results = trt_infer.infer([dummy_images, dummy_images1])
    assert results.shape == (batch_size,) + OUTPUT_SHAPE, "Incorrect output shape"

    engine = load_engine(output_engine_path)
    assert type(engine) == trt.tensorrt.ICudaEngine, (
        "CUDA engine type failed."
    )
    del engine
