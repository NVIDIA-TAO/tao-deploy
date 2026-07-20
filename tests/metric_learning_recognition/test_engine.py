# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from nvidia_tao_deploy.cv.ml_recog.engine_builder import MLRecogEngineBuilder
from nvidia_tao_deploy.cv.ml_recog.inferencer import MLRecogInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
OUTPUT_DIM = 256


@pytest.mark.metric_learning_recognition
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "metric_learning_recognition/ml_model_epoch=000.onnx")])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, data_type, batch_size):

    output_engine_path = f"/tmp/mlrecog.{data_type}.engine"

    workspace_size = 1
    min_batch_size = 1
    opt_batch_size = 2
    max_batch_size = 4
    num_channels = 3
    input_width = TARGET_WIDTH
    input_height = TARGET_HEIGHT
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.226, 0.226, 0.226]

    if not os.path.exists(output_engine_path):

        builder = MLRecogEngineBuilder(
            workspace=workspace_size,
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            img_std=img_std,
            img_mean=img_mean)
        builder.create_network(model_path, file_format="onnx")
        builder.create_engine(output_engine_path, data_type)

        assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = MLRecogInferencer(output_engine_path,
                                  input_shape=(batch_size, num_channels, input_height, input_width),
                                  batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_images = []
    for _ in range(batch_size):
        dummy_image = np.random.randint(low=0, high=255, size=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.get_embeddings_from_batch(dummy_images)

    assert y_pred.shape == (batch_size, OUTPUT_DIM), "Incorrect output dimensions"
