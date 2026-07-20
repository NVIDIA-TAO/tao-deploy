# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.classification_tf1.engine_builder import ClassificationEngineBuilder
from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
OUTPUT_SHAPE = (1000,)


@pytest.mark.classification_tf1
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "classification_tf2/efficientnet-b0-qat_050.etlt")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("data_format", ["channel_last"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, model_key, data_type, data_format, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/classification_tf1.{data_type}.engine"

    builder = ClassificationEngineBuilder(min_batch_size=1,
                                          opt_batch_size=2,
                                          max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format=file_format)
    builder.create_engine(output_engine_path, data_type, tf2=True)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = ClassificationInferencer(output_engine_path, data_format=data_format, batch_size=batch_size)

    # Check engine model shape
    if data_format == "channel_first":
        c, h, w = trt_infer.input_tensors[0].shape
    else:
        h, w, c = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    dummy_images = []

    for _ in range(batch_size):
        # Load dummy image
        dummy_image = np.random.randint(low=0, high=255, size=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        dummy_image = np.asarray(dummy_image, np.float32)
        if data_format == "channel_first":
            dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images)

    assert y_pred.shape == (batch_size,) + OUTPUT_SHAPE, "Incorrect output dimensions"
