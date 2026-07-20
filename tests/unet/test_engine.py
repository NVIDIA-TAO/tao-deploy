# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unet Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.unet.engine_builder import UNetEngineBuilder
from nvidia_tao_deploy.cv.unet.inferencer import UNetInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 960
TARGET_HEIGHT = 544


@pytest.mark.unet
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "unet/model.step-4772476.etlt")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_its_mini.jpg")])
@pytest.mark.parametrize("model_key", ["tlt_encode"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("activation", ["sigmoid"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, img_path, model_key, data_type, activation, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/unet.{data_type}.engine"

    builder = UNetEngineBuilder(min_batch_size=1,
                                opt_batch_size=2,
                                max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = UNetInferencer(output_engine_path, batch_size=batch_size, activation=activation)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
    dummy_image = np.asarray(dummy_image, np.float32)

    dummy_images = []

    # Load dummy image
    for _ in range(batch_size):
        dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images)
    assert y_pred.shape == (batch_size, TARGET_HEIGHT, TARGET_WIDTH), "Shape of prediction is incorrect"
