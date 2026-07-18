# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCDNet Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_model
from nvidia_tao_deploy.cv.ocdnet.engine_builder import OCDNetEngineBuilder
from nvidia_tao_deploy.cv.ocdnet.inferencer import OCDNetInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/ocdnet/model"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/test"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 736


@pytest.mark.ocrnet
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "dcn_resnet18_Uber_finetune_icdar15_best.onnx")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "img/img_1.jpg")])
@pytest.mark.parametrize("model_key", ["nvidia_tao"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, img_path, model_key, data_type, batch_size):
    # Engine building part
    if model_path.endswith("onnx"):
        tmp_onnx_file, file_format = decode_model(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/ocdnet.{data_type}.engine"

    builder = OCDNetEngineBuilder(width=TARGET_WIDTH,
                                  height=TARGET_HEIGHT,
                                  min_batch_size=1,
                                  opt_batch_size=2,
                                  max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = OCDNetInferencer(engine_path=output_engine_path,
                                 batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_images = []

    for _ in range(batch_size):
        dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    results = trt_infer.infer(dummy_images)

    assert results.shape == (batch_size, 1, TARGET_HEIGHT, TARGET_WIDTH)
