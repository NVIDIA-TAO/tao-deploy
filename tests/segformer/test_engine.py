# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segformer Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
import cv2
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_model
from nvidia_tao_deploy.cv.segformer.engine_builder import SegformerEngineBuilder
from nvidia_tao_deploy.cv.segformer.inferencer import SegformerInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 512
TARGET_HEIGHT = 512


@pytest.mark.unet
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "segformer/segformer_model.onnx")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_camvid.png")])
@pytest.mark.parametrize("model_key", ["tlt_encode"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, img_path, model_key, data_type, batch_size):
    # Engine building part

    tmp_onnx_file, file_format = decode_model(model_path, model_key)

    output_engine_path = f"/tmp/segformer.{data_type}.engine"

    builder = SegformerEngineBuilder(min_batch_size=1,
                                     opt_batch_size=2,
                                     max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = SegformerInferencer(output_engine_path, batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

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
    # Predictions are in shape (batch_size, 1, TARGET_HEIGHT, TARGET_WIDTH)
    assert y_pred.shape == (batch_size, 1, TARGET_HEIGHT, TARGET_WIDTH), "Shape of prediction is incorrect"

    for pred in y_pred:
        # Store predictions as mask
        pred = np.argmax(pred, axis=0).astype(np.uint8) * 255
        # resize to original image size
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        output = Image.fromarray(pred.astype(np.uint8))
        assert output.size == (TARGET_WIDTH, TARGET_HEIGHT), "Output image size is incorrect"
