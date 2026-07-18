# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Faster RCNN Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.faster_rcnn.engine_builder import FRCNNEngineBuilder
from nvidia_tao_deploy.cv.faster_rcnn.inferencer import FRCNNInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"

# UFF models no longer supported in DFLW 24.06+
# os.path.join(model_path, "faster_rcnn/frcnn_kitti_resnet18.epoch24_trt8.etlt"
@pytest.mark.faster_rcnn
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "faster_rcnn/frcnn_kitti_resnet18_onnx.epoch12.etlt")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_its_mini.jpg")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1])
def test_fp_engine(model_path, img_path, model_key, data_type, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/faster_rcnn.{data_type}.engine"

    builder = FRCNNEngineBuilder(min_batch_size=1,
                                 opt_batch_size=1,
                                 max_batch_size=1)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    input_shape = builder._input_dims["input_image"]
    trt_infer = FRCNNInferencer(output_engine_path, input_shape=(1,) + input_shape, batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert c == 3, "Incorrect channel size"

    # Load dummy images
    dummy_images = []
    for _ in range(batch_size):
        dummy_image = Image.open(img_path).resize((w, h))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images)
    assert y_pred[0].shape[-1] == 6, "Shape of prediction is incorrect"
