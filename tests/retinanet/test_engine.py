# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RetinaNet Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.retinanet.engine_builder import RetinaNetEngineBuilder
from nvidia_tao_deploy.cv.retinanet.inferencer import RetinaNetInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"

# UFF no longer supported in DLFw 24.06+
# os.path.join(model_path, "retinanet/retinanet_resnet18_epoch_080_its_trt825.etlt"
@pytest.mark.retinanet
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "retinanet/retinanet_resnet18_epoch_015_kitti_onnx.etlt")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_its_mini.jpg")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, img_path, model_key, data_type, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/retinanet.{data_type}.engine"

    builder = RetinaNetEngineBuilder(min_batch_size=1,
                                     opt_batch_size=2,
                                     max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = RetinaNetInferencer(output_engine_path, data_format="channel_first", batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert c == 3, "Incorrect channel size"

    dummy_images = []

    # Load dummy image
    for _ in range(batch_size):
        dummy_image = Image.open(img_path).resize((w, h))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images)
    assert y_pred.shape[-1] == 6, "Shape of prediction is incorrect"
