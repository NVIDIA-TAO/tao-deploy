# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DetectNetv2 Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils import LEGACY_API_MODE
from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.detectnet_v2.engine_builder import DetectNetEngineBuilder
from nvidia_tao_deploy.cv.detectnet_v2.inferencer import DetectNetInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
TARGET_CLASSES = {
    "peoplenet": ["person", "bag", "face"],
    "dashcamnet": ["car", "bicycle", "person", "road_sign"],
}
MODEL_LIST = [
    os.path.join(model_path, "detectnet_v2/resnet34_peoplenet.etlt")
]
if LEGACY_API_MODE():
    MODEL_LIST.append(os.path.join(model_path, "detectnet_v2/resnet18_dashcamnet_pruned.etlt"))


@pytest.mark.detectnet_v2
@pytest.mark.parametrize(
    "model_path",
    MODEL_LIST    
)
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_its_mini.jpg")])
@pytest.mark.parametrize("model_key", ["tlt_encode"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [2])
def test_fp_engine(model_path, img_path, model_key, data_type, batch_size):
    # Engine building part
    target_classes = None
    for network, class_list in TARGET_CLASSES.items():
        if network in model_path:
            target_classes = class_list
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/detectnet_v2.{data_type}.engine"

    builder = DetectNetEngineBuilder(min_batch_size=1,
                                     opt_batch_size=2,
                                     max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = DetectNetInferencer(output_engine_path, target_classes=target_classes, batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    dummy_images = []

    for _ in range(batch_size):
        # Load dummy image
        dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images)
    assert sorted(y_pred.keys()) == sorted(target_classes), "Output with incorrect keys in the dict"
