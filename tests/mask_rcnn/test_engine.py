# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask RCNN Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.mask_rcnn.engine_builder import MRCNNEngineBuilder
from nvidia_tao_deploy.cv.mask_rcnn.inferencer import MRCNNInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 960
TARGET_HEIGHT = 576
NUM_CLASSES = 2


@pytest.mark.skip(reason="UFF models no longer supported")
@pytest.mark.mask_rcnn
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "mask_rcnn/model.step-600000.etlt")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_astro17.jpg")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("max_detections_per_image", [100])
def test_fp_engine(model_path, img_path, model_key, data_type, max_detections_per_image):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/mask_rcnn.{data_type}.engine"

    builder = MRCNNEngineBuilder()
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = MRCNNInferencer(output_engine_path,
                                nms_size=max_detections_per_image,
                                n_classes=NUM_CLASSES,
                                mask_size=28)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
    dummy_image = np.asarray(dummy_image, np.float32)

    # Add batch dim
    dummy_image = np.expand_dims(dummy_image, 0)

    y_pred = trt_infer.infer(dummy_image, scales=[1.0, 1.0])

    pred_keys = ['num_detections', 'detection_classes', 'detection_scores', 'detection_boxes', 'detection_masks']
    assert sorted(y_pred.keys()) == sorted(pred_keys), "Prediction does not have correct format"
    assert y_pred['detection_classes'].shape[-1] == max_detections_per_image, "Number of detection is incorrect"
    assert y_pred['detection_boxes'].shape[1:] == (max_detections_per_image, 4), "Number of detection is incorrect"
