# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EfficientDet Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.efficientdet_tf1.engine_builder import EfficientDetEngineBuilder
from nvidia_tao_deploy.cv.efficientdet_tf1.inferencer import EfficientDetInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
NUM_CLASSES = 2


@pytest.mark.efficientdet_tf1
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "efficientdet_tf1/model.step-39060_trt825.etlt")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_avlp.jpg")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("max_detections_per_image", [100])
@pytest.mark.parametrize("batch_size", [1])
def test_fp_engine(model_path, img_path, model_key, data_type, max_detections_per_image, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/efficientdet_tf1.{data_type}.engine"

    builder = EfficientDetEngineBuilder(min_batch_size=1,
                                        opt_batch_size=2,
                                        max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type, tf2=True)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = EfficientDetInferencer(output_engine_path, batch_size=batch_size,
                                       data_format="channel_last",
                                       max_detections_per_image=max_detections_per_image)

    # Check engine model shape
    h, w, c = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    dummy_images = []

    for _ in range(batch_size):
        # Load dummy image
        dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
        dummy_image = np.asarray(dummy_image, np.float32)
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images, scales=[1.0, 1.0])

    pred_keys = ['num_detections', 'detection_classes', 'detection_scores', 'detection_boxes']
    assert sorted(y_pred.keys()) == sorted(pred_keys), "Prediction does not have correct format"
    assert y_pred['detection_classes'].shape[-1] == max_detections_per_image, "Number of detection is incorrect"
    assert y_pred['detection_boxes'].shape[1:] == (max_detections_per_image, 4), "Number of detection is incorrect"
