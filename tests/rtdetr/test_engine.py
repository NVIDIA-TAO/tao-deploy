# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RT-DETR Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.deformable_detr.engine_builder import DDETRDetEngineBuilder
from nvidia_tao_deploy.cv.deformable_detr.inferencer import DDETRInferencer
from nvidia_tao_deploy.cv.deformable_detr.utils import post_process


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
NUM_CLASSES = 5


@pytest.mark.rtdetr
@pytest.mark.parametrize(
    "model_path",
    [
        os.path.join(model_path, "rtdetr/its_rtdetr_model_960x544.onnx")
    ]
)
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "L0/temp_astro17.jpg")])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, img_path, data_type, batch_size):
    # Engine building part
    tmp_onnx_file = model_path
    file_format = os.path.splitext(model_path)[-1].strip(".")

    output_engine_path = f"/tmp/rtdetr.{data_type}.engine"

    builder = DDETRDetEngineBuilder(min_batch_size=1,
                                    opt_batch_size=2,
                                    max_batch_size=4,
                                    workspace=10,  # Setting a max of 10GB.
                                    )
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = DDETRInferencer(output_engine_path,
                                batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
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

    pred_logits, pred_boxes = trt_infer.infer(dummy_images)
    assert pred_logits.shape == (batch_size, 300, NUM_CLASSES), "Logit dimension is incorrect"
    assert pred_boxes.shape == (batch_size, 300, 4), "Bounding box dimension is incorrect"

    # Post-processing
    target_sizes = np.array([[TARGET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT]])
    class_labels, scores, boxes = post_process(pred_logits, pred_boxes, target_sizes)

    assert np.min(scores) >= 0, "Scores range is incorrect"
    assert np.min(class_labels) >= 0 and np.max(class_labels) < NUM_CLASSES, "Class label range is incorrect"
