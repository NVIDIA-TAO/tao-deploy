# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CenterPose Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np

from nvidia_tao_deploy.utils.decoding import decode_model
from nvidia_tao_deploy.cv.centerpose.engine_builder import CenterPoseEngineBuilder
from nvidia_tao_deploy.cv.centerpose.inferencer import CenterPoseInferencer


onnx_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 512
TARGET_HEIGHT = 512


@pytest.mark.parametrize("onnx_path", [os.path.join(onnx_path, "centerpose/bike_DLA34.onnx")])
@pytest.mark.parametrize("num_channels", [3])
# @pytest.mark.parametrize("data_type", ["fp32", "fp16"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(onnx_path, data_type, num_channels, batch_size):
    # Check the onnx model exists.
    if not os.path.exists(onnx_path):
        raise ValueError(f"The ONNX model path {onnx_path} is incorrect.")
    tmp_onnx_file, file_format = decode_model(onnx_path)

    # # Engine building part
    output_engine_path = f"/tmp/centerpose_{data_type}.engine"

    builder = CenterPoseEngineBuilder(workspace=1,
                                      min_batch_size=1,
                                      opt_batch_size=2,
                                      max_batch_size=4,
                                      )
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    if not os.path.exists(output_engine_path):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {output_engine_path} does not exist!")

    trt_infer = CenterPoseInferencer(output_engine_path, batch_size=batch_size, data_format="channel_first")

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == num_channels, "Incorrect channel size"

    # Create dummy images
    dummy_images = []
    for _ in range(batch_size):
        dummy_image = np.random.randint(low=0, high=255, size=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    det = trt_infer.infer(dummy_images)

    pred_keys = ['bboxes', 'scores', 'kps', 'clses', 'obj_scale', 'kps_displacement_mean', 'kps_heatmap_mean']
    assert sorted(det.keys()) == sorted(pred_keys), "Prediction does not have correct format"
    assert det['bboxes'].shape[-1] == 4, "Incorrect bounding boxes format"
    assert det['scores'].shape[-1] == 1, "Incorrect score format"
    assert det['kps'].shape[-1] == 16, "Incorrect keypoints format"
    assert det['clses'].shape[-1] == 1, "Incorrect class format"
    assert det['obj_scale'].shape[-1] == 3, "Incorrect object scale format"
    assert det['kps_displacement_mean'].shape[-1] == 16, "Incorrect displacement heatmap format"
    assert det['kps_heatmap_mean'].shape[-1] == 16, "Incorrect heatmap format"

    assert len({v.shape[0] for v in det.values()}) == 1, "Incorrect batch size processing"

# TODO: Need to add a functional test for evaluation with accuracy check.
