# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""D-DETR Post-Processing Unit Tests."""

import pytest
import os
import numpy as np

from nvidia_tao_deploy.cv.deformable_detr.utils import post_process


TARGET_WIDTH = 960
TARGET_HEIGHT = 544
NUM_CLASSES = 4


@pytest.mark.deformable_detr
@pytest.mark.parametrize("prediction_shape", [(1, 300, 4), (8, 200, 4)])
def test_sigmoid(prediction_shape):
    # Set random probability range
    pred_logits = np.random.rand(*prediction_shape) * np.random.randint(-10, 10, size=prediction_shape)
    pred_boxes = np.random.rand(*prediction_shape)

    target_sizes = np.array([[TARGET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT] for _ in range(len(pred_boxes))])
    class_labels, scores, boxes = post_process(pred_logits, pred_boxes, target_sizes)

    assert np.min(scores) >= 0 and np.max(scores) <= 1, "Scores range is incorrect"

    assert (scores == np.sort(scores, axis=-1)[:, ::-1]).all(), "Score is not sorted in descending order"
    assert np.min(class_labels) >= 0 and np.max(class_labels) < NUM_CLASSES, "Class label range is incorrect"
    assert np.min(boxes[..., 0]) >= 0 and np.max(boxes[..., 2]) <= TARGET_WIDTH, "x range is not correct"
    assert np.min(boxes[..., 1]) >= 0 and np.max(boxes[..., 3]) <= TARGET_WIDTH, "y range is not correct"
