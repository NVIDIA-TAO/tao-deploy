# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification Data Loading Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
import tempfile

from nvidia_tao_deploy.cv.ml_recog.dataloader import MLRecogClassificationLoader, MLRecogInferenceLoader

TARGET_WIDTH = 224
TARGET_HEIGHT = 224


@pytest.mark.metric_learning_recognition
@pytest.mark.parametrize("num_channels", [1, 3])
def test_classification_dataloader(num_channels):
    # 4 classes dataset scenario
    dummy_classes = {"a": 0, "b": 1, "c": 2, "d": 3}
    batch_size = 1
    input_shape = (num_channels, TARGET_HEIGHT, TARGET_WIDTH)

    with tempfile.TemporaryDirectory() as tmpdirname:

        for cls in dummy_classes.keys():
            # Create class directory
            os.makedirs(os.path.join(tmpdirname, cls), exist_ok=True)

            # Load dummy image
            img_width = np.random.randint(low=200, high=300)
            img_height = np.random.randint(low=200, high=300)

            dummy_image = np.random.randint(low=0, high=255, size=(img_height, img_width, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_image)

            # Save dummy image
            dummy_image.save(os.path.join(tmpdirname, cls, "0000.jpg"))

        dl = MLRecogClassificationLoader(
            input_shape,
            tmpdirname,
            dummy_classes,
            batch_size=batch_size,
            image_mean=None,
            image_std=None,
            dtype=np.float32)

        for (imgs, labels), cls in zip(dl, dummy_classes.values()):
            assert labels[0] == cls, "Label does not match"

            b, c, h, w = imgs.shape

            assert b == batch_size, "Batch size does not match"
            assert c == num_channels, "Num channels does not match"
            assert h == TARGET_HEIGHT, "Height does not match"
            assert w == TARGET_WIDTH, "Width does not match"


@pytest.mark.metric_learning_recognition
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("input_type", ["classification_folder", "image_folder"])
def test_inference_dataloader(num_channels, input_type):
    # 4 classes dataset scenario
    dummy_classes = {"a": 0, "b": 1, "c": 2, "d": 3}
    input_shape = (num_channels, TARGET_HEIGHT, TARGET_WIDTH)

    with tempfile.TemporaryDirectory() as tmpdirname:

        for cls in dummy_classes.keys():

            # Load dummy image
            img_width = np.random.randint(low=200, high=300)
            img_height = np.random.randint(low=200, high=300)

            dummy_image = np.random.randint(low=0, high=255, size=(img_height, img_width, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_image)

            # Save dummy image
            if input_type == "classification_folder":
                # Create class directory
                os.makedirs(os.path.join(tmpdirname, cls), exist_ok=True)
                dummy_image.save(os.path.join(tmpdirname, cls, "0000.jpg"))
            elif input_type == "image_folder":
                dummy_image.save(os.path.join(tmpdirname, cls + "_0000.jpg"))

        dl = MLRecogInferenceLoader(
            input_shape,
            tmpdirname,
            input_type,
            batch_size=1,
            image_mean=None,
            image_std=None,
            dtype=np.float32)

        for imgs in dl:

            b, c, h, w = imgs.shape

            assert b == 1, "Batch size does not match"
            assert c == num_channels, "Num channels does not match"
            assert h == TARGET_HEIGHT, "Height does not match"
            assert w == TARGET_WIDTH, "Width does not match"
