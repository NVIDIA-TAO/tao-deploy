# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification Data Loading Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
import tempfile

from nvidia_tao_deploy.cv.classification_tf1.dataloader import ClassificationLoader


TARGET_WIDTH = 224
TARGET_HEIGHT = 224


@pytest.mark.classification_tf1
@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
# @pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("num_channels", [3]) # TODO: Add grayscale support
@pytest.mark.parametrize("interpolation_method", ["bilinear", "bicubic"])
@pytest.mark.parametrize("mode", ["caffe", "torch"])
@pytest.mark.parametrize("crop", ["center", "random"])
def test_dataloader(data_format, num_channels, interpolation_method, mode, crop):
    # 4 classes dataset scenario
    dummy_classes = {"a": 0, "b": 1, "c": 2, "d": 3}

    if data_format == "channels_first":
        input_shape = (num_channels, TARGET_HEIGHT, TARGET_WIDTH)
    else:
        input_shape = (TARGET_HEIGHT, TARGET_WIDTH, num_channels)

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
    
        dl = ClassificationLoader(
            input_shape,
            [tmpdirname],
            dummy_classes,
            data_format=data_format,
            interpolation_method=interpolation_method,
            mode=mode,
            crop=crop,
            batch_size=1,
            image_mean=None,
            dtype=np.float32)

        for (imgs, labels), cls in zip(dl, dummy_classes.values()):
            assert labels[0] == cls, "Label does not match"

            if data_format == "channels_first":
                b, c, h, w = imgs.shape
            else:
                b, h, w, c = imgs.shape
            
            assert b == 1, "Batch size does not match"
            assert c == num_channels, "Height does not match"
            assert h == TARGET_HEIGHT, "Height does not match"
            assert w == TARGET_WIDTH, "Height does not match"
