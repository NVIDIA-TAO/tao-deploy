# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segformer Data Loading Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
import tempfile

from nvidia_tao_deploy.cv.segformer.dataloader import SegformerLoader


TARGET_WIDTH = 960
TARGET_HEIGHT = 544
NUM_CLASSES = 2


@pytest.mark.faster_rcnn
@pytest.mark.parametrize("data_format", ["channels_first"])
@pytest.mark.parametrize("num_channels", [3])  # Grayscale is also channel size 3
@pytest.mark.parametrize("resize_method", ["bilinear", "BICUBIC"])
@pytest.mark.parametrize("input_image_type", ["color", "grayscale"])
@pytest.mark.parametrize("keep_ratio", [True, False])
def test_dataloader(data_format, num_channels, input_image_type, resize_method, keep_ratio):
    if data_format == "channels_first":
        input_shape = (num_channels, TARGET_HEIGHT, TARGET_WIDTH)
    else:
        input_shape = (TARGET_HEIGHT, TARGET_WIDTH, num_channels)

    with tempfile.TemporaryDirectory() as tmpdirname:
        image_dir = os.path.join(tmpdirname, "images")
        label_dir = os.path.join(tmpdirname, "labels")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i in range(3):
            # Load dummy image
            img_width = np.random.randint(low=800, high=1000)
            img_height = np.random.randint(low=500, high=600)
            
            dummy_image = np.random.randint(low=0, high=255, size=(img_height, img_width, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_image)

            # Save dummy image
            dummy_image.save(os.path.join(image_dir, f"{i}.jpg"))

            # Dummy mask label with near-center being 1
            dummy_mask = np.zeros((img_height, img_width))
            dummy_mask[262:282, 470:490] = 1
            dummy_mask = Image.fromarray((dummy_mask * 255).astype(np.uint8))

            # Save dummy mask
            dummy_mask.save(os.path.join(label_dir, f"{i}.png"))

        dl = SegformerLoader(
            shape=input_shape,
            image_data_source=[image_dir],
            label_data_source=[label_dir],
            num_classes=NUM_CLASSES,
            batch_size=1,
            resize_method=resize_method,
            input_image_type=input_image_type,
            keep_ratio=keep_ratio,
            image_mean=[123.675, 116.28, 103.53],
            image_std=[58.395, 57.12, 57.375],
            dtype=np.float32)

        for (imgs, labels) in dl:
            if data_format == "channels_first":
                b, c, h, w = imgs.shape
            else:
                b, h, w, c = imgs.shape

            assert b == 1, "Batch size does not match"
            assert c == num_channels, "Height does not match"
            assert h == TARGET_HEIGHT, "Height does not match"
            assert w == TARGET_WIDTH, "Height does not match"

            assert labels.shape[1:] == (TARGET_HEIGHT, TARGET_WIDTH), "Label dim does not match"
