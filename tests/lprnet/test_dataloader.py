# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LPRNet Data Loading Unit Tests."""

import pytest
import os
import numpy as np
import random
import string
from PIL import Image
import tempfile

from nvidia_tao_deploy.cv.lprnet.dataloader import LPRNetLoader


TARGET_WIDTH = 96
TARGET_HEIGHT = 48
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

@pytest.mark.lprnet
@pytest.mark.parametrize("data_format", ["channels_first"])
@pytest.mark.parametrize("num_channels", [3])
def test_dataloader(data_format, num_channels):

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

            # Dummy label of length 7
            label = ''.join(random.choices(string.ascii_letters + string.digits, k=7)).upper()
            np.savetxt(os.path.join(label_dir, f"{i}.txt"), [label], fmt="%s")
    
        dl = LPRNetLoader(
            (3, TARGET_HEIGHT, TARGET_WIDTH),
            [image_dir],
            [label_dir],
            classes,
            batch_size=1,
            dtype=np.float32)

        for (imgs, labels) in dl:
            assert isinstance(labels[0], str), "Invalid label type"
            if data_format == "channels_first":
                b, c, h, w = imgs.shape
            else:
                b, h, w, c = imgs.shape
            
            assert b == 1, "Batch size does not match"
            assert c == num_channels, "Height does not match"
            assert h == TARGET_HEIGHT, "Height does not match"
            assert w == TARGET_WIDTH, "Height does not match"
