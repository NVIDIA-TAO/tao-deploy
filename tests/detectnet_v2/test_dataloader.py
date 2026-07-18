# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DetectNetv2 Data Loading Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
import tempfile

from nvidia_tao_deploy.cv.detectnet_v2.dataloader import DetectNetKITTILoader


TARGET_WIDTH = 960
TARGET_HEIGHT = 544
CLASSMAP = {
    "car": "car",
    "bicycle": "bicycle",
    "person": "person",
    "road_sign": "road_sign"
}

KITTI_DICT = {
    "0000": [
        ["person", 0.00, 0, 0.00, 394.00, 321.00, 470.00, 387.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ["car", 0.00, 0, 0.00, 549.00, 103.00, 569.00, 163.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    ],
    "0020": [
        ["road_sign", 0.00, 0, 0.00, 212.00, 126.00, 239.00, 190.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    ]
}


@pytest.mark.faster_rcnn
@pytest.mark.parametrize("data_format", ["channels_first"])
@pytest.mark.parametrize("num_channels", [3]) # TODO: Add grayscale support
@pytest.mark.parametrize("keep_aspect_ratio", [True, False])
def test_dataloader(data_format, num_channels, keep_aspect_ratio):

    with tempfile.TemporaryDirectory() as tmpdirname:
        image_dir = os.path.join(tmpdirname, "images")
        label_dir = os.path.join(tmpdirname, "labels")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for k, v in KITTI_DICT.items():
            # Load dummy image
            img_width = np.random.randint(low=800, high=1000)
            img_height = np.random.randint(low=500, high=600)

            dummy_image = np.random.randint(low=0, high=255, size=(img_height, img_width, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_image)

            # Save dummy image
            dummy_image.save(os.path.join(image_dir, f"{k}.jpg"))

            # Save dummy label
            np.savetxt(os.path.join(label_dir, f"{k}.txt"), v, fmt="%s")
    
        dl = DetectNetKITTILoader(
            shape=(num_channels, TARGET_HEIGHT, TARGET_WIDTH),
            image_dirs=[image_dir],
            label_dirs=[label_dir],
            mapping_dict=CLASSMAP,
            batch_size=1,
            dtype=np.float32)

        for (imgs, labels) in dl:
            assert labels.shape[-1] == 6, "Label dim does not match"

            if data_format == "channels_first":
                b, c, h, w = imgs.shape
            else:
                b, h, w, c = imgs.shape
            
            assert b == 1, "Batch size does not match"
            assert c == num_channels, "Height does not match"
            assert h == TARGET_HEIGHT, "Height does not match"
            assert w == TARGET_WIDTH, "Height does not match"
