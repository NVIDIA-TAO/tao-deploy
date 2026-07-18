# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask RCNN Data Loading Unit Tests."""

import pytest
import os
import numpy as np
import json
from PIL import Image
import tempfile

from nvidia_tao_deploy.cv.mask_rcnn.dataloader import MRCNNCOCOLoader


TARGET_WIDTH = 960
TARGET_HEIGHT = 576
COCO_DICT = {
    "images": [
        {
            "license": 1,
            "file_name": "000000552775.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000552775.jpg",
            "height": 500,
            "width": 375,
            "id": 552775
        },
        {
            "license": 3,
            "file_name": "000000394940.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000394940.jpg",
            "height": 640,
            "width": 426,
            "id": 394940
        },
        {
            "license": 2,
            "file_name": "000000015335.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000015335.jpg",
            "height": 480,
            "width": 640,            
            "id": 15335
        }
    ],
    "annotations": [
        {
            "area": 475.48160000000036,
            "iscrowd": 0,
            "image_id": 552775,
            "bbox": [
                286.65,
                96.17,
                19.33,
                27.98
            ],
            "category_id": 1,
            "id": 83208
        },
        {
            "area": 384.27450000000016,
            "iscrowd": 0,
            "image_id": 552775,
            "bbox": [
                251.46,
                93.75,
                19.76,
                29.02
            ],
            "category_id": 2,
            "id": 93799
        },
        {
            "area": 14986.082449999996,
            "iscrowd": 0,
            "image_id": 552775,
            "bbox": [
                1.87,
                133.08,
                94.02,
                204.39
            ],
            "category_id": 1,
            "id": 1749992
        },
        {
            "area": 114105.64780000002,
            "iscrowd": 0,
            "image_id": 394940,
            "bbox": [
                0.0,
                77.18,
                425.79,
                413.66
            ],
            "category_id": 1,
            "id": 2164128
        },
        {
            "area": 56257.02764999998,
            "iscrowd": 0,
            "image_id": 15335,
            "bbox": [
                365.92,
                15.07,
                274.08,
                458.47
            ],
            "category_id": 1,
            "id": 1216332
        },
        {
            "area": 44578.0246,
            "iscrowd": 0,
            "image_id": 15335,
            "bbox": [
                173.66,
                139.15,
                208.18,
                335.46
            ],
            "category_id": 1,
            "id": 1230490
        },
        {
            "area": 5927.082549999999,
            "iscrowd": 0,
            "image_id": 15335,
            "bbox": [
                237.45,
                46.43,
                99.64,
                107.15
            ],
            "category_id": 2,
            "id": 1248228
        },
        {
            "area": 2162.5128,
            "iscrowd": 0,
            "image_id": 15335,
            "bbox": [
                599.96,
                422.9,
                40.04,
                57.1
            ],
            "category_id": 1,
            "id": 1879878
        }
    ],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person"
        },
        {
            "supercategory": "vehicle",
            "id": 2,
            "name": "bicycle"
        },
    ]
}

@pytest.mark.mask_rcnn
@pytest.mark.parametrize("data_format", ["channels_first"])
@pytest.mark.parametrize("num_channels", [3])
def test_dataloader(data_format, num_channels):
    if data_format == "channels_first":
        input_shape = (1, num_channels, TARGET_HEIGHT, TARGET_WIDTH)
    else:
        input_shape = (1, TARGET_HEIGHT, TARGET_WIDTH, num_channels)

    with tempfile.TemporaryDirectory() as tmpdirname:
        image_dir = os.path.join(tmpdirname, "images")
        os.makedirs(image_dir, exist_ok=True)

        # 4 images scenario
        for row in COCO_DICT['images']:
            # Load dummy image
            img_width = row['width']
            img_height = row['height']
            filename = row['file_name']

            dummy_image = np.random.randint(low=0, high=255, size=(img_height, img_width, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_image)

            # Save dummy image
            dummy_image.save(os.path.join(image_dir, filename))

        # Store json file
        with open(os.path.join(tmpdirname, "temp.json"), "w") as f:
            json.dump(COCO_DICT, f)

        dl = MRCNNCOCOLoader(
            os.path.join(tmpdirname, "temp.json"),
            shape=input_shape,
            data_format="channels_first",
            dtype=np.float32,
            batch_size=1,
            image_dir=image_dir)

        for imgs, _, _, labels in dl:
            assert imgs[0].shape == input_shape[1:], "Incorrect image dimension"
            assert labels[0][0].shape[-1] == 7, "Incorrect label format"
