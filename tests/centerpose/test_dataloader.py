# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CenterPose Data Loading Unit Tests."""

import pytest
import os
import numpy as np
import json
from PIL import Image
import tempfile
from omegaconf import OmegaConf

from nvidia_tao_deploy.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_deploy.cv.centerpose.dataloader import CPPredictDataset

TARGET_WIDTH = 512
TARGET_HEIGHT = 512
JSON_DICT = {
    "AR_data": {
        "plane_center": [
            0.09141058474779129,
            -0.4654481112957001,
            -2.084526300430298
        ],
        "plane_normal": [
            -0.6214213967323303,
            0.614223301410675,
            0.48637959361076355
        ]
    },
    "camera_data": {
        "camera_projection_matrix": [
            [
                1.52635657787323,
                0.0,
                0.009838283061981201,
                0.0
            ],
            [
                0.0,
                2.035142183303833,
                -0.01886594295501709,
                0.0
            ],
            [
                0.0,
                0.0,
                -0.9999997615814209,
                -0.0009999998146668077
            ],
            [
                0.0,
                0.0,
                -1.0,
                0.0
            ]
        ],
        "camera_view_matrix": [
            [
                -0.7896934747695923,
                -0.611926257610321,
                -0.04394054785370827,
                5.343637943267822
            ],
            [
                -0.5105103850364685,
                0.6157151460647583,
                0.6002286672592163,
                10.670602798461914
            ],
            [
                -0.3402407467365265,
                0.49642857909202576,
                -0.7986208200454712,
                -8.142728805541992
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0000005960464478
            ]
        ],
        "height": 800,
        "intrinsics": {
            "cx": 294.1318766276042,
            "cy": 395.8563486735026,
            "fx": 610.5426534016927,
            "fy": 610.5426534016927
        },
        "width": 600
    },
    "objects": [
        {
            "class": "bike",
            "keypoints_3d": [
                [
                    -0.24901613593101501,
                    -0.12291328608989716,
                    -1.8083537817001343
                ],
                [
                    0.8577174544334412,
                    0.013282767497003078,
                    -1.7100194692611694
                ],
                [
                    -0.3679855167865753,
                    -0.33202818036079407,
                    -2.8399617671966553
                ],
                [
                    0.17686404287815094,
                    0.6983523964881897,
                    -1.1576735973358154
                ],
                [
                    -1.0488389730453491,
                    0.353040486574173,
                    -2.2876148223876953
                ],
                [
                    0.5508071184158325,
                    -0.5988670587539673,
                    -1.3290935754776
                ],
                [
                    -0.6748962998390198,
                    -0.9441789984703064,
                    -2.459033966064453
                ],
                [
                    -0.1300462931394577,
                    0.08620256930589676,
                    -0.7767467498779297
                ],
                [
                    -1.3557497262954712,
                    -0.25910934805870056,
                    -1.9066879749298096
                ]
            ],
            "location": [
                -0.2490161657333374,
                -0.12291321158409119,
                -1.8083552122116089
            ],
            "name": "bike_0",
            "projected_cuboid": [
                [
                    264.1417980194092,
                    311.9422912597656
                ],
                [
                    310.40804386138916,
                    702.4796009063721
                ],
                [
                    234.25626754760742,
                    316.9247627258301
                ],
                [
                    674.2854595184326,
                    489.41755294799805
                ],
                [
                    399.9269485473633,
                    116.0151720046997
                ],
                [
                    30.356186628341675,
                    649.2752075195312
                ],
                [
                    71.1406409740448,
                    228.42774391174316
                ],
                [
                    373.51176738739014,
                    293.7072515487671
                ],
                [
                    222.6496696472168,
                    -38.29150199890137
                ]
            ],
            "provenance": "objectron",
            "quaternion_xyzw": [
                -0.42756551547095173,
                0.8197863197504603,
                0.09010790219499404,
                -0.37016035159383637
            ],
            "scale": [
                0.783598780632019,
                1.1126405000686646,
                1.702457070350647
            ],
            "visibility": 1.0
        }
    ]
}

@pytest.mark.parametrize("data_format", ["channels_first"])
@pytest.mark.parametrize("num_channels", [3])
def test_dataloader(data_format, num_channels):
    if data_format == "channels_first":
        input_shape = (1, num_channels, TARGET_HEIGHT, TARGET_WIDTH)
    else:
        input_shape = (1, TARGET_HEIGHT, TARGET_WIDTH, num_channels)

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Load dummy image
        img_width = JSON_DICT['camera_data']['width']
        img_height = JSON_DICT['camera_data']['height']

        dummy_image = np.random.randint(low=0, high=255, size=(img_height, img_width, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_image)

        # Save dummy image
        dummy_image.save(os.path.join(tmpdirname, "temp.png"))

        # Dataset config
        data_config = OmegaConf.structured(CenterPoseDatasetConfig())

        # Store json file
        with open(os.path.join(tmpdirname, "temp.json"), "w") as f:
            json.dump(JSON_DICT, f)

        dl = CPPredictDataset(
            dataset_config=data_config,
            inference_data=tmpdirname,
            shape=input_shape,
            dtype=np.float32,
            evaluate=True)

        for batches, img_paths, batch_params in dl.get_evaluation_batch():
            batch_c, batch_s, _, batch_intrinsic = batch_params

            assert batches[0].shape == input_shape[1:], "Incorrect image dimension"
            assert batch_c.shape[1] == 2, "Incorrect principle points format"
            assert batch_s.shape[1] == 1, "Incorrect maximum format"
            assert batch_intrinsic.shape[1:] == (3, 3), "Incorrect intrinsic matrix format"
