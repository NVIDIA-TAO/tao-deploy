# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet Data Loading Unit Tests."""

import pytest

from nvidia_tao_deploy.cv.ocrnet.dataloader import OCRNetLoader

DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 100
DEFAULT_IMG_DIR = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/L0/img/"
DEFAULT_GT_TXT = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/L0/gt.txt"

@pytest.mark.ocrnet
@pytest.mark.parametrize("shape", [[1, DEFAULT_HEIGHT, DEFAULT_WIDTH]])
@pytest.mark.parametrize("image_dirs", [[DEFAULT_IMG_DIR]])
@pytest.mark.parametrize("label_txts", [[DEFAULT_GT_TXT]])
@pytest.mark.parametrize("batch_size", [1])
def test_dataloader(shape, image_dirs, label_txts, batch_size):
    dl = OCRNetLoader(shape=shape,
                      image_dirs = image_dirs,
                      batch_size=batch_size,
                      label_txts=label_txts)
    for imgs, labels in dl:
        assert imgs.shape == (batch_size, DEFAULT_HEIGHT, DEFAULT_WIDTH)
        assert len(labels) == 1
        assert labels[0] == "stand"
