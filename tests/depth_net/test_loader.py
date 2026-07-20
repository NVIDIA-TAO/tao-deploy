# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast tests for DepthNetDataLoader parsing and batching."""

import os
import tempfile
import numpy as np
import pytest

from nvidia_tao_deploy.cv.depth_net.dataloader import DepthNetDataLoader


def _write(tmp, lines):
    for ln in lines:
        tmp.write((ln + "\n").encode("utf-8"))
    tmp.flush()


def _touch_images(tmpdir, names):
    paths = []
    for n in names:
        p = os.path.join(tmpdir, n)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # create tiny png
        import cv2
        cv2.imwrite(p, (np.random.rand(8, 8, 3) * 255).astype(np.uint8))
        paths.append(p)
    return paths


@pytest.mark.parametrize("shape", [(2, 3, 16, 16)])
def test_loader_mono_no_gt(tmp_path, shape):
    lefts = _touch_images(tmp_path, ["l/0.png", "l/1.png", "l/2.png"])  # more than batch
    with tempfile.NamedTemporaryFile() as tf:
        _write(tf, [lefts[0], lefts[1]])
        loader = DepthNetDataLoader(
            data_sources=[{"dataset_name": "RelativeMonoDataset", "data_file": tf.name}],
            shape=shape,
            dtype=np.float32,
            preprocessor="DepthNet",
            evaluation=False,
        )
        it = loader.get_batch()
        batch, paths, scales = next(it)
        assert batch.shape == shape
        assert len(paths) <= shape[0]
        assert len(scales) == len(paths)


@pytest.mark.parametrize("shape", [(2, 3, 16, 16)])
def test_loader_stereo_no_gt(shape, tmp_path):
    lefts = _touch_images(tmp_path, ["l/0.png", "l/1.png"])  # equal pairs
    rights = _touch_images(tmp_path, ["r/0.png", "r/1.png"]) 
    with tempfile.NamedTemporaryFile() as tf:
        _write(tf, [f"{l} {r}" for l, r in zip(lefts, rights)])
        loader = DepthNetDataLoader(
            data_sources=[{"dataset_name": "GenericDataset", "data_file": tf.name}],
            shape=shape,
            dtype=np.float32,
            preprocessor="DepthNet",
            evaluation=False,
        )
        it = loader.get_batch()
        batch, paths, scales = next(it)
        assert isinstance(batch, dict)
        assert batch["left_image"].shape == shape
        assert batch["right_image"].shape == shape
        assert len(paths) == len(scales)

