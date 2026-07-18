# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests to decrpyt models."""

import pytest
import os
from nvidia_tao_deploy.utils.decoding import decode_model


@pytest.mark.core
@pytest.mark.parametrize("model_path", ["/home/scratch.metropolis2/tao_ci/tao_deploy/models/efficientdet_tf1/model.step-39060_trt825.etlt"])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
def test_decode_model(model_path, model_key):
    tmp_onnx_file, file_format = decode_model(model_path, model_key)

    assert os.path.exists(tmp_onnx_file)
    assert file_format == "onnx"
