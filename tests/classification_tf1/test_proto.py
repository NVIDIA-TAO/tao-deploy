# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests to load configurations from proto."""

import pytest
# from nvidia_tao_deploy.cv.classification_tf1.proto.utils import load_proto


@pytest.mark.skip
@pytest.mark.core
@pytest.mark.parametrize("proto_path", ["nvidia_tao_deploy/cv/classification_tf1/specs/experiment_spec.txt"])
def test_protob_based_spec_load(proto_path):
    es = load_proto(proto_path)

    assert es.train_config.preprocess_mode == "caffe"
    assert es.eval_config.enable_center_crop == True
    assert es.model_config.resize_interpolation_method == 0
