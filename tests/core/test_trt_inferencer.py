# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRTInferencer.define_input_tensor() on the TensorRT 8.x API path."""

from unittest import mock

import numpy as np
import pytest

from nvidia_tao_deploy.inferencer import trt_inferencer


class _Inferencer(trt_inferencer.TRTInferencer):
    """Minimal concrete inferencer; no engine is loaded or allocated."""

    def infer(self, imgs):
        """Not used."""

    def draw_bbox(self, *args, **kwargs):
        """Not used."""

    def __del__(self):
        """Nothing to release."""


def _make_inferencer(implicit_batch):
    inf = _Inferencer.__new__(_Inferencer)
    inf.data_format = "channel_first"
    inf.engine = mock.Mock(has_implicit_batch_dimension=implicit_batch, max_batch_size=4)
    inf.context = mock.Mock()
    inf.input_tensors = []
    return inf


@pytest.mark.parametrize(
    "tensor_shape, implicit_batch, expected",
    [
        ((3, 544, 960), True, "uff"),
        ((1, 3, 544, 960), False, "onnx"),
    ],
)
def test_trt8_etlt_type(monkeypatch, tensor_shape, implicit_batch, expected):
    """On the TensorRT 8.x path, implicit-batch (UFF) engines get etlt_type "uff"."""
    monkeypatch.setattr(trt_inferencer, "TRT_8_API", True)
    inf = _make_inferencer(implicit_batch)

    inf.define_input_tensor("input_1", np.float32, tensor_shape, None, 1)

    assert inf.etlt_type == expected
    if implicit_batch:
        assert inf.max_batch_size == 4
