# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration test, generating all probe tensors/graphs at runtime."""

import os
from types import SimpleNamespace

import pytest

from nvidia_tao_deploy.cv.sparse4d.runtime import load_plugin


@pytest.mark.skipif(not os.getenv("SPARSE4D_PLUGIN_PATH"), reason="Set plugin path/hash in a TensorRT GPU environment")
def test_msda_plugin_bounds_and_nonfinite():
    """Gate both compute precisions using the explicitly selected trusted binary."""
    from nvidia_tao_deploy.cv.sparse4d.plugin_check import verify_plugin  # pylint: disable=import-outside-toplevel
    handle = load_plugin(SimpleNamespace(path=os.environ["SPARSE4D_PLUGIN_PATH"],
                                         sha256=os.environ["SPARSE4D_PLUGIN_SHA256"]))
    assert verify_plugin() == {"float32": "passed", "float16": "passed"}
    del handle
