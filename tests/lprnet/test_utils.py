# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test lprnet model builder."""

import os
import pytest

from nvidia_tao_deploy.cv.lprnet.utils import decode_ctc_conf

# Temporarily skipping this test on CI.
# to be fixed in another MR.
@pytest.mark.lprnet
@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
def test_ctc_decoder():
    classes = ["a", "b", "c"]
    blank_id = 3

    pred_id = [[0, 0, 0, 3, 1, 2]]
    pred_conf = [[0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]
    expected_lp = "abc"
    decoded_lp, _ = decode_ctc_conf((pred_id, pred_conf), classes, blank_id)

    assert expected_lp == decoded_lp[0]
