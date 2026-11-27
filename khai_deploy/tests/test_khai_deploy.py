# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import khai_deploy
from khai.core.cv import container


def test_import():
    assert khai_deploy.__name__ == "khai_deploy"
    assert container.CONTRACT_VERSION >= 1
