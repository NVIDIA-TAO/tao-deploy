# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the model marker without requiring a GPU or deployment runtime."""


def pytest_configure(config):
    """Allow Sparse4D selection with pytest's strict marker validation."""
    config.addinivalue_line("markers", "sparse4d: Sparse4D deployment unit and opt-in GPU tests")
