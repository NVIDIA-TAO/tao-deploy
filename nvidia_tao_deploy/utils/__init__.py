# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO Deploy utils"""

from packaging import version
import tensorrt as trt


def LEGACY_API_MODE():
    """Check if API is to be run TensorRT 8.x compatible."""
    return version.Version(trt.__version__) < version.Version("9.0.0")
