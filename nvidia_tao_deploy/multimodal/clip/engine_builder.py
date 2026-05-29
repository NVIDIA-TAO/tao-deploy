# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP TensorRT engine builder."""

import logging

from nvidia_tao_deploy.engine.builder import EngineBuilder

logging.basicConfig(
    format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
    level="INFO",
)
logger = logging.getLogger(__name__)


class CLIPEngineBuilder(EngineBuilder):
    """Parses a CLIP ONNX graph and builds a TensorRT engine from it."""

    def __init__(self, data_format="channels_first", **kwargs):
        """Init.

        Args:
            data_format (str): Input data format.
        """
        super().__init__(**kwargs)
        self._data_format = data_format
