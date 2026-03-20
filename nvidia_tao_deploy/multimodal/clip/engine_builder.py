# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
