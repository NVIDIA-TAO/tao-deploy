# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Engine Builder Unit Tests."""

from unittest.mock import patch

from nvidia_tao_deploy.multimodal.clip.engine_builder import CLIPEngineBuilder
from nvidia_tao_deploy.engine.builder import EngineBuilder


class TestCLIPEngineBuilderInit:
    """Tests for CLIPEngineBuilder initialization."""

    @patch.object(EngineBuilder, '__init__', return_value=None)
    def test_default_data_format(self, mock_init):
        builder = CLIPEngineBuilder()
        assert builder._data_format == "channels_first"
        mock_init.assert_called_once_with()

    @patch.object(EngineBuilder, '__init__', return_value=None)
    def test_custom_data_format(self, mock_init):
        builder = CLIPEngineBuilder(data_format="channels_last")
        assert builder._data_format == "channels_last"

    @patch.object(EngineBuilder, '__init__', return_value=None)
    def test_kwargs_forwarded(self, mock_init):
        CLIPEngineBuilder(
            data_format="channels_first",
            workspace=4096,
            max_batch_size=32,
            strongly_typed=True,
        )
        mock_init.assert_called_once_with(
            workspace=4096,
            max_batch_size=32,
            strongly_typed=True,
        )

    def test_inherits_engine_builder(self):
        assert issubclass(CLIPEngineBuilder, EngineBuilder)
