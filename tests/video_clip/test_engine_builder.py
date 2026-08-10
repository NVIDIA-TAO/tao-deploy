# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP Engine Builder Unit Tests."""

from types import SimpleNamespace
from unittest.mock import patch

from nvidia_tao_deploy.multimodal.video_clip.engine_builder import VideoCLIPEngineBuilder
from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.multimodal.video_clip.scripts.gen_trt_engine import (
    _apply_fp16_precision_constraints,
)


class TestVideoCLIPEngineBuilderInit:
    """Tests for VideoCLIPEngineBuilder initialization."""

    @patch.object(EngineBuilder, '__init__', return_value=None)
    def test_default_data_format(self, mock_init):
        builder = VideoCLIPEngineBuilder()
        assert builder._data_format == "channels_first"
        mock_init.assert_called_once_with()

    @patch.object(EngineBuilder, '__init__', return_value=None)
    def test_kwargs_forwarded(self, mock_init):
        VideoCLIPEngineBuilder(
            data_format="channels_first",
            workspace=8192,
            max_batch_size=16,
            strongly_typed=True,
        )
        mock_init.assert_called_once_with(
            workspace=8192,
            max_batch_size=16,
            strongly_typed=True,
        )

    def test_inherits_engine_builder(self):
        assert issubclass(VideoCLIPEngineBuilder, EngineBuilder)


class TestFP16PrecisionConstraints:
    """Tests for InternVideo2 FP16 normalization safeguards."""

    def test_selects_only_block_norm_reductions(self):
        builder = VideoCLIPEngineBuilder.__new__(VideoCLIPEngineBuilder)
        selected = [
            "/model/vision_encoder/blocks.0/norm1/Pow",
            "/model/vision_encoder/blocks.0/norm1/ReduceMean",
            "/model/vision_encoder/blocks.23/norm2/Pow",
            "/model/vision_encoder/blocks.23/norm2/ReduceMean",
        ]
        excluded = [
            "/model/vision_encoder/blocks.0/attn/q_norm/Pow",
            "/model/vision_encoder/clip_projector/ReduceMean",
            "/model/text_encoder/transformer.0/LayerNormalization",
        ]
        builder.network = [
            SimpleNamespace(name=name) for name in selected + excluded
        ]

        assert builder.get_fp16_precision_constraints() == {
            name: "fp32" for name in selected
        }

    def test_required_constraints_override_only_matching_entries(self):
        required_name = "/model/vision_encoder/blocks.0/norm1/Pow"
        builder = SimpleNamespace(
            get_fp16_precision_constraints=lambda: {required_name: "fp32"}
        )
        kwargs = {
            "layers_precision": {
                required_name: "fp16",
                "/custom/layer": "fp16",
            }
        }

        count = _apply_fp16_precision_constraints(builder, kwargs)

        assert count == 1
        assert kwargs["layers_precision"] == {
            required_name: "fp32",
            "/custom/layer": "fp16",
        }
