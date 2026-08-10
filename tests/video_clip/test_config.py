# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP deploy config unit tests."""

from nvidia_tao_deploy.config.multimodal.video_clip.default_config import (
    VideoCLIPTrtConfig,
    VideoCLIPModelConfig,
)


def test_default_precision_is_fp32():
    """FP32 remains the conservative default precision."""
    assert VideoCLIPTrtConfig().data_type == "fp32"


def test_default_image_size_and_type():
    m = VideoCLIPModelConfig()
    assert m.image_size == 224
    assert m.type == "internvideo2-clip-l14"
    assert m.canonicalize_text is False
