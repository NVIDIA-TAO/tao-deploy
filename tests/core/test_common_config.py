# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration shared by TAO Deploy models."""

from nvidia_tao_deploy.config.common.common_config import CheckpointerConfig, TrainConfig


def test_replace_periodic_checkpointing_is_disabled_by_default():
    """Keep the deploy schema aligned with the opt-in PyTorch setting."""
    config = TrainConfig().checkpointer
    field = type(config).__dataclass_fields__["replace_periodic"]

    assert isinstance(config, CheckpointerConfig)
    assert config.enable_topk is False
    assert config.replace_periodic is False
    assert field.metadata["default_value"] is False
    assert field.metadata["value_type"] == "bool"
