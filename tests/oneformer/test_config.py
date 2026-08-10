# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneFormer evaluation schema compatibility tests."""

from nvidia_tao_deploy.config.oneformer.evaluate import OneFormerEvaluateConfig


def test_evaluation_task_matches_tao_pytorch_contract():
    """Deploy accepts the task selector emitted by tao-pytorch."""
    config = OneFormerEvaluateConfig()

    assert config.task == "semantic"
    assert config.__dataclass_fields__["task"].metadata["valid_options"] == (
        "semantic,panoptic"
    )
