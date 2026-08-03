# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for DINOv3 backbone options in tao-deploy downstream configs (bug 6465432).

DINOv3 backbones are registered as selectable backbones for the segformer and
visual_changenet tasks (nvidia_tao_pytorch/config/{segformer,visual_changenet}/
default_config.py list them in the backbone ``type`` valid_options). The
tao-deploy mirror of those configs stopped at the nvdinov2 backbones, so the
deploy-side gen_trt_engine spec rejected a DINOv3-backboned model. This guards
against that mirror falling behind again.
"""
from dataclasses import fields

import pytest

from nvidia_tao_deploy.config.segformer.default_config import (
    BackboneConfig as SegformerBackboneConfig,
)
from nvidia_tao_deploy.config.visual_changenet.default_config import (
    BackboneConfig as ChangeNetBackboneConfig,
)

# The DINOv3 backbones registered for dense downstream tasks (segformer /
# visual_changenet) in tao-pytorch. 7B is intentionally excluded - it is not a
# registered downstream backbone.
DINOV3_DOWNSTREAM_BACKBONES = [
    "vit_small_dinov3",
    "vit_small_plus_dinov3",
    "vit_base_dinov3",
    "vit_large_dinov3",
    "vit_huge_plus_dinov3",
]


def _backbone_type_options(backbone_config_cls):
    """Return the backbone ``type`` field's valid_options as a list of strings."""
    (type_field,) = [f for f in fields(backbone_config_cls) if f.name == "type"]
    return type_field.metadata["valid_options"].split(",")


@pytest.mark.core
@pytest.mark.parametrize(
    "backbone_config_cls",
    [SegformerBackboneConfig, ChangeNetBackboneConfig],
    ids=["segformer", "visual_changenet"],
)
def test_dinov3_backbones_present(backbone_config_cls):
    options = _backbone_type_options(backbone_config_cls)
    missing = [b for b in DINOV3_DOWNSTREAM_BACKBONES if b not in options]
    assert not missing, f"deploy backbone enum missing DINOv3 options: {missing}"
