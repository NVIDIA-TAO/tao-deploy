# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP configuration unit tests."""

from pathlib import Path

from omegaconf import OmegaConf

from nvidia_tao_deploy.config.multimodal.clip.default_config import (
    CLIPExperimentConfig,
    CLIPValDataConfig,
)


def test_attribute_pairs_file_structured_config():
    """CLIP dataset items preserve attribute_pairs_file through OmegaConf."""
    config = OmegaConf.structured(CLIPValDataConfig())
    config = OmegaConf.merge(
        config,
        {
            "datasets": [
                {
                    "image_dir": "/data/images",
                    "attribute_pairs_file": "/data/test_pairs.json",
                }
            ]
        },
    )

    serialized = OmegaConf.to_container(config, resolve=True)

    assert serialized["datasets"][0]["attribute_pairs_file"] == "/data/test_pairs.json"


def test_default_spec_accepts_attribute_pairs_file():
    """CLIP default spec preserves the optional PAS pairs path."""
    spec_path = Path(__file__).parents[2].joinpath(
        "nvidia_tao_deploy",
        "multimodal",
        "clip",
        "specs",
        "experiment_spec.yaml",
    )
    spec = OmegaConf.load(spec_path)
    config = OmegaConf.merge(
        OmegaConf.structured(CLIPExperimentConfig()),
        spec,
    )

    assert config.dataset.val.datasets[0].attribute_pairs_file is None
