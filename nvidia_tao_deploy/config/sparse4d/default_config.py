# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the batch-one Sparse4D export ABI (not a training spec)."""

from dataclasses import dataclass, field
from typing import List

from nvidia_tao_deploy.config.utils.types import (
    BOOL_FIELD, DATACLASS_FIELD, FLOAT_FIELD, INT_FIELD, LIST_FIELD, STR_FIELD,
)


@dataclass
class ModelConfig:
    """Dimensions must agree with the exported model and prepared frames."""

    num_cameras: int = INT_FIELD(6, valid_min=1)
    input_height: int = INT_FIELD(256, valid_min=1)
    input_width: int = INT_FIELD(448, valid_min=1)
    num_queries: int = INT_FIELD(900, valid_min=1)
    num_temp_instances: int = INT_FIELD(600, valid_min=1)
    embed_dims: int = INT_FIELD(256, valid_min=1)
    anchor_dims: int = INT_FIELD(11, valid_options="11")
    num_classes: int = INT_FIELD(7, valid_min=1)


@dataclass
class PluginConfig:
    """User-supplied MSDA library; no proprietary binary is bundled."""

    path: str = STR_FIELD("", description="Absolute path to a trusted MSDA plugin library.")
    sha256: str = STR_FIELD("", description="Required SHA256 of that exact library.")


@dataclass
class GenTrtEngineConfig:
    """A fixed batch-one/camera-count profile for all seven or eight inputs."""

    results_dir: str = STR_FIELD("")
    gpu_id: int = INT_FIELD(0, valid_min=0)
    onnx_file: str = STR_FIELD("")
    save_engine: str = STR_FIELD("")
    data_type: str = STR_FIELD("fp32", valid_options="fp32,fp16")
    workspace_size: float = FLOAT_FIELD(4.0, valid_min=0.1)


@dataclass
class InferenceConfig:
    """Ordered prepared frames and recurrent decoding settings."""

    results_dir: str = STR_FIELD("")
    num_gpus: int = INT_FIELD(1, valid_options="1")
    gpu_ids: List[int] = field(default_factory=lambda: [0], metadata=LIST_FIELD([0]).metadata)
    trt_engine: str = STR_FIELD("")
    manifest: str = STR_FIELD("", description="JSON manifest of prepared, non-pickle NPZ frames.")
    num_output: int = INT_FIELD(300, valid_min=1)
    score_threshold: float = FLOAT_FIELD(0.05, valid_min=0.0, valid_max=1.0)
    tracking_threshold: float = FLOAT_FIELD(0.2, valid_min=0.0, valid_max=1.0)
    confidence_decay: float = FLOAT_FIELD(0.8, valid_min=0.0, valid_max=1.0)
    max_time_interval: float = FLOAT_FIELD(2.0, valid_min=0.0)
    save_raw: bool = BOOL_FIELD(False)


@dataclass
class ExperimentConfig:
    """Sparse4D's deployment-only schema, discoverable by model_agnostic."""

    model_name: str = STR_FIELD("sparse4d")
    results_dir: str = STR_FIELD("")
    model: ModelConfig = field(default_factory=ModelConfig, metadata=DATACLASS_FIELD(ModelConfig()).metadata)
    plugin: PluginConfig = field(default_factory=PluginConfig, metadata=DATACLASS_FIELD(PluginConfig()).metadata)
    gen_trt_engine: GenTrtEngineConfig = field(
        default_factory=GenTrtEngineConfig, metadata=DATACLASS_FIELD(GenTrtEngineConfig()).metadata)
    inference: InferenceConfig = field(default_factory=InferenceConfig,
                                       metadata=DATACLASS_FIELD(InferenceConfig()).metadata)
