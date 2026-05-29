# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric Learning Recognition convert onnx model to TRT engine."""

import logging
import os

from nvidia_tao_deploy.config.ml_recog.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.initialize_experiments import initialize_gen_trt_engine_experiment
from nvidia_tao_deploy.cv.ml_recog.engine_builder import MLRecogEngineBuilder
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.decorators import monitor_status

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="export", schema=ExperimentConfig
)
@monitor_status(name='ml_recog', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    engine_builder_kwargs, create_engine_kwargs = initialize_gen_trt_engine_experiment(cfg)

    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size

    # INT8 related configs
    img_std = cfg.dataset.pixel_std
    img_mean = cfg.dataset.pixel_mean

    builder = MLRecogEngineBuilder(
        **engine_builder_kwargs,
        workspace=workspace_size // 1024,
        img_std=img_std,
        img_mean=img_mean)

    builder.create_network(cfg.gen_trt_engine.onnx_file, "onnx")
    builder.create_engine(**create_engine_kwargs)


if __name__ == '__main__':
    main()
