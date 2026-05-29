# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet convert onnx model to TRT engine."""

import logging
import os

from nvidia_tao_deploy.config.visual_changenet.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.initialize_experiments import initialize_gen_trt_engine_experiment
from nvidia_tao_deploy.cv.common.utils import is_qdq_quantized_onnx
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.engine.builder import EngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name='visual_changenet', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    engine_builder_kwargs, create_engine_kwargs = initialize_gen_trt_engine_experiment(cfg)

    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size

    # For ViT-L, override workspace to be larger. #TODO: Check if needed.
    if cfg.model.backbone == "vit_large_dinov2" and workspace_size < 24080:
        logger.warning("Overriding workspace_size from {} to 20480 due to ViT's model size".format(workspace_size))
        workspace_size = 20480

    # Detect if the ONNX model is quantized
    strongly_typed = is_qdq_quantized_onnx(cfg.gen_trt_engine.onnx_file)
    if strongly_typed:
        logger.info("QDQ quantized ONNX model detected. Enabling strongly typed mode.")

    builder = EngineBuilder(**engine_builder_kwargs,
                            workspace=workspace_size,
                            strongly_typed=strongly_typed)
    builder.create_network(cfg.gen_trt_engine.onnx_file, 'onnx')
    builder.create_engine(**create_engine_kwargs)


if __name__ == '__main__':
    main()
