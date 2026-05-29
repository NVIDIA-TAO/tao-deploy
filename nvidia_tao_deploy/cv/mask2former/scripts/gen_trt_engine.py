# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask2former convert onnx model to TRT engine."""

import logging
import os

from nvidia_tao_deploy.config.mask2former.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.initialize_experiments import initialize_gen_trt_engine_experiment
from nvidia_tao_deploy.cv.common.utils import is_qdq_quantized_onnx
from nvidia_tao_deploy.cv.mask2former.engine_builder import Mask2formerEngineBuilder

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="gen_trt_engine", schema=ExperimentConfig
)
@monitor_status(name='mask2former', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert onnx model to TRT engine."""
    engine_builder_kwargs, create_engine_kwargs = initialize_gen_trt_engine_experiment(cfg)

    workspace_size = cfg.gen_trt_engine.tensorrt.workspace_size

    # Detect if the ONNX model is quantized
    strongly_typed = is_qdq_quantized_onnx(cfg.gen_trt_engine.onnx_file)
    if strongly_typed:
        logger.info("QDQ quantized ONNX model detected. Enabling strongly typed mode.")

    builder = Mask2formerEngineBuilder(
        **engine_builder_kwargs,
        workspace=workspace_size // 1024,
        img_std=None,
        strongly_typed=strongly_typed)

    builder.create_network(cfg.gen_trt_engine.onnx_file)
    create_engine_kwargs["layers_precision"] = {
        "/sem_seg_head/predictor/transformer_cross_attention_layers": "fp32",
        "/post_processor/Div": "fp32",
        "/post_processor/ReduceSum": "fp32",
        "/post_processor/Add": "fp32",
    }
    builder.create_engine(**create_engine_kwargs)

    print(f"TensorRT engine was saved at {cfg.gen_trt_engine.trt_engine}.")


if __name__ == '__main__':
    main()
