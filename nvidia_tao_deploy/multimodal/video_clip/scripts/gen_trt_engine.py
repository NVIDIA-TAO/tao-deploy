# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP ONNX model to TensorRT engine conversion."""

import glob
import os
import shutil

from nvidia_tao_deploy.config.multimodal.video_clip.default_config import (
    VideoCLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_deploy.multimodal.video_clip.engine_builder import VideoCLIPEngineBuilder
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.initialize_experiments import (
    initialize_gen_trt_engine_experiment,
)
from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging as logger
from nvidia_tao_deploy.cv.common.utils import is_qdq_quantized_onnx
from nvidia_tao_deploy.utils.decoding import decode_model

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _apply_fp16_precision_constraints(builder, create_engine_kwargs):
    """Keep sensitive vision normalization reductions in FP32."""
    required = builder.get_fp16_precision_constraints()
    configured = dict(create_engine_kwargs.get("layers_precision") or {})
    configured.update(required)
    create_engine_kwargs["layers_precision"] = configured
    return len(required)


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name='video_clip', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert a Video-CLIP ONNX model to a TensorRT engine."""
    trt_cfg = cfg.gen_trt_engine
    logger.info("ONNX file: %s", trt_cfg.onnx_file)
    logger.info("Output engine: %s", trt_cfg.trt_engine)
    logger.info("Data type: %s", trt_cfg.tensorrt.data_type)
    logger.info(
        "Batch size: min=%d, opt=%d, max=%d",
        trt_cfg.tensorrt.min_batch_size,
        trt_cfg.tensorrt.opt_batch_size,
        trt_cfg.tensorrt.max_batch_size,
    )

    tmp_onnx_file, file_format = decode_model(trt_cfg.onnx_file)
    logger.info("Decoded model format: %s", file_format)

    engine_builder_kwargs, create_engine_kwargs = (
        initialize_gen_trt_engine_experiment(cfg)
    )

    strongly_typed = bool(getattr(trt_cfg.tensorrt, "strongly_typed", False))
    if strongly_typed:
        logger.info(
            "Strongly-typed mode enabled via config for a mixed-precision ONNX "
            "(e.g. ModelOpt AutoCast keeping the vision RMSNorm in FP32)."
        )
    elif file_format == "onnx":
        strongly_typed = is_qdq_quantized_onnx(tmp_onnx_file)
        if strongly_typed:
            logger.info(
                "QDQ quantized ONNX model detected. Enabling strongly typed mode."
            )

    logger.info("Building TensorRT engine...")
    builder = VideoCLIPEngineBuilder(
        **engine_builder_kwargs,
        workspace=trt_cfg.tensorrt.workspace_size,
        strongly_typed=strongly_typed,
        data_format="channels_first",
    )
    builder.create_network(tmp_onnx_file, file_format)
    if str(trt_cfg.tensorrt.data_type).lower() == "fp16" and not strongly_typed:
        num_constraints = _apply_fp16_precision_constraints(
            builder, create_engine_kwargs
        )
        if num_constraints:
            logger.info(
                "Pinned %d vision block normalization layers to FP32.",
                num_constraints,
            )
        else:
            logger.warning(
                "No decomposed vision block normalization layers were found "
                "for FP16 precision constraints."
            )
    builder.create_engine(**create_engine_kwargs)
    logger.info("Engine saved to: %s", trt_cfg.trt_engine)

    _copy_export_artifacts(trt_cfg.onnx_file, trt_cfg.trt_engine)


def _copy_export_artifacts(onnx_path, engine_path):
    """Copy *_config.yaml and *_tokenizer/ from the ONNX dir to the engine dir."""
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    engine_dir = os.path.dirname(os.path.abspath(engine_path))

    if os.path.normpath(onnx_dir) == os.path.normpath(engine_dir):
        return

    for pattern, is_dir in [("*_config.yaml", False), ("*_tokenizer", True)]:
        for src in glob.glob(os.path.join(onnx_dir, pattern)):
            name = os.path.basename(src)
            dst = os.path.join(engine_dir, name)
            if os.path.exists(dst):
                continue
            if is_dir and os.path.isdir(src):
                shutil.copytree(src, dst)
            elif not is_dir and os.path.isfile(src):
                shutil.copy2(src, dst)
            logger.info("Copied %s -> %s", src, dst)


if __name__ == '__main__':
    main()
