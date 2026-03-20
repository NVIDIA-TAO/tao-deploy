# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLIP ONNX model to TensorRT engine conversion."""

import glob
import os
import shutil

from nvidia_tao_core.config.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)

from nvidia_tao_deploy.multimodal.clip.engine_builder import CLIPEngineBuilder
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.initialize_experiments import (
    initialize_gen_trt_engine_experiment,
)
from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging as logger
from nvidia_tao_deploy.cv.common.utils import is_qdq_quantized_onnx
from nvidia_tao_deploy.utils.decoding import decode_model

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name='clip', mode='gen_trt_engine')
def main(cfg: ExperimentConfig) -> None:
    """Convert CLIP ONNX model to TensorRT engine."""
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

    strongly_typed = False
    if file_format == "onnx":
        strongly_typed = is_qdq_quantized_onnx(tmp_onnx_file)
        if strongly_typed:
            logger.info(
                "QDQ quantized ONNX model detected. "
                "Enabling strongly typed mode."
            )

    logger.info("Building TensorRT engine...")
    builder = CLIPEngineBuilder(
        **engine_builder_kwargs,
        workspace=trt_cfg.tensorrt.workspace_size,
        strongly_typed=strongly_typed,
        data_format="channels_first",
    )
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(**create_engine_kwargs)
    logger.info("Engine saved to: %s", trt_cfg.trt_engine)

    _copy_export_artifacts(trt_cfg.onnx_file, trt_cfg.trt_engine)


def _copy_export_artifacts(onnx_path, engine_path):
    """Copy *_config.yaml and *_tokenizer/ from ONNX dir to engine dir."""
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    engine_dir = os.path.dirname(os.path.abspath(engine_path))

    if os.path.normpath(onnx_dir) == os.path.normpath(engine_dir):
        return

    for pattern, is_dir in [("*_config.yaml", False),
                            ("*_tokenizer", True)]:
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
