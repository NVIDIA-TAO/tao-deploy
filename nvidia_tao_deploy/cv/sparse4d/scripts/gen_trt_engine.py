# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a Sparse4D TensorRT engine."""

from pathlib import Path

from nvidia_tao_deploy.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.sparse4d.engine_builder import build_engine


@hydra_runner(config_path=str(Path(__file__).resolve().parents[1] / "specs"),
              config_name="gen_trt_engine", schema=ExperimentConfig)
@monitor_status(name="sparse4d", mode="gen_trt_engine")
def main(cfg: ExperimentConfig):
    """Run engine generation with TAO status and experiment logging."""
    try:
        build_engine(cfg)
    except Exception as error:
        # Native ONNX/PyCUDA errors do not derive from RuntimeError. Translate at
        # this action boundary so monitor_status records FAILURE and re-raises.
        # Preserve the cause; process-control BaseExceptions are not intercepted.
        raise ValueError(f"Sparse4D engine generation failed: {error}") from error


if __name__ == "__main__":
    main()
