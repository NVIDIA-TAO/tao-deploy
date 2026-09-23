# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infer an ordered sequence of prepared Sparse4D frames."""

from pathlib import Path

from nvidia_tao_deploy.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.sparse4d.inferencer import run_inference


@hydra_runner(config_path=str(Path(__file__).resolve().parents[1] / "specs"),
              config_name="inference", schema=ExperimentConfig)
@monitor_status(name="sparse4d", mode="inference")
def main(cfg: ExperimentConfig):
    """Run recurrent inference with TAO status and experiment logging."""
    try:
        run_inference(cfg)
    except RuntimeError as error:
        # Route TensorRT/PyCUDA failures through the shared status error handler.
        raise ValueError(f"Sparse4D inference failed: {error}") from error


if __name__ == "__main__":
    main()
