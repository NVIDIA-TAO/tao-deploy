# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO Deploy DepthNet scripts module.

This module contains the main operational scripts for DepthNet deployment and evaluation.
It provides command-line interfaces for model conversion, inference, and evaluation
tasks using TensorRT optimization.

Available Scripts:
- gen_trt_engine.py: Convert ONNX models to TensorRT engines
- inference.py: Perform high-performance inference on images
- evaluate.py: Evaluate model performance with comprehensive metrics

Each script supports:
- Hydra-based configuration management
- TensorRT optimization for NVIDIA GPUs
- Batch processing capabilities
- Comprehensive logging and monitoring
- Integration with TAO framework

Usage:
    Scripts can be executed directly or through the main depth_net entrypoint:
    ```bash
    # Direct execution
    python scripts/inference.py --config-file config.yaml

    # Through entrypoint
    depth_net inference --config-file config.yaml
    ```
"""
