# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint module for DepthNet operations.

This module provides the main command-line interface for DepthNet operations in the
TAO Deploy framework. It serves as the primary entry point for all DepthNet-related
tasks including model conversion, inference, and evaluation.

The module exports the main entrypoint function that handles:
- Command-line argument parsing
- Task routing to appropriate submodules
- Integration with the TAO framework's hydra-based configuration system

Supported Operations:
- gen_trt_engine: Convert ONNX models to TensorRT engines
- inference: Perform high-performance inference on images
- evaluate: Evaluate model performance with comprehensive metrics

Usage:
    The module is typically used through the command-line interface:
    ```bash
    depth_net <operation> --config-file config.yaml
    ```

Dependencies:
    - argparse for command-line parsing
    - nvidia_tao_deploy.cv.common.entrypoint for hydra integration
    - nvidia_tao_deploy.cv.depth_net.scripts for operation implementations
"""
