# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO Deploy DepthNet Module.

This module provides comprehensive depth estimation capabilities for computer vision applications
using NVIDIA's TAO (Train, Adapt, Optimize) framework. It includes tools for inference, evaluation,
and deployment of depth estimation models optimized with TensorRT for high-performance inference
on NVIDIA GPUs.

The module supports various depth estimation models including:
- NvDepthAnythingV2: Advanced monocular depth estimation model
- Foundation Stereo: Stereo-based depth estimation model

Key Features:
- TensorRT-optimized inference for real-time performance
- Comprehensive evaluation metrics for depth accuracy (abs_rel, d1, bp1, bp2, bp3, epe)
- Support for both monocular and stereo depth estimation
- Visualization tools for depth maps with color mapping
- Batch processing capabilities for improved throughput
- Integration with TAO framework for model optimization
- Support for multiple input formats (PFM, PNG)
- Configurable preprocessing and post-processing pipelines

Usage:
    The module can be used for:
    - Real-time depth estimation from single images or stereo pairs
    - Model evaluation and benchmarking with ground truth data
    - Depth map visualization and analysis
    - Integration into larger computer vision pipelines
    - TensorRT engine generation from ONNX models

Command Line Interface:
    ```bash
    # Generate TensorRT engine
    depth_net gen_trt_engine --config-file config.yaml

    # Perform inference
    depth_net inference --config-file config.yaml

    # Evaluate model
    depth_net evaluate --config-file config.yaml
    ```

Dependencies:
    - TensorRT for optimized inference
    - OpenCV for image processing
    - NumPy for numerical operations
    - Matplotlib for visualization
    - TAO framework for model management
    - Hydra for configuration management
"""
