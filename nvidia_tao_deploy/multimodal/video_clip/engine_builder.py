# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP TensorRT engine builder."""

import logging

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder

logging.basicConfig(
    format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
    level="INFO",
)
logger = logging.getLogger(__name__)


class VideoCLIPEngineBuilder(EngineBuilder):
    """Parses a Video-CLIP ONNX graph and builds a TensorRT engine from it.

    Video-CLIP's combined ONNX has a 5-D vision input (B, T, C, H, W). The base
    ``EngineBuilder`` resolves each ONNX input axis independently — static dims
    pass through and only a dynamic batch axis uses the min/opt/max profile — so
    the 5-D video input needs no special handling here (the ONNX is exported with
    dynamic batch + static T/C/H/W).
    """

    def __init__(self, data_format="channels_first", **kwargs):
        """Init.

        Args:
            data_format (str): Input data format.
        """
        super().__init__(**kwargs)
        self._data_format = data_format

    def _extra_network_flags(self):
        """Request a strongly-typed network for a mixed-precision ONNX.

        TensorRT 10 enables strongly-typed mode via a NETWORK creation flag (not
        just the builder-config flag the base sets); it is required so an AutoCast
        mixed-precision ONNX (vision RMSNorm kept FP32) builds as intended.
        """
        if self._strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
            return 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        return 0

    def get_fp16_precision_constraints(self):
        """Return FP32 constraints for numerically sensitive vision norms."""
        constraints = {}
        for layer in self.network:
            name = layer.name
            is_vision_block = name.startswith(
                "/model/vision_encoder/blocks."
            )
            is_block_norm = "/norm1/" in name or "/norm2/" in name
            is_reduction = name.endswith(("/Pow", "/ReduceMean"))
            if is_vision_block and is_block_norm and is_reduction:
                constraints[name] = "fp32"
        return constraints
