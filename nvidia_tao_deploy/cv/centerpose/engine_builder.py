# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CenterPose TensorRT engine builder."""

import logging
import os

import tensorrt as trt

from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.engine.calibrator import EngineCalibrator
from nvidia_tao_deploy.cv.centerpose.dataloader import CPPredictDataset

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


class CenterPoseEngineBuilder(EngineBuilder):
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(
        self,
        cfg=None,
        **kwargs
    ):
        """Init.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.cfg = cfg

    def set_calibrator(self,
                       inputs=None,
                       calib_cache=None,
                       calib_input=None,
                       calib_num_images=5000,
                       calib_batch_size=8,
                       calib_data_file=None):
        """Simple function to set an int8 calibrator. (Default is ImageBatcher based)

        Args:
            inputs (list): Inputs to the network
            calib_input (str): The path to a directory holding the calibration images.
            calib_cache (str): The path where to write the calibration cache to,
                         or if it already exists, load it from.
            calib_num_images (int): The maximum number of images to use for calibration.
            calib_batch_size (int): The batch size to use for the calibration process.

        Returns:
            No explicit returns.
        """
        logger.info("Calibrating using ImageBatcher")
        self.config.int8_calibrator = EngineCalibrator(calib_cache)
        if not os.path.exists(calib_cache):
            calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
            calib_dtype = trt.nptype(inputs[0].dtype)
            self.config.int8_calibrator.set_image_batcher(
                CPPredictDataset(self.cfg.dataset,
                                 calib_input,
                                 calib_shape,
                                 calib_dtype,
                                 calib_num_images,
                                 exact_batches=True))
