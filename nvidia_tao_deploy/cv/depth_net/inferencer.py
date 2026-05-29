# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT Engine class for DepthNet inference.

This module provides the DepthNetInferencer class which handles TensorRT-optimized
inference for depth estimation models. It supports both monocular and stereo depth
estimation with efficient batch processing capabilities.
"""

import logging
import numpy as np
import tensorrt as trt

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference

logger = logging.getLogger(__name__)


def trt_output_process(y_encoded):
    """
    Process TensorRT model outputs to proper numpy array format.

    This function reshapes the raw TensorRT outputs into the expected numpy array
    format for further processing. It handles the conversion from TensorRT's
    internal representation to standard numpy arrays by extracting the host data
    and reshaping it according to the tensor's numpy shape specification.

    Args:
        y_encoded (list): List of TensorRT outputs in numpy format, where each
            element contains the raw output data and shape information. Each
            element should have 'host' and 'numpy_shape' attributes.

    Returns:
        list: List of processed numpy arrays with proper shapes for each output.
            The arrays are ready for further processing and analysis.

    Raises:
        AttributeError: If TensorRT output objects don't have required attributes.
        ValueError: If the output shapes are invalid or incompatible.

    Example:
        >>> raw_outputs = trt_inferencer.get_raw_outputs()
        >>> processed_outputs = trt_output_process(raw_outputs)
        >>> depth_map = processed_outputs[0]  # First output is depth map
        >>> print(f"Depth map shape: {depth_map.shape}")
    """
    # Host buffer is MAX-sized; slice to runtime volume before reshape.
    return [
        np.reshape(out.host[:int(np.prod(out.numpy_shape))], out.numpy_shape)
        for out in y_encoded
    ]


class DepthNetInferencer(TRTInferencer):
    """
    TensorRT-optimized inferencer for DepthNet models.

    This class provides high-performance inference capabilities for depth estimation
    models using TensorRT optimization. It supports both monocular and stereo depth
    estimation, with automatic handling of different input configurations.

    The inferencer can process:
    - Single images for monocular depth estimation
    - Stereo image pairs for stereo depth estimation
    - Batch processing for improved throughput

    Attributes:
        input_tensors (list): List of input tensor specifications from the TRT engine.
        output_tensors (list): List of output tensor specifications from the TRT engine.
        max_batch_size (int): Maximum batch size supported by the TRT engine.
        context (trt.IExecutionContext): TensorRT execution context for inference.
    """

    def __init__(self, engine_path, input_shape=None, batch_size=None, data_format="channel_first"):
        """
        Initialize the DepthNet TensorRT inferencer.

        Args:
            engine_path (str): Path to the TensorRT engine file (.trt or .engine).
            input_shape (tuple, optional): Input shape specification for dynamic shape engines.
                Format: (batch, channel, height, width). Defaults to None.
            batch_size (int, optional): Batch size for dynamic shape engines.
                If None, uses the engine's default batch size. Defaults to None.
            data_format (str, optional): Data format for input tensors.
                Must be either "channel_first" (NCHW) or "channel_last" (NHWC).
                Defaults to "channel_first".

        Raises:
            FileNotFoundError: If the engine file does not exist.
            RuntimeError: If the TensorRT engine cannot be loaded or initialized.

        Example:
            >>> inferencer = DepthNetInferencer(
            ...     engine_path="depth_model.trt",
            ...     batch_size=4,
            ...     data_format="channel_first"
            ... )
        """
        # Dynamic engine: pre-resolve MAX profile shape for buffer alloc.
        if input_shape is None:
            try:
                _logger = trt.Logger(trt.Logger.WARNING)
                _runtime = trt.Runtime(_logger)
                with open(engine_path, 'rb') as _f:
                    _engine = _runtime.deserialize_cuda_engine(_f.read())
                _input_name = _engine.get_tensor_name(0)
                _engine_shape = tuple(int(d) for d in _engine.get_tensor_shape(_input_name))
                if any(d < 0 for d in _engine_shape):
                    # Profile is (min, opt, max); index 2 = MAX.
                    _profile_shape = _engine.get_tensor_profile_shape(_input_name, 0)
                    input_shape = tuple(int(d) for d in _profile_shape[2])
                    logger.info(
                        "DepthNetInferencer: dynamic engine — using MAX shape %s.",
                        input_shape,
                    )
                del _engine
                del _runtime
            except Exception as _e:
                logger.warning(
                    "DepthNetInferencer: dynamic-shape pre-resolve failed (%s).", _e,
                )

        # Load TRT engine
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
            reshape=False
        )

    def infer(self, imgs):
        """
        Perform inference on input images to generate depth estimates.

        This method handles both monocular and stereo depth estimation depending on
        the input format. For stereo estimation, it expects a dictionary with
        'left_image' and 'right_image' keys. For monocular estimation, it expects
        a single image or batch of images.

        Args:
            imgs (Union[dict, np.ndarray, list]): Input images for inference.
                - For stereo: dict with keys 'left_image' and 'right_image'
                - For monocular: single image array or list of image arrays
                Images should be preprocessed and in the correct format for the model.

        Returns:
            np.ndarray: Predicted depth maps with shape (batch_size, height, width)
                or (batch_size, 1, height, width) depending on the model output format.

        Raises:
            ValueError: If input format is invalid or images are not properly formatted.
            RuntimeError: If inference fails due to TensorRT errors.

        Example:
            # Monocular depth estimation
            >>> single_image = np.random.rand(1, 3, 480, 640).astype(np.float32)
            >>> depth_map = inferencer.infer(single_image)

            # Stereo depth estimation
            >>> stereo_input = {
            ...     'left_image': np.random.rand(1, 3, 480, 640).astype(np.float32),
            ...     'right_image': np.random.rand(1, 3, 480, 640).astype(np.float32)
            ... }
            >>> depth_map = inferencer.infer(stereo_input)
        """
        if isinstance(imgs, dict):
            # 2 inputs: [left_image, right_image] if imgs is dict
            left_images = imgs["left_image"]
            right_images = imgs["right_image"]
            inputs = (left_images, right_images)
        else:
            # 1 input: [imgs] if imgs is not dict
            inputs = [imgs]

        # Per-call shape binding for dynamic engines.
        is_dynamic_engine = False
        for img, tensor in zip(inputs, self.input_tensors):
            if getattr(tensor, "optimization_profile", None) is not None:
                is_dynamic_engine = True
                actual = tuple(int(d) for d in img.shape)
                self.context.set_input_shape(tensor.tensor_name, actual)

        # inputs is already a list from above
        self._copy_input_to_host(inputs)

        # ...fetch model outputs...
        # 1 named results: [outputs]
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True)

        # Output buffer is MAX-sized; refresh numpy_shape from live context.
        if is_dynamic_engine:
            for out in results:
                runtime_shape = tuple(
                    int(d) for d in self.context.get_tensor_shape(out.name)
                )
                out.numpy_shape = runtime_shape

        # Process TRT outputs to proper format
        results = trt_output_process(results)
        return results[0]
