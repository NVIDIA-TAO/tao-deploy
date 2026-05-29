# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base utility functions for TensorRT inferencer."""

from typing import Optional
import numpy as np
from nvidia_tao_deploy.engine.builder import TRT_8_API

import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt

BINDING_TO_NPTYPE = {
    "Input": np.float32,
    "NMS": np.float32,
    "NMS_1": np.int32,
    "BatchedNMS": np.int32,
    "BatchedNMS_1": np.float32,
    "BatchedNMS_2": np.float32,
    "BatchedNMS_3": np.float32,
    "generate_detections": np.float32,
    "mask_head/mask_fcn_logits/BiasAdd": np.float32,
    "softmax_1": np.float32,
    "input_1": np.float32,
    # D-DETR
    "inputs": np.float32,
    "pred_boxes": np.float32,
    "pred_logits": np.float32,
    "pred_masks": np.float32
}


class HostDeviceMem(object):
    """Clean data structure to handle host/device memory."""

    def __init__(self,
                 host_mem,
                 device_mem,
                 npshape,
                 name: str = None,
                 data_format: str = "channels_first"):
        """Initialize a HostDeviceMem data structure.

        Args:
            host_mem (cuda.pagelocked_empty): A cuda.pagelocked_empty memory buffer.
            device_mem (cuda.mem_alloc): Allocated memory pointer to the buffer in the GPU.
            npshape (tuple): Shape of the input dimensions.

        Returns:
            HostDeviceMem instance.
        """
        self.host = host_mem
        self.device = device_mem
        self.numpy_shape = npshape
        assert data_format in ["channels_first", "channels_last"], (
            f"Invalid data format received. {data_format}"
        )
        self.data_format = data_format
        self.name = name

    def __str__(self):
        """String containing pointers to the TRT Memory."""
        return "Name: " + self.name + "\nHost:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """Return the canonical string representation of the object."""
        return self.__str__()


def do_inference(context, bindings, inputs,
                 outputs, stream,
                 batch_size=1, execute_v2=False,
                 return_raw=False):
    """Generalization for multiple inputs/outputs.

    inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference.
    if execute_v2 or not TRT_8_API:
        if TRT_8_API:
            context.execute_v2(bindings=bindings)
        else:
            context.execute_async_v3(stream_handle=stream.handle)
    else:
        # This is only usable if it's not running in compatibility mode.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()

    if return_raw:
        return outputs

    # Return only the host outputs.
    return [out.host for out in outputs]


def allocate_buffers(engine, context=None, reshape=False, profile_idx: Optional[int] = None):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine
        context (trt.IExecutionContext): Context for dynamic shape engine
        reshape (bool): To reshape host memory or not (FRCNN)

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Dynamic engines: seed unset inputs with MAX profile shape so that
    # output shapes resolve in the main loop below. Inputs whose shape was
    # already narrowed by an earlier set_input_shape (e.g. via _setup_input)
    # are left alone — overwriting them would break per-call batch narrowing
    # used by static-batch and batch-only-dynamic networks.
    if context is not None:
        for _idx in range(engine.num_io_tensors):
            _name = engine.get_tensor_name(_idx)
            if engine.get_tensor_mode(_name) != trt.TensorIOMode.INPUT:
                continue
            _eng_shape = engine.get_tensor_shape(_name)
            if not any(d < 0 for d in _eng_shape):
                continue
            _ctx_shape = context.get_tensor_shape(_name)
            if any(d < 0 for d in _ctx_shape):
                _prof = engine.get_tensor_profile_shape(_name, profile_idx or 0)
                assert len(_prof) == 3, f"Bad profile {_prof}"
                context.set_input_shape(_name, _prof[-1])  # MAX

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    for idx in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(idx)
        is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        tensor_shape = engine.get_tensor_shape(tensor_name)
        if is_input:
            # Any dynamic axis → substitute MAX profile shape for alloc.
            if any(d < 0 for d in tensor_shape):
                profile_shape = engine.get_tensor_profile_shape(tensor_name, profile_idx)
                assert len(profile_shape) == 3, (
                    "There should be min, opt and max shapes."
                    f"There are only {len(profile_shape)} with size {profile_shape}"
                )
                tensor_shape = profile_shape[-1]
        else:
            tensor_shape = context.get_tensor_shape(tensor_name)
        shape_valid = all(dim > 0 for dim in tensor_shape)
        if not shape_valid:
            raise ValueError(f"Tensor {tensor_name} has dynamic shape, but no profile was selected.")

        size = trt.volume(tensor_shape)
        trt_data_type = engine.get_tensor_dtype(tensor_name)
        if tensor_name in BINDING_TO_NPTYPE.keys():
            dtype = BINDING_TO_NPTYPE[tensor_name]
        else:
            if trt.nptype(trt_data_type):
                dtype = np.dtype(trt.nptype(trt_data_type))
            else:
                size = int(size * trt_data_type.itemsize)
                dtype = np.uint8

        host_mem = cuda.pagelocked_empty(size, dtype)
        if reshape and not is_input:
            if engine.has_implicit_batch_dimension:
                target_shape = (engine.max_batch_size, tensor_shape[1], tensor_shape[2], tensor_shape[3])
                host_mem = host_mem.reshape(*target_shape)
            host_mem = host_mem.reshape(*tensor_shape)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Bindings holds addresses to device-allocated memory
        bindings.append(int(device_mem))

        # The tensor_shape we set here becomes the numpy_shape attribute we access during inference
        # For static batch sizes, the inputs and outputs will both have the static batch size
        # For dynamic batch sizes, the inputs will have max_batch_size, while the outputs will have the actual_batch_size
        if is_input:
            inputs.append(HostDeviceMem(host_mem, device_mem, tensor_shape, name=tensor_name))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, tensor_shape, name=tensor_name))

    return inputs, outputs, bindings, stream
