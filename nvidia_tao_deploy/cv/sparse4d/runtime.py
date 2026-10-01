# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT 10 named-tensor execution, with explicit multi-input shapes."""

import ctypes
from pathlib import Path

import numpy as np

from nvidia_tao_deploy.cv.sparse4d.contract import sha256


_LOGGER = None


def trt_logger(trt):
    """Keep one logger alive for every TensorRT object in the process."""
    global _LOGGER  # pylint: disable=global-statement
    if _LOGGER is None:
        _LOGGER = trt.Logger(trt.Logger.WARNING)
    return _LOGGER


def load_plugin(config):
    """Load a trusted, hash-pinned library before parsing/deserializing engines."""
    path = Path(config.path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("plugin.path must be an absolute path to a trusted MSDA shared library")
    digest = sha256(path)
    if len(config.sha256) != 64 or digest != config.sha256.lower():
        raise ValueError("plugin.sha256 must match the selected MSDA library")
    # TensorRT must be initialized before static plugin creators are registered.
    # Loading some plugin builds before importing TensorRT can segfault.
    trt, _ = trt_modules()
    trt.init_libnvinfer_plugins(trt_logger(trt), "")
    # Keep the returned CDLL alive for the entire builder/runtime lifetime.
    return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def trt_modules():
    """Import GPU dependencies only when an engine is actually used."""
    import pycuda.autoinit  # noqa pylint: disable=import-outside-toplevel,unused-import
    import pycuda.driver as cuda  # pylint: disable=import-outside-toplevel
    import tensorrt as trt  # pylint: disable=import-outside-toplevel
    if int(trt.__version__.split(".", maxsplit=1)[0]) != 10:
        raise ValueError("Sparse4D currently supports TensorRT 10.x only")
    return trt, cuda


class EngineRunner:
    """Own a runtime, context, stream and reusable exact-shape CUDA buffers."""

    def __init__(self, serialized):
        """The caller must load the MSDA plugin first."""
        self.buffers = {}
        self.trt, self.cuda = trt_modules()
        self.logger = trt_logger(self.trt)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise ValueError("Could not deserialize engine; check plugin, GPU and TensorRT versions")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise ValueError("Could not create TensorRT execution context")
        self.stream = self.cuda.Stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in self.names
                            if self.engine.get_tensor_mode(n) == self.trt.TensorIOMode.INPUT]

    def __enter__(self):
        """Return the resource-owning runner."""
        return self

    def __exit__(self, *_):
        """Release device buffers even when execution or validation fails."""
        self.close()

    def close(self):
        """Synchronize before freeing buffers potentially still in use."""
        try:
            self.stream.synchronize()
        finally:
            for _, device in self.buffers.values():
                device.free()
            self.buffers.clear()

    def __call__(self, inputs):
        """Set every input shape before allocating any output tensor."""
        if set(inputs) != set(self.input_names):
            raise ValueError(f"Expected engine inputs {self.input_names}, got {sorted(inputs)}")
        for name, value in inputs.items():
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            if value.dtype != dtype or not value.flags.c_contiguous:
                raise ValueError(f"{name} must be contiguous {dtype}")
            if not self.context.set_input_shape(name, value.shape):
                raise ValueError(f"Shape {value.shape} rejected for {name}")
        missing = self.context.infer_shapes()
        if missing:
            raise ValueError(f"Unresolved engine tensor shapes: {missing}")
        for name in self.names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise ValueError(f"Unsupported data-dependent shape for {name}: {shape}")
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            old = self.buffers.get(name)
            if old is None or old[0].shape != shape:
                if old is not None:
                    old[1].free()
                host = self.cuda.pagelocked_empty(shape, dtype)
                self.buffers[name] = (host, self.cuda.mem_alloc(host.nbytes))
            host, device = self.buffers[name]
            if not self.context.set_tensor_address(name, int(device)):
                raise ValueError(f"Could not bind {name}")
            if name in inputs:
                np.copyto(host, inputs[name], casting="no")
                self.cuda.memcpy_htod_async(device, host, self.stream)
        if not self.context.execute_async_v3(self.stream.handle):
            raise ValueError("TensorRT execute_async_v3 failed")
        for name, (host, device) in self.buffers.items():
            if name not in inputs:
                self.cuda.memcpy_dtoh_async(host, device, self.stream)
        self.stream.synchronize()
        return {name: host.copy() for name, (host, _) in self.buffers.items() if name not in inputs}
