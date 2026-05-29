# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UFF module importer."""

from packaging import version
import tensorrt as trt

UFF_ENABLED = version.Version(trt.__version__) < version.Version("9.0.0")
MetaGraph = None

# UFF is not enabled from TensorRT 9.x and above.
if UFF_ENABLED:
    try:
        from uff.model.uff_pb2 import MetaGraph  # pylint: disable=unused-import
    except ImportError:
        print("Loading uff directly from the package source code")
        print("UFF usage is going to be deprecated after TensorRT 10.x. Please move to using ONNX models for TAO > 6.1.0")
        # @scha: To disable tensorflow import issue
        import importlib
        import types

        # TODO @vpraveen: use importlib.util.find_spec() instead of pkgutil.get_loader()
        package = importlib.util.find_spec("uff")
        # Returns __init__.py path
        src_code = package.get_filename().replace('__init__.py', 'model/uff_pb2.py')

        loader = importlib.machinery.SourceFileLoader('helper', src_code)
        helper = types.ModuleType(loader.name)
        loader.exec_module(helper)
        MetaGraph = helper.MetaGraph
