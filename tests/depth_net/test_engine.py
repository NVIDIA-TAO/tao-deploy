# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet Engine Building and Inferencer Unit Tests."""

import pytest
import os
import os
import tensorrt as trt
import cv2
import glob
import tempfile

from nvidia_tao_deploy.engine.builder import EngineBuilder
from nvidia_tao_deploy.cv.depth_net.inferencer import DepthNetInferencer
from nvidia_tao_deploy.cv.depth_net.dataloader import DepthNetDataLoader


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"

@pytest.fixture(scope="session", params=["fp32"])
def mono_engine(tmp_path_factory, request):
    data_type = request.param
    onnx_file = os.path.join(model_path, "depth_net/deployable_relative_depthanythingv2_large_v1.0.onnx")
    
    # Check if model file exists
    if not os.path.exists(onnx_file):
        model_dir = os.path.dirname(onnx_file)
        print(f"\n{'='*80}")
        print(f"ERROR: Model file not found: {onnx_file}")
        print(f"{'='*80}")
        if os.path.exists(model_dir):
            print(f"\nContents of {model_dir}:")
            try:
                contents = os.listdir(model_dir)
                if contents:
                    for item in sorted(contents):
                        item_path = os.path.join(model_dir, item)
                        if os.path.isdir(item_path):
                            print(f"  [DIR]  {item}")
                        else:
                            size = os.path.getsize(item_path)
                            print(f"  [FILE] {item} ({size} bytes)")
                else:
                    print("  (directory is empty)")
            except Exception as e:
                print(f"  Error listing directory: {e}")
        else:
            print(f"\nDirectory does not exist: {model_dir}")
            parent_dir = os.path.dirname(model_dir)
            if os.path.exists(parent_dir):
                print(f"\nContents of parent directory {parent_dir}:")
                try:
                    for item in sorted(os.listdir(parent_dir)):
                        print(f"  {item}")
                except Exception as e:
                    print(f"  Error listing directory: {e}")
        print(f"{'='*80}\n")
    
    assert onnx_file.endswith("onnx"), f"{onnx_file} has incorrect extension"
    engines_dir = tmp_path_factory.mktemp("depthnet_engines")
    engine_path = os.path.join(str(engines_dir), f"nvdepthanythingv2.{data_type}.engine")
    if not os.path.exists(engine_path):
        # Build with max_batch_size to cover test param up to 8
        builder = EngineBuilder(min_batch_size=1, opt_batch_size=4, max_batch_size=8, verbose=True)
        builder.create_network(onnx_file, "onnx")
        builder.create_engine(engine_path, data_type)
    return engine_path


@pytest.mark.parametrize("left_dir", [os.path.join(data_path, "depth_net/left/")])
@pytest.mark.parametrize("target_width", [924])
@pytest.mark.parametrize("target_height", [518])
def test_build_engine_mono(mono_engine, left_dir, target_height, target_width):
    assert os.path.exists(mono_engine), "Engine was not generated"


@pytest.mark.parametrize("left_dir", [os.path.join(data_path, "depth_net/left/")])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("target_width", [924])
@pytest.mark.parametrize("target_height", [518])
def test_infer_mono(mono_engine, left_dir, batch_size, target_height, target_width):
    trt_infer = DepthNetInferencer(mono_engine, batch_size=batch_size)
    c, h, w = trt_infer.input_tensors[0].shape
    assert h == target_height and w == target_width and c == 3

    left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    assert len(left_images) > 0, f"No images found in {left_dir}"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
        max_items = min(len(left_images), batch_size)
        for p in left_images[:max_items]:
            tf.write(f"{p}\n")
        data_file = tf.name

    loader = DepthNetDataLoader(
        data_sources=[{"dataset_name": "RelativeMonoDataset", "data_file": data_file}],
        shape=(batch_size, c, h, w),
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        preprocessor="DepthNet",
        evaluation=False,
    )

    for batch, img_paths, scales in loader.get_batch():
        pred_depths = trt_infer.infer(batch)
        assert pred_depths.shape == (batch_size, target_height, target_width)

        for scale, pred_depth, img_path in zip(scales, pred_depths, img_paths):
            new_h, new_w = h, w
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            pred_depth = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            assert pred_depth.shape == (orig_h, orig_w)

@pytest.mark.skip(reason="Stereo model not available - skipping stereo depth tests until model is provided")
@pytest.fixture(scope="session", params=["fp32", "fp16"])
def stereo_engine(tmp_path_factory, request):
    data_type = request.param
    onnx_file = os.path.join(model_path, "depth_net/deployable_foundationstereo_small_v1.0.onnx")
    
    # Check if model file exists
    if not os.path.exists(onnx_file):
        model_dir = os.path.dirname(onnx_file)
        print(f"\n{'='*80}")
        print(f"ERROR: Model file not found: {onnx_file}")
        print(f"{'='*80}")
        if os.path.exists(model_dir):
            print(f"\nContents of {model_dir}:")
            try:
                contents = os.listdir(model_dir)
                if contents:
                    for item in sorted(contents):
                        item_path = os.path.join(model_dir, item)
                        if os.path.isdir(item_path):
                            print(f"  [DIR]  {item}")
                        else:
                            size = os.path.getsize(item_path)
                            print(f"  [FILE] {item} ({size} bytes)")
                else:
                    print("  (directory is empty)")
            except Exception as e:
                print(f"  Error listing directory: {e}")
        else:
            print(f"\nDirectory does not exist: {model_dir}")
            parent_dir = os.path.dirname(model_dir)
            if os.path.exists(parent_dir):
                print(f"\nContents of parent directory {parent_dir}:")
                try:
                    for item in sorted(os.listdir(parent_dir)):
                        print(f"  {item}")
                except Exception as e:
                    print(f"  Error listing directory: {e}")
        print(f"{'='*80}\n")
    
    assert onnx_file.endswith("onnx"), f"{onnx_file} has incorrect extension"
    engines_dir = tmp_path_factory.mktemp("depthnet_engines")
    engine_path = os.path.join(str(engines_dir), f"foundationstereo.{data_type}.engine")
    if not os.path.exists(engine_path):
        builder = EngineBuilder(min_batch_size=1, opt_batch_size=2, max_batch_size=2, verbose=True)
        builder.create_network(onnx_file, "onnx")
        builder.create_engine(engine_path, data_type)
    return engine_path


@pytest.mark.skip(reason="Stereo model not available - skipping until deployable_foundationstereo_small_v1.0.onnx is provided")
@pytest.mark.parametrize("left_dir", [os.path.join(data_path, "depth_net/left/")])
@pytest.mark.parametrize("right_dir", [os.path.join(data_path, "depth_net/right/")])
@pytest.mark.parametrize("target_width", [736])
@pytest.mark.parametrize("target_height", [320])
def test_build_engine_stereo(stereo_engine, left_dir, right_dir, target_height, target_width):
    assert os.path.exists(stereo_engine), "Engine was not generated"


@pytest.mark.skip(reason="Stereo model not available - skipping until deployable_foundationstereo_small_v1.0.onnx is provided")
@pytest.mark.parametrize("left_dir", [os.path.join(data_path, "depth_net/left/")])
@pytest.mark.parametrize("right_dir", [os.path.join(data_path, "depth_net/right/")])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("target_width", [736])
@pytest.mark.parametrize("target_height", [320])
def test_infer_stereo(stereo_engine, left_dir, right_dir, batch_size, target_height, target_width):
    trt_infer = DepthNetInferencer(stereo_engine, batch_size=batch_size)
    c, h, w = trt_infer.input_tensors[0].shape
    assert h == target_height and w == target_width and c == 3

    left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
    assert len(left_images) > 0 and len(left_images) == len(right_images), "Stereo image pairs mismatch"
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
        max_items = min(len(left_images), batch_size)
        for l, r in zip(left_images[:max_items], right_images[:max_items]):
            tf.write(f"{l} {r}\n")
        data_file = tf.name

    loader = DepthNetDataLoader(
        data_sources=[{"dataset_name": "GenericDataset", "data_file": data_file}],
        shape=(batch_size, c, h, w),
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        preprocessor="DepthNet",
        evaluation=False,
    )

    for batch, img_paths, scales in loader.get_batch():
        pred_depths = trt_infer.infer(batch)
        assert pred_depths.shape == (batch_size, target_height, target_width)

        for scale, pred_depth, img_path in zip(scales, pred_depths, img_paths):
            new_h, new_w = h, w
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            pred_depth = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            assert pred_depth.shape == (orig_h, orig_w)
