# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multitask Classification Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.multitask_classification.engine_builder import MClassificationEngineBuilder
from nvidia_tao_deploy.cv.multitask_classification.inferencer import MClassificationInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 60
TARGET_HEIGHT = 80
CLASSMAP = {
    "tasks": ["base_color", "category", "season"],
    "class_mapping": {
        "base_color": {"0": "Black", "1": "Blue", "2": "Brown", "3": "Green", "4": "Grey", "5": "Navy Blue", "6": "Pink", "7": "Purple", "8": "Red", "9": "Silver", "10": "White"},
        "category": {"0": "Bags", "1": "Bottomwear", "2": "Eyewear", "3": "Fragrance", "4": "Innerwear", "5": "Jewellery", "6": "Sandal", "7": "Shoes", "8": "Topwear", "9": "Watches"},
        "season": {"0": "Fall", "1": "Spring", "2": "Summer", "3": "Winter"}
    }
}
OUTPUT_SHAPE = (1, 20)


@pytest.mark.skip(reason="UFF models no longer supported")
@pytest.mark.multitask_classification
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "multitask_classification/mcls_export.etlt")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("data_format", ["channel_first"])
@pytest.mark.parametrize("batch_size", [2])
def test_fp_engine(model_path, model_key, data_type, data_format, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/multitask_classification.{data_type}.engine"

    builder = MClassificationEngineBuilder(output_tasks=CLASSMAP['tasks'],
                                           min_batch_size=1,
                                           opt_batch_size=2,
                                           max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format=file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = MClassificationInferencer(output_engine_path, batch_size=batch_size)

    # Check engine model shape
    if data_format == "channel_first":
        c, h, w = trt_infer.input_tensors[0].shape
    else:
        h, w, c = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_images = []
    for _ in range(batch_size):
        dummy_image = np.random.randint(low=0, high=255, size=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    y_pred = trt_infer.infer(dummy_images)

    for i, (k, v) in enumerate(CLASSMAP['class_mapping'].items()):
        assert y_pred[i].shape[-1] == len(v), f"Task {k} has incorrect output dimensions"
