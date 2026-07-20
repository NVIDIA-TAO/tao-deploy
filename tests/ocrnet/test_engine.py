# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image

from nvidia_tao_deploy.utils.decoding import decode_etlt
from nvidia_tao_deploy.cv.ocrnet.engine_builder import OCRNetEngineBuilder
from nvidia_tao_deploy.cv.ocrnet.inferencer import OCRNetInferencer


model_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/ocrnet"
data_path = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/L0/img/"
TARGET_WIDTH = 100
TARGET_HEIGHT = 32
TARGET_SEQ = 26
ATTN_TARGET_WIDTH = 200
ATTN_TARGET_HEIGHT = 64


@pytest.mark.ocrnet
@pytest.mark.parametrize("model_path", [os.path.join(model_path, "best_accuracy.etlt")])
@pytest.mark.parametrize("img_path", [os.path.join(data_path, "word_888.png")])
@pytest.mark.parametrize("model_key", ["nvidia_tao"])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fp_engine(model_path, img_path, model_key, data_type, batch_size):
    # Engine building part
    if model_path.endswith("etlt"):
        tmp_onnx_file, file_format = decode_etlt(model_path, model_key)
    else:
        raise ValueError(f"{model_path} has incorrect extension")

    output_engine_path = f"/tmp/ocrnet.{data_type}.engine"

    builder = OCRNetEngineBuilder(min_batch_size=1,
                                  opt_batch_size=2,
                                  max_batch_size=4)
    builder.create_network(tmp_onnx_file, file_format)
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = OCRNetInferencer(engine_path=output_engine_path,
                                 batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 1, "Incorrect channel size"

    # Load dummy image
    dummy_images = []

    for _ in range(batch_size):
        dummy_image = Image.open(img_path).convert("L").resize((TARGET_WIDTH, TARGET_HEIGHT))
        dummy_image = np.asarray(dummy_image, np.float32)[np.newaxis, :, :]
        dummy_images.append(dummy_image)

    # Add batch dim
    dummy_images = np.stack(dummy_images, axis=0)

    output_ids, output_probs = trt_infer.infer(dummy_images)

    assert output_ids.shape == (batch_size, TARGET_SEQ)
    assert output_probs.shape == (batch_size, TARGET_SEQ)
