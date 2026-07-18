# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO Engine Building and Inferencer Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from nvidia_tao_deploy.cv.grounding_dino.engine_builder import GDINODetEngineBuilder
from nvidia_tao_deploy.cv.grounding_dino.inferencer import GDINOInferencer
from nvidia_tao_deploy.cv.grounding_dino.utils import post_process, tokenize_captions


model_dir = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_dir = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"
TARGET_WIDTH = 960
TARGET_HEIGHT = 544
MAX_TEXT_LEN = 256


@pytest.mark.deformable_detr
@pytest.mark.parametrize("model_path", [os.path.join(model_dir, "grounding_dino/swint.onnx")])
@pytest.mark.parametrize("img_path", [os.path.join(data_dir, "L0/temp_astro17.jpg")])
@pytest.mark.parametrize("data_type", ["fp32"])
@pytest.mark.parametrize("batch_size", [1])
def test_fp_engine(model_path, img_path, data_type, batch_size):
    # Engine building part

    output_engine_path = f"/tmp/grounding_dino.{data_type}.engine"

    # @seanf note: g-dino only supports a batch size of 1
    builder = GDINODetEngineBuilder(min_batch_size=1,
                                    opt_batch_size=1,
                                    max_batch_size=1)
    builder.create_network(model_path, "onnx")
    builder.create_engine(output_engine_path, data_type)

    assert os.path.exists(output_engine_path), "Engine was not generated"

    # Inferencer part
    trt_infer = GDINOInferencer(output_engine_path,
                                num_classes=MAX_TEXT_LEN,
                                batch_size=batch_size)

    # Check engine model shape
    c, h, w = trt_infer.input_tensors[0].shape

    assert h == TARGET_HEIGHT, "Incorrect height size"
    assert w == TARGET_WIDTH, "Incorrect width size"
    assert c == 3, "Incorrect channel size"

    # Load dummy image
    dummy_image = Image.open(img_path).resize((TARGET_WIDTH, TARGET_HEIGHT))
    dummy_image = np.asarray(dummy_image, np.float32)

    # Add batch dim
    dummy_image = np.expand_dims(dummy_image, 0)
    dummy_image = np.transpose(dummy_image, axes=(0, 3, 1, 2))

    cat_list = ["person", "chair"]
    caption = [" . ".join(cat_list) + ' .'] * batch_size

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(tokenizer, cat_list, caption, MAX_TEXT_LEN)
    inputs = (dummy_image, input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks)

    pred_logits, pred_boxes = trt_infer.infer(inputs)
    assert pred_logits.shape[1:] == (900, MAX_TEXT_LEN), "Logit dimension is incorrect"
    assert pred_boxes.shape[1:] == (900, 4), "Bounding box dimension is incorrect"

    # Post-processing
    target_sizes = np.array([[TARGET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT]])
    class_labels, scores, boxes = post_process(pred_logits, pred_boxes, target_sizes, pos_map)

    assert np.min(scores) >= 0, "Scores range is incorrect"
    assert np.min(class_labels) >= 0 and np.max(class_labels) < MAX_TEXT_LEN, "Class label range is incorrect"
