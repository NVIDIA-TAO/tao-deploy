# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP Inferencer Unit Tests (CPU-only, mocked TRT)."""

import numpy as np
import pytest
import tensorrt as trt
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvidia_tao_deploy.multimodal.video_clip.inferencer import (
    VideoCLIPInferencer,
    create_video_clip_inferencer,
    trt_output_process_fn,
)
from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer


def _mock_output_tensors():
    return [
        SimpleNamespace(tensor_name="image_embedding"),
        SimpleNamespace(tensor_name="text_embedding"),
        SimpleNamespace(tensor_name="logit_scale"),
        SimpleNamespace(tensor_name="logit_bias"),
    ]


def _mock_input_tensors():
    # 5-D video image input (B, T=8, C=3, H=224, W=224) + (B, 77) text inputs
    return [
        SimpleNamespace(tensor_shape=(1, 8, 3, 224, 224), tensor_dtype=trt.float32),
        SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
        SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
    ]


def _make_infer():
    def mock_parent_init(self, *args, **kwargs):
        self.output_tensors = _mock_output_tensors()
        self.trt_runtime = None
        self.context = None
        self.engine = None
        self.stream = None
        self.inputs = []
        self.outputs = []

    with patch.object(TRTInferencer, '__init__', mock_parent_init):
        infer = VideoCLIPInferencer("/fake/engine")
    infer.input_tensors = _mock_input_tensors()
    infer._copy_input_to_host = MagicMock()
    infer.context = MagicMock()
    infer.bindings, infer.inputs, infer.outputs = [], [], []
    infer.stream = MagicMock()
    infer.max_batch_size = 1
    infer.execute_async = True
    return infer


class TestProperties:
    def test_inherits_trt_inferencer(self):
        assert issubclass(VideoCLIPInferencer, TRTInferencer)

    def test_image_input_shape_is_5d(self):
        infer = _make_infer()
        assert infer.image_input_shape == (8, 3, 224, 224)

    def test_num_frames(self):
        assert _make_infer().num_frames == 8

    def test_context_length(self):
        assert _make_infer().context_length == 77

    def test_image_input_dtype(self):
        assert _make_infer().image_input_dtype == np.float32


class TestEmbeddings:
    def test_get_image_embeddings_normalized(self):
        infer = _make_infer()
        raw = np.array([[3.0, 4.0]], dtype=np.float32)  # norm 5 -> normalizes to 1
        outputs = [raw, np.zeros((1, 2)), np.array([1.0]), np.array([0.0])]
        with patch.object(infer, '_run', return_value=outputs):
            result = infer.get_image_embeddings(
                np.zeros((1, 8, 3, 224, 224), dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0, rtol=1e-5)

    def test_get_text_embeddings_builds_5d_dummy(self):
        infer = _make_infer()
        raw = np.array([[5.0, 12.0]], dtype=np.float32)  # norm 13
        outputs = [np.zeros((1, 2)), raw, np.array([1.0]), np.array([0.0])]
        with patch.object(infer, '_run', return_value=outputs) as mock_run:
            result = infer.get_text_embeddings(np.zeros((1, 77), dtype=np.int64))
            # _run was called with a 5-D dummy image matching the engine input
            called_imgs = mock_run.call_args[0][0][0]
            assert called_imgs.shape == (1, 8, 3, 224, 224)
        np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0, rtol=1e-5)

    def test_partial_batch_output_is_trimmed(self):
        # Engine returns max_batch_size rows; a 3-image input must yield 3 rows,
        # dropping the padded (zero-input) rows. Mock a 5-row padded output.
        infer = _make_infer()
        padded = np.random.rand(5, 2).astype(np.float32)
        outputs = [padded, padded, np.array([1.0]), np.array([0.0])]
        with patch.object(infer, '_run', return_value=outputs):
            imgs = np.zeros((3, 8, 3, 224, 224), dtype=np.float32)
            img_res = infer.get_image_embeddings(imgs)
            txt_res = infer.get_text_embeddings(np.zeros((3, 77), dtype=np.int64))
        assert img_res.shape == (3, 2)
        assert txt_res.shape == (3, 2)

    def test_infer_auto_fills_ids_and_mask(self):
        infer = _make_infer()
        outputs = [np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0]), np.array([0.0])]
        with patch(
            'nvidia_tao_deploy.multimodal.video_clip.inferencer.do_inference',
            return_value=[SimpleNamespace(host=np.zeros(2), numpy_shape=(1, 2))] * 4,
        ):
            infer.infer(np.zeros((1, 8, 3, 224, 224), dtype=np.float32))
        call_args = infer._copy_input_to_host.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[1].shape == (1, 77) and call_args[1].dtype == np.int64
        assert call_args[2].shape == (1, 77)
        np.testing.assert_array_equal(call_args[2], np.ones((1, 77), dtype=np.int64))


class TestFactory:
    def test_combined_engine_file(self, tmp_path):
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"fake")

        def mock_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.input_tensors = _mock_input_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_init):
            result = create_video_clip_inferencer(str(engine))
        assert isinstance(result, VideoCLIPInferencer)

    def test_missing_engine_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_video_clip_inferencer(str(tmp_path / "nope.engine"))


def test_trt_output_process_fn_reshape():
    out = SimpleNamespace(host=np.arange(6, dtype=np.float32), numpy_shape=(2, 3))
    assert trt_output_process_fn(out).shape == (2, 3)
