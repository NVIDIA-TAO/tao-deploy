# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Inferencer Unit Tests."""

import numpy as np
import pytest
import tensorrt as trt
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from nvidia_tao_deploy.multimodal.clip.inferencer import (
    CLIPInferencer,
    CLIPSeparateInferencer,
    _SingleEncoderInferencer,
    create_clip_inferencer,
    trt_output_process_fn,
)
from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer


def _mock_output_tensors():
    """Create mock output tensors for CLIP engine."""
    return [
        SimpleNamespace(tensor_name="image_embedding"),
        SimpleNamespace(tensor_name="text_embedding"),
        SimpleNamespace(tensor_name="logit_scale"),
    ]


def _mock_input_tensors():
    """Create mock input tensors for CLIP engine."""
    return [
        SimpleNamespace(tensor_shape=(1, 3, 224, 224)),
        SimpleNamespace(tensor_shape=(1, 77)),
        SimpleNamespace(tensor_shape=(1, 77)),  # attention_mask
    ]


def _mock_output_tensors_with_bias():
    """Create mock output tensors for CLIP engine including logit_bias."""
    return [
        SimpleNamespace(tensor_name="image_embedding"),
        SimpleNamespace(tensor_name="text_embedding"),
        SimpleNamespace(tensor_name="logit_scale"),
        SimpleNamespace(tensor_name="logit_bias"),
    ]


def _mock_input_tensors_with_dtype():
    """Create mock input tensors with tensor_dtype for property tests."""
    return [
        SimpleNamespace(tensor_shape=(1, 3, 224, 224), tensor_dtype=trt.float32),
        SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
        SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
    ]


def _mock_vision_output_tensors():
    """Mock output tensors for a separate vision engine."""
    return [
        SimpleNamespace(tensor_name="image_embedding"),
        SimpleNamespace(tensor_name="logit_scale"),
        SimpleNamespace(tensor_name="logit_bias"),
    ]


def _mock_text_output_tensors():
    """Mock output tensors for a separate text engine."""
    return [
        SimpleNamespace(tensor_name="text_embedding"),
        SimpleNamespace(tensor_name="logit_scale"),
        SimpleNamespace(tensor_name="logit_bias"),
    ]


def _setup_mock_inferencer(instance):
    """Set up mock attributes on a CLIPInferencer instance after init."""
    instance.output_tensors = _mock_output_tensors()
    instance.input_tensors = _mock_input_tensors()
    instance._copy_input_to_host = MagicMock()
    instance.context = MagicMock()
    instance.bindings = []
    instance.inputs = []
    instance.outputs = []
    instance.stream = MagicMock()
    instance.max_batch_size = 1
    instance.execute_async = True
    instance.trt_runtime = None
    instance.engine = None


def _make_mock_encoder(output_tensors, input_tensors):
    """Create a mock _SingleEncoderInferencer with given IO tensors."""
    mock = MagicMock(spec=_SingleEncoderInferencer)
    mock.input_tensors = input_tensors
    mock.output_tensors = output_tensors
    names = [t.tensor_name for t in output_tensors]
    mock.output_index.side_effect = lambda name: names.index(name)
    return mock


class TestTrtOutputProcessFn:
    """Tests for trt_output_process_fn."""

    def test_reshape_1d(self):
        mock_output = SimpleNamespace(
            host=np.array([1.0, 2.0, 3.0, 4.0]),
            numpy_shape=(2, 2),
        )
        result = trt_output_process_fn(mock_output)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_reshape_batch_features(self):
        data = np.arange(24, dtype=np.float32)
        mock_output = SimpleNamespace(
            host=data,
            numpy_shape=(2, 3, 4),
        )
        result = trt_output_process_fn(mock_output)
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_equal(result.flatten(), data)

    def test_single_element(self):
        mock_output = SimpleNamespace(
            host=np.array([42.0]),
            numpy_shape=(1,),
        )
        result = trt_output_process_fn(mock_output)
        np.testing.assert_array_equal(result, np.array([42.0]))

    def test_preserves_dtype(self):
        mock_output = SimpleNamespace(
            host=np.array([1, 2, 3, 4], dtype=np.float16),
            numpy_shape=(2, 2),
        )
        result = trt_output_process_fn(mock_output)
        assert result.dtype == np.float16


class TestCLIPInferencerInit:
    """Tests for CLIPInferencer initialization."""

    def test_inherits_trt_inferencer(self):
        assert issubclass(CLIPInferencer, TRTInferencer)

    def test_default_params(self):
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None
            self.context = None
            self.engine = None
            self.stream = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/path/to/engine")
            assert infer._output_names == [
                "image_embedding", "text_embedding", "logit_scale"
            ]

    def test_custom_params(self):
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None
            self.context = None
            self.engine = None
            self.stream = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer(
                "/path/to/engine",
                input_shape=(1, 3, 224, 224),
                batch_size=8,
                data_format="channel_last",
            )
            assert infer._output_names == [
                "image_embedding", "text_embedding", "logit_scale"
            ]


class TestCLIPInferencerInfer:
    """Tests for CLIPInferencer.infer."""

    def test_infer_single_output(self):
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)

        mock_result = SimpleNamespace(
            host=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            numpy_shape=(1, 3),
        )

        with patch(
            'nvidia_tao_deploy.multimodal.clip.inferencer.do_inference',
            return_value=[mock_result],
        ):
            imgs = np.random.rand(1, 3, 224, 224).astype(np.float32)
            results = infer.infer(imgs)

        assert len(results) == 1
        assert results[0].shape == (1, 3)

    def test_infer_multiple_outputs(self):
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)
        infer.max_batch_size = 2

        mock_img_features = SimpleNamespace(
            host=np.arange(6, dtype=np.float32),
            numpy_shape=(2, 3),
        )
        mock_text_features = SimpleNamespace(
            host=np.arange(8, dtype=np.float32),
            numpy_shape=(2, 4),
        )

        with patch(
            'nvidia_tao_deploy.multimodal.clip.inferencer.do_inference',
            return_value=[mock_img_features, mock_text_features],
        ):
            imgs = np.random.rand(2, 3, 224, 224).astype(np.float32)
            results = infer.infer(imgs)

        assert len(results) == 2
        assert results[0].shape == (2, 3)
        assert results[1].shape == (2, 4)

    def test_infer_auto_creates_input_ids_and_attention_mask(self):
        """Test that infer creates zero input_ids and ones attention_mask when not provided."""
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)

        mock_result = SimpleNamespace(
            host=np.zeros(3, dtype=np.float32),
            numpy_shape=(1, 3),
        )

        with patch(
            'nvidia_tao_deploy.multimodal.clip.inferencer.do_inference',
            return_value=[mock_result],
        ):
            imgs = np.random.rand(1, 3, 224, 224).astype(np.float32)
            infer.infer(imgs)

        call_args = infer._copy_input_to_host.call_args[0][0]
        assert len(call_args) == 3  # image, input_ids, attention_mask
        assert call_args[1].shape == (1, 77)
        assert call_args[1].dtype == np.int64
        assert call_args[2].shape == (1, 77)  # attention_mask
        assert call_args[2].dtype == np.int64
        np.testing.assert_array_equal(call_args[2], np.ones((1, 77), dtype=np.int64))


class TestCLIPInferencerEmbeddings:
    """Tests for CLIPInferencer embedding extraction methods."""

    def test_get_image_embeddings_normalized(self):
        """Test that image embeddings are L2 normalized."""
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)

        raw_feats = np.array([[3.0, 4.0]], dtype=np.float32)
        mock_outputs = [
            raw_feats,
            np.zeros((1, 2)),
            np.array([1.0]),
        ]

        with patch.object(infer, '_run', return_value=mock_outputs):
            result = infer.get_image_embeddings(
                np.zeros((1, 3, 224, 224), dtype=np.float32)
            )

        norm = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_get_text_embeddings_normalized(self):
        """Test that text embeddings are L2 normalized."""
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)

        raw_feats = np.array([[5.0, 12.0]], dtype=np.float32)
        mock_outputs = [
            np.zeros((1, 2)),
            raw_feats,
            np.array([1.0]),
        ]

        with patch.object(infer, '_run', return_value=mock_outputs):
            result = infer.get_text_embeddings(
                np.zeros((1, 77), dtype=np.int64)
            )

        norm = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_get_logit_scale(self):
        """Test logit_scale extraction."""
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)

        expected_scale = 4.6052
        mock_outputs = [
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.array([[expected_scale]]),
        ]

        with patch.object(infer, '_run', return_value=mock_outputs):
            result = infer.get_logit_scale()

        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected_scale, rtol=1e-5)

    def test_get_logit_bias(self):
        """Test logit_bias extraction."""
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors_with_bias()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        _setup_mock_inferencer(infer)
        infer.output_tensors = _mock_output_tensors_with_bias()
        infer._output_names = [t.tensor_name for t in infer.output_tensors]

        expected_bias = -10.0
        mock_outputs = [
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.array([[4.6]]),
            np.array([[expected_bias]]),
        ]

        with patch.object(infer, '_run', return_value=mock_outputs):
            result = infer.get_logit_bias()

        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected_bias, rtol=1e-5)


class TestCLIPInferencerProperties:
    """Tests for CLIPInferencer uniform accessor properties."""

    def _make_infer(self):
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_output_tensors_with_bias()
            self.trt_runtime = None

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            infer = CLIPInferencer("/fake/engine")

        infer.input_tensors = _mock_input_tensors_with_dtype()
        infer.trt_runtime = None
        infer.context = None
        infer.engine = None
        infer.stream = None
        infer.inputs = []
        infer.outputs = []
        return infer

    def test_image_input_shape(self):
        infer = self._make_infer()
        assert infer.image_input_shape == (3, 224, 224)

    def test_image_input_dtype(self):
        infer = self._make_infer()
        assert infer.image_input_dtype == np.float32

    def test_context_length(self):
        infer = self._make_infer()
        assert infer.context_length == 77


# ======================================================================
# CLIPSeparateInferencer tests
# ======================================================================

class TestCLIPSeparateInferencer:
    """Tests for CLIPSeparateInferencer."""

    def _make_separate_infer(self):
        """Create a CLIPSeparateInferencer with mocked sub-engines."""
        infer = CLIPSeparateInferencer.__new__(CLIPSeparateInferencer)
        infer._vision = _make_mock_encoder(
            _mock_vision_output_tensors(),
            [SimpleNamespace(tensor_shape=(1, 3, 224, 224), tensor_dtype=trt.float32)],
        )
        infer._text = _make_mock_encoder(
            _mock_text_output_tensors(),
            [
                SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
                SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
            ],
        )
        return infer

    def test_image_input_shape(self):
        infer = self._make_separate_infer()
        assert infer.image_input_shape == (3, 224, 224)

    def test_image_input_dtype(self):
        infer = self._make_separate_infer()
        assert infer.image_input_dtype == np.float32

    def test_context_length(self):
        infer = self._make_separate_infer()
        assert infer.context_length == 77

    def test_get_image_embeddings_normalized(self):
        infer = self._make_separate_infer()
        raw = np.array([[3.0, 4.0]], dtype=np.float32)
        infer._vision.run.return_value = [
            raw,
            np.array([[4.6]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]

        result = infer.get_image_embeddings(
            np.zeros((1, 3, 224, 224), dtype=np.float32)
        )
        infer._vision.run.assert_called_once()
        norm = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_get_text_embeddings_normalized(self):
        infer = self._make_separate_infer()
        raw = np.array([[5.0, 12.0]], dtype=np.float32)
        infer._text.run.return_value = [
            raw,
            np.array([[4.6]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]

        result = infer.get_text_embeddings(
            np.zeros((1, 77), dtype=np.int64)
        )
        infer._text.run.assert_called_once()
        norm = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_get_logit_scale(self):
        infer = self._make_separate_infer()
        expected = 4.6052
        infer._vision.run.return_value = [
            np.zeros((1, 2), dtype=np.float32),
            np.array([[expected]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]
        result = infer.get_logit_scale()
        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_get_logit_bias(self):
        infer = self._make_separate_infer()
        expected = -10.0
        infer._vision.run.return_value = [
            np.zeros((1, 2), dtype=np.float32),
            np.array([[4.6]], dtype=np.float32),
            np.array([[expected]], dtype=np.float32),
        ]
        result = infer.get_logit_bias()
        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_image_only_does_not_touch_text_engine(self):
        """Vision-only inference should never call the text engine."""
        infer = self._make_separate_infer()
        infer._vision.run.return_value = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[4.6]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]
        infer.get_image_embeddings(np.zeros((1, 3, 224, 224), dtype=np.float32))
        infer._text.run.assert_not_called()

    def test_text_only_does_not_touch_vision_engine(self):
        """Text-only inference should never call the vision engine."""
        infer = self._make_separate_infer()
        infer._text.run.return_value = [
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[4.6]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]
        infer.get_text_embeddings(np.zeros((1, 77), dtype=np.int64))
        infer._vision.run.assert_not_called()

    # --- Single-pillar mode tests ---

    def test_vision_only_mode(self):
        """Vision-only inferencer can extract image embeddings."""
        infer = CLIPSeparateInferencer.__new__(CLIPSeparateInferencer)
        infer._vision = _make_mock_encoder(
            _mock_vision_output_tensors(),
            [SimpleNamespace(tensor_shape=(1, 3, 224, 224), tensor_dtype=trt.float32)],
        )
        infer._text = None

        assert infer.has_vision
        assert not infer.has_text
        assert infer.image_input_shape == (3, 224, 224)
        assert infer.context_length is None

        infer._vision.run.return_value = [
            np.array([[3.0, 4.0]], dtype=np.float32),
            np.array([[4.6]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]
        result = infer.get_image_embeddings(
            np.zeros((1, 3, 224, 224), dtype=np.float32)
        )
        assert result.shape == (1, 2)

    def test_vision_only_text_raises(self):
        """Vision-only inferencer raises on text embedding request."""
        infer = CLIPSeparateInferencer.__new__(CLIPSeparateInferencer)
        infer._vision = _make_mock_encoder(
            _mock_vision_output_tensors(),
            [SimpleNamespace(tensor_shape=(1, 3, 224, 224), tensor_dtype=trt.float32)],
        )
        infer._text = None

        with pytest.raises(RuntimeError, match="Text engine not loaded"):
            infer.get_text_embeddings(np.zeros((1, 77), dtype=np.int64))

    def test_text_only_mode(self):
        """Text-only inferencer can extract text embeddings."""
        infer = CLIPSeparateInferencer.__new__(CLIPSeparateInferencer)
        infer._vision = None
        infer._text = _make_mock_encoder(
            _mock_text_output_tensors(),
            [
                SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
                SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32),
            ],
        )

        assert not infer.has_vision
        assert infer.has_text
        assert infer.image_input_shape is None
        assert infer.context_length == 77

        infer._text.run.return_value = [
            np.array([[5.0, 12.0]], dtype=np.float32),
            np.array([[4.6]], dtype=np.float32),
            np.array([[-10.0]], dtype=np.float32),
        ]
        result = infer.get_text_embeddings(np.zeros((1, 77), dtype=np.int64))
        assert result.shape == (1, 2)

    def test_text_only_image_raises(self):
        """Text-only inferencer raises on image embedding request."""
        infer = CLIPSeparateInferencer.__new__(CLIPSeparateInferencer)
        infer._vision = None
        infer._text = _make_mock_encoder(
            _mock_text_output_tensors(),
            [SimpleNamespace(tensor_shape=(1, 77), tensor_dtype=trt.int32)],
        )

        with pytest.raises(RuntimeError, match="Vision engine not loaded"):
            infer.get_image_embeddings(
                np.zeros((1, 3, 224, 224), dtype=np.float32)
            )

    def test_both_none_raises(self):
        """Passing both paths as None raises ValueError."""
        with pytest.raises(ValueError, match="At least one engine"):
            CLIPSeparateInferencer(
                vision_engine_path=None, text_engine_path=None
            )


# ======================================================================
# _SingleEncoderInferencer tests
# ======================================================================

class TestSingleEncoderInferencer:
    """Tests for _SingleEncoderInferencer."""

    def test_inherits_trt_inferencer(self):
        assert issubclass(_SingleEncoderInferencer, TRTInferencer)

    def test_output_index(self):
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_vision_output_tensors()
            self.trt_runtime = None
            self.context = None
            self.engine = None
            self.stream = None
            self.inputs = []
            self.outputs = []

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            enc = _SingleEncoderInferencer("/fake/engine")

        assert enc.output_index("image_embedding") == 0
        assert enc.output_index("logit_scale") == 1
        assert enc.output_index("logit_bias") == 2

    def test_infer_raises(self):
        """The abstract infer() should raise — callers must use run()."""
        def mock_parent_init(self, *args, **kwargs):
            self.output_tensors = _mock_vision_output_tensors()
            self.trt_runtime = None
            self.context = None
            self.engine = None
            self.stream = None
            self.inputs = []
            self.outputs = []

        with patch.object(TRTInferencer, '__init__', mock_parent_init):
            enc = _SingleEncoderInferencer("/fake/engine")

        with pytest.raises(NotImplementedError):
            enc.infer(np.zeros((1, 3, 224, 224)))


# ======================================================================
# create_clip_inferencer factory tests
# ======================================================================

def _noop_trt_init(self, *args, **kwargs):
    """No-op TRTInferencer.__init__ that sets all required attributes."""
    self.output_tensors = []
    self.input_tensors = []
    self.trt_runtime = None
    self.context = None
    self.engine = None
    self.stream = None
    self.inputs = []
    self.outputs = []


def _combined_trt_init(self, *args, **kwargs):
    """Mock TRTInferencer.__init__ that sets combined-engine attributes."""
    self.output_tensors = _mock_output_tensors()
    self.input_tensors = _mock_input_tensors()
    self.trt_runtime = None
    self.context = None
    self.engine = None
    self.stream = None
    self.inputs = []
    self.outputs = []


class TestCreateClipInferencer:
    """Tests for the create_clip_inferencer factory function."""

    # --- Case 1: combined engine file ---

    def test_combined_engine(self, tmp_path):
        """Factory returns CLIPInferencer when a single engine file exists."""
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _combined_trt_init):
            result = create_clip_inferencer(str(engine))
        assert isinstance(result, CLIPInferencer)

    # --- Case 2: base path, _vision + _text discovered ---

    def test_separate_engines(self, tmp_path):
        """Factory returns CLIPSeparateInferencer when _vision/_text exist."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(
                str(tmp_path / "model.engine")
            )
        assert isinstance(result, CLIPSeparateInferencer)

    # --- Case 3: user points to _vision or _text directly ---

    def test_point_to_vision_finds_pair(self, tmp_path):
        """Pointing to _vision engine auto-discovers _text partner."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(
                str(tmp_path / "model_vision.engine")
            )
        assert isinstance(result, CLIPSeparateInferencer)

    def test_point_to_text_finds_pair(self, tmp_path):
        """Pointing to _text engine auto-discovers _vision partner."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(
                str(tmp_path / "model_text.engine")
            )
        assert isinstance(result, CLIPSeparateInferencer)

    def test_point_to_vision_only(self, tmp_path):
        """Pointing to _vision without _text gives vision-only inferencer."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(
                str(tmp_path / "model_vision.engine")
            )
        assert isinstance(result, CLIPSeparateInferencer)
        assert result.has_vision
        assert not result.has_text

    def test_point_to_text_only(self, tmp_path):
        """Pointing to _text without _vision gives text-only inferencer."""
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(
                str(tmp_path / "model_text.engine")
            )
        assert isinstance(result, CLIPSeparateInferencer)
        assert not result.has_vision
        assert result.has_text

    # --- Case 4: directory path ---

    def test_directory_with_separate_engines(self, tmp_path):
        """Pointing to directory discovers separate engine pair."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(str(tmp_path))
        assert isinstance(result, CLIPSeparateInferencer)

    def test_directory_with_combined_engine(self, tmp_path):
        """Pointing to directory discovers combined engine."""
        (tmp_path / "model.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _combined_trt_init):
            result = create_clip_inferencer(str(tmp_path))
        assert isinstance(result, CLIPInferencer)

    def test_directory_prefers_separate_over_combined(self, tmp_path):
        """Directory with both separate and combined prefers separate."""
        (tmp_path / "model.engine").write_bytes(b"fake")
        (tmp_path / "model_vision.engine").write_bytes(b"fake")
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(str(tmp_path))
        assert isinstance(result, CLIPSeparateInferencer)

    def test_directory_with_vision_only(self, tmp_path):
        """Directory with only _vision engine gives vision-only inferencer."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(str(tmp_path))
        assert isinstance(result, CLIPSeparateInferencer)
        assert result.has_vision
        assert not result.has_text

    def test_directory_with_text_only(self, tmp_path):
        """Directory with only _text engine gives text-only inferencer."""
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(str(tmp_path))
        assert isinstance(result, CLIPSeparateInferencer)
        assert not result.has_vision
        assert result.has_text

    def test_empty_directory_raises(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No .engine files"):
            create_clip_inferencer(str(tmp_path))

    # --- Case 7: combined preferred when explicitly pointed at ---

    def test_prefers_combined_when_both_exist(self, tmp_path):
        """If both combined and separate files exist, combined wins
        when user explicitly points to the combined file."""
        (tmp_path / "model.engine").write_bytes(b"fake")
        (tmp_path / "model_vision.engine").write_bytes(b"fake")
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _combined_trt_init):
            result = create_clip_inferencer(str(tmp_path / "model.engine"))
        assert isinstance(result, CLIPInferencer)

    # --- Error cases ---

    def test_no_engine_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="TRT engine not found"):
            create_clip_inferencer(str(tmp_path / "nonexistent.engine"))

    def test_error_message_lists_checked_paths(self, tmp_path):
        base = str(tmp_path / "model.engine")
        with pytest.raises(FileNotFoundError) as exc_info:
            create_clip_inferencer(base)
        msg = str(exc_info.value)
        assert "model.engine" in msg
        assert "model_vision.engine" in msg
        assert "model_text.engine" in msg

    def test_only_vision_engine_via_base_path(self, tmp_path):
        """Having only _vision via base path gives vision-only inferencer."""
        (tmp_path / "model_vision.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(str(tmp_path / "model.engine"))
        assert isinstance(result, CLIPSeparateInferencer)
        assert result.has_vision
        assert not result.has_text

    def test_only_text_engine_via_base_path(self, tmp_path):
        """Having only _text via base path gives text-only inferencer."""
        (tmp_path / "model_text.engine").write_bytes(b"fake")

        with patch.object(TRTInferencer, '__init__', _noop_trt_init):
            result = create_clip_inferencer(str(tmp_path / "model.engine"))
        assert isinstance(result, CLIPSeparateInferencer)
        assert not result.has_vision
        assert result.has_text
