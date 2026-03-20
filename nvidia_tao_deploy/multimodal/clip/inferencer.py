# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLIP TensorRT inferencer."""

import glob
import logging
import os

import numpy as np
import tensorrt as trt

from nvidia_tao_deploy.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_deploy.inferencer.utils import do_inference

logger = logging.getLogger(__name__)


def trt_output_process_fn(y_encoded):
    """Process TRT model output to numpy array."""
    return np.reshape(y_encoded.host, y_encoded.numpy_shape)


class CLIPInferencer(TRTInferencer):
    """Manages TensorRT objects for combined CLIP model inference.

    Combined engines have three inputs (image, input_ids, attention_mask)
    and outputs including image_embedding, text_embedding, logit_scale,
    and optionally logit_bias.
    """

    def __init__(self, engine_path, input_shape=None, batch_size=None,
                 data_format="channel_first"):
        """Initialize TensorRT objects for model inference.

        Args:
            engine_path (str): Path to TensorRT engine file.
            input_shape (tuple): (batch, channel, height, width) for dynamic engines.
            batch_size (int): Batch size for dynamic engines.
            data_format (str): channel_first or channel_last.
        """
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
        )
        self._output_names = [
            t.tensor_name for t in self.output_tensors
        ]

    @property
    def image_input_shape(self):
        """Image input shape (C, H, W)."""
        return tuple(self.input_tensors[0].tensor_shape[1:])

    @property
    def image_input_dtype(self):
        """Numpy dtype for image input."""
        return trt.nptype(self.input_tensors[0].tensor_dtype)

    @property
    def context_length(self):
        """Text sequence length from engine input."""
        return self.input_tensors[1].tensor_shape[1]

    def _run(self, input_list):
        """Low-level engine execution with arbitrary input list.

        Args:
            input_list (list[np.ndarray]): One array per engine input tensor,
                in the order they appear in self.numpy_array.

        Returns:
            list[np.ndarray]: One array per engine output tensor.
        """
        self._copy_input_to_host(input_list)
        results = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True,
        )
        return [trt_output_process_fn(r) for r in results]

    def _output_index(self, name):
        return self._output_names.index(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, imgs, scales=None, attention_mask=None):  # pylint: disable=arguments-renamed
        """Run inference on a batch of preprocessed images (and optional text).

        Args:
            imgs (np.ndarray): (B, C, H, W) preprocessed image batch.
            scales (np.ndarray | None): For CLIP, this is input_ids (B, seq_len)
                tokenised text. If None, zeros are used (text outputs will be
                meaningless). Named 'scales' to match parent class signature.
            attention_mask (np.ndarray | None): (B, seq_len) attention mask.
                If None, ones are used (attend to all tokens).

        Returns:
            list[np.ndarray]: All engine outputs.
        """
        input_ids = scales
        if input_ids is None:
            seq_len = self.input_tensors[1].tensor_shape[1]
            input_ids = np.zeros(
                (imgs.shape[0], seq_len), dtype=np.int64
            )

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

        return self._run([imgs, input_ids, attention_mask])

    def get_image_embeddings(self, imgs):
        """Extract normalised image embeddings only.

        Args:
            imgs (np.ndarray): (B, C, H, W) preprocessed images.

        Returns:
            np.ndarray: (B, D) L2-normalised image features.
        """
        outputs = self.infer(imgs)
        idx = self._output_index("image_embedding")
        feats = outputs[idx]
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
        return feats

    def get_text_embeddings(self, input_ids):
        """Extract normalised text embeddings only.

        Args:
            input_ids (np.ndarray): (B, seq_len) tokenised text.

        Returns:
            np.ndarray: (B, D) L2-normalised text features.
        """
        dummy_imgs = np.zeros(
            (input_ids.shape[0], *self.input_tensors[0].tensor_shape[1:]),
            dtype=np.float32,
        )
        outputs = self.infer(dummy_imgs, input_ids)
        idx = self._output_index("text_embedding")
        feats = outputs[idx]
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
        return feats

    def get_logit_scale(self):
        """Return the learned logit_scale scalar from a dummy forward pass."""
        dummy_imgs = np.zeros(
            (1, *self.input_tensors[0].tensor_shape[1:]),
            dtype=np.float32,
        )
        outputs = self.infer(dummy_imgs)
        idx = self._output_index("logit_scale")
        return float(outputs[idx].item())

    def get_logit_bias(self):
        """Return the learned logit_bias scalar from a dummy forward pass."""
        dummy_imgs = np.zeros(
            (1, *self.input_tensors[0].tensor_shape[1:]),
            dtype=np.float32,
        )
        outputs = self.infer(dummy_imgs)
        idx = self._output_index("logit_bias")
        return float(outputs[idx].item())


# ======================================================================
# Separate-encoder support
# ======================================================================

class _SingleEncoderInferencer(TRTInferencer):
    """Concrete TRTInferencer for a single CLIP encoder (vision or text)."""

    def __init__(self, engine_path, batch_size=None,
                 data_format="channel_first"):
        super().__init__(
            engine_path,
            batch_size=batch_size,
            data_format=data_format,
        )
        self._output_names = [
            t.tensor_name for t in self.output_tensors
        ]

    def infer(self, imgs, scales=None):
        """Not used directly — call run() instead."""
        raise NotImplementedError("Use run() for single-encoder engines")

    def run(self, input_list):
        """Run inference with an arbitrary list of input arrays."""
        self._copy_input_to_host(input_list)
        results = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
            batch_size=self.max_batch_size,
            execute_v2=self.execute_async,
            return_raw=True,
        )
        return [trt_output_process_fn(r) for r in results]

    def output_index(self, name):
        """Return output tensor index by name."""
        return self._output_names.index(name)


class CLIPSeparateInferencer:
    """CLIP inferencer using separate vision and text TRT engines.

    Either engine may be ``None`` for single-pillar inference (e.g.
    image-only or text-only). Methods that require the absent engine
    raise ``RuntimeError`` with a clear message.

    Vision engine inputs: image
    Vision engine outputs: image_embedding, logit_scale, logit_bias

    Text engine inputs: input_ids, attention_mask
    Text engine outputs: text_embedding, logit_scale, logit_bias
    """

    def __init__(self, vision_engine_path=None, text_engine_path=None,
                 batch_size=None, data_format="channel_first"):
        """Initialize with one or both separate engines.

        Args:
            vision_engine_path (str | None): Path to vision TRT engine.
            text_engine_path (str | None): Path to text TRT engine.
            batch_size (int): Batch size for dynamic engines.
            data_format (str): channel_first or channel_last.

        Raises:
            ValueError: If both paths are None.
        """
        if vision_engine_path is None and text_engine_path is None:
            raise ValueError(
                "At least one engine path must be provided "
                "(vision_engine_path or text_engine_path)."
            )

        if vision_engine_path is not None:
            logger.info("Loading vision engine: %s", vision_engine_path)
            self._vision = _SingleEncoderInferencer(
                vision_engine_path, batch_size=batch_size,
                data_format=data_format,
            )
        else:
            logger.info("No vision engine — text-only mode")
            self._vision = None

        if text_engine_path is not None:
            logger.info("Loading text engine: %s", text_engine_path)
            self._text = _SingleEncoderInferencer(
                text_engine_path, batch_size=batch_size,
                data_format=data_format,
            )
        else:
            logger.info("No text engine — vision-only mode")
            self._text = None

    def _require_vision(self):
        """Raise if vision engine is not loaded."""
        if self._vision is None:
            raise RuntimeError(
                "Vision engine not loaded. Provide a vision engine to "
                "extract image embeddings or logit_scale/logit_bias."
            )

    def _require_text(self):
        """Raise if text engine is not loaded."""
        if self._text is None:
            raise RuntimeError(
                "Text engine not loaded. Provide a text engine to "
                "extract text embeddings."
            )

    @property
    def has_vision(self):
        """Whether a vision engine is loaded."""
        return self._vision is not None

    @property
    def has_text(self):
        """Whether a text engine is loaded."""
        return self._text is not None

    @property
    def image_input_shape(self):
        """Image input shape (C, H, W) from vision engine, or None."""
        if self._vision is None:
            return None
        return tuple(self._vision.input_tensors[0].tensor_shape[1:])

    @property
    def image_input_dtype(self):
        """Numpy dtype for image input, or None."""
        if self._vision is None:
            return None
        return trt.nptype(self._vision.input_tensors[0].tensor_dtype)

    @property
    def context_length(self):
        """Text sequence length from text engine, or None."""
        if self._text is None:
            return None
        return self._text.input_tensors[0].tensor_shape[1]

    def get_image_embeddings(self, imgs):
        """Extract L2-normalised image embeddings via the vision engine.

        Args:
            imgs (np.ndarray): (B, C, H, W) preprocessed images.

        Returns:
            np.ndarray: (B, D) L2-normalised image features.
        """
        self._require_vision()
        outputs = self._vision.run([imgs])
        idx = self._vision.output_index("image_embedding")
        feats = outputs[idx]
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
        return feats

    def get_text_embeddings(self, input_ids):
        """Extract L2-normalised text embeddings via the text engine.

        Args:
            input_ids (np.ndarray): (B, seq_len) tokenised text.

        Returns:
            np.ndarray: (B, D) L2-normalised text features.
        """
        self._require_text()
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        outputs = self._text.run([input_ids, attention_mask])
        idx = self._text.output_index("text_embedding")
        feats = outputs[idx]
        feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
        return feats

    def get_logit_scale(self):
        """Return the learned logit_scale scalar from the vision engine."""
        self._require_vision()
        dummy = np.zeros(
            (1, *self._vision.input_tensors[0].tensor_shape[1:]),
            dtype=np.float32,
        )
        outputs = self._vision.run([dummy])
        idx = self._vision.output_index("logit_scale")
        return float(outputs[idx].item())

    def get_logit_bias(self):
        """Return the learned logit_bias scalar from the vision engine."""
        self._require_vision()
        dummy = np.zeros(
            (1, *self._vision.input_tensors[0].tensor_shape[1:]),
            dtype=np.float32,
        )
        outputs = self._vision.run([dummy])
        idx = self._vision.output_index("logit_bias")
        return float(outputs[idx].item())


def _find_separate_pair(base, ext):
    """Return (vision_path, text_path) if both exist, else (None, None)."""
    vision_path = f"{base}_vision{ext}"
    text_path = f"{base}_text{ext}"
    if os.path.isfile(vision_path) and os.path.isfile(text_path):
        return vision_path, text_path
    return None, None


def _find_engines_in_dir(directory):
    """Scan a directory for engine files and return the best match.

    Returns (mode, path_or_pair) where mode is 'combined', 'separate',
    'vision_only', 'text_only', or None.
    """
    engines = sorted(glob.glob(os.path.join(directory, "*.engine")))
    if not engines:
        return None, None

    vision = [e for e in engines if e.endswith("_vision.engine")]
    text = [e for e in engines if e.endswith("_text.engine")]

    if vision and text:
        return "separate", (vision[0], text[0])

    non_pair = [
        e for e in engines
        if not e.endswith("_vision.engine")
        and not e.endswith("_text.engine")
    ]
    if non_pair:
        return "combined", non_pair[0]

    if vision:
        return "vision_only", vision[0]
    if text:
        return "text_only", text[0]

    return None, None


def _build_from_dir(directory, kwargs):
    """Build inferencer from a directory by scanning for engine files."""
    mode, result = _find_engines_in_dir(directory)
    if mode == "separate":
        logger.info("Separate engines found in dir: %s, %s",
                     result[0], result[1])
        return CLIPSeparateInferencer(result[0], result[1], **kwargs)
    if mode == "combined":
        logger.info("Combined engine found in dir: %s", result)
        return CLIPInferencer(result, **kwargs)
    if mode == "vision_only":
        logger.info("Vision-only engine found in dir: %s", result)
        return CLIPSeparateInferencer(
            vision_engine_path=result, **kwargs
        )
    if mode == "text_only":
        logger.info("Text-only engine found in dir: %s", result)
        return CLIPSeparateInferencer(
            text_engine_path=result, **kwargs
        )
    raise FileNotFoundError(
        f"No .engine files found in directory: {directory}"
    )


def _build_from_file(trt_engine, base, ext, kwargs):
    """Build inferencer when trt_engine points to an existing file."""
    for suffix in ("_vision", "_text"):
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            vp, tp = _find_separate_pair(stem, ext)
            if vp and tp:
                logger.info("Separate engines found: %s, %s", vp, tp)
                return CLIPSeparateInferencer(vp, tp, **kwargs)
            if suffix == "_vision":
                logger.info("Vision-only engine: %s", trt_engine)
                return CLIPSeparateInferencer(
                    vision_engine_path=trt_engine, **kwargs
                )
            logger.info("Text-only engine: %s", trt_engine)
            return CLIPSeparateInferencer(
                text_engine_path=trt_engine, **kwargs
            )

    logger.info("Combined engine found: %s", trt_engine)
    return CLIPInferencer(trt_engine, **kwargs)


def _build_from_base(base, ext, kwargs):
    """Build inferencer from a base path by discovering variants."""
    vp, tp = _find_separate_pair(base, ext)
    if vp and tp:
        logger.info("Separate engines found: %s, %s", vp, tp)
        return CLIPSeparateInferencer(vp, tp, **kwargs)

    vision_only = f"{base}_vision{ext}"
    text_only = f"{base}_text{ext}"
    if os.path.isfile(vision_only):
        logger.info("Vision-only engine (via base path): %s", vision_only)
        return CLIPSeparateInferencer(
            vision_engine_path=vision_only, **kwargs
        )
    if os.path.isfile(text_only):
        logger.info("Text-only engine (via base path): %s", text_only)
        return CLIPSeparateInferencer(
            text_engine_path=text_only, **kwargs
        )
    return None


def create_clip_inferencer(trt_engine, batch_size=None,
                           data_format="channel_first"):
    """Factory that returns the right CLIP inferencer based on engine files.

    Handles all common ways a user might specify the engine path:

    1. Path to a combined engine file (``model.engine``)
    2. Base path whose ``_vision`` / ``_text`` variants exist
       (``model.engine`` → looks for ``model_vision.engine`` +
       ``model_text.engine``)
    3. Direct path to a ``_vision`` or ``_text`` engine — automatically
       finds its partner (or runs single-pillar if partner is absent)
    4. Path to a directory — scans for engine files and auto-detects
       combined vs. separate

    Args:
        trt_engine (str): Path to engine file, base path, or directory.
        batch_size (int | None): Batch size for dynamic engines.
        data_format (str): Input data format.

    Returns:
        CLIPInferencer or CLIPSeparateInferencer.

    Raises:
        FileNotFoundError: If no matching engine files are found.
    """
    kwargs = {"batch_size": batch_size, "data_format": data_format}

    if os.path.isdir(trt_engine):
        return _build_from_dir(trt_engine, kwargs)

    base, ext = os.path.splitext(trt_engine)

    if os.path.isfile(trt_engine):
        return _build_from_file(trt_engine, base, ext, kwargs)

    result = _build_from_base(base, ext, kwargs)
    if result is not None:
        return result

    raise FileNotFoundError(
        f"TRT engine not found. Checked:\n"
        f"  Combined: {trt_engine}\n"
        f"  Separate: {base}_vision{ext} + {base}_text{ext}\n"
        f"  Directory: {os.path.dirname(trt_engine) or '.'}\n"
        f"Accepted formats:\n"
        f"  - Path to combined engine file\n"
        f"  - Base path (auto-discovers _vision/_text variants)\n"
        f"  - Path to _vision or _text engine (finds partner)\n"
        f"  - Path to directory containing engine file(s)"
    )
