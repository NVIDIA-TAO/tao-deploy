# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP TensorRT inferencer (combined engine)."""

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


class VideoCLIPInferencer(TRTInferencer):
    """Manages TensorRT objects for a combined Video-CLIP model.

    The combined engine has three inputs (image, input_ids, attention_mask) and
    outputs image_embedding, text_embedding, logit_scale, and optionally
    logit_bias. The image input is 5-D ``(B, T, C, H, W)`` for video; all shape
    handling reads ``input_tensors[0].tensor_shape[1:]`` so the extra temporal
    axis is transparent.
    """

    def __init__(self, engine_path, input_shape=None, batch_size=None,
                 data_format="channel_first"):
        """Initialize TensorRT objects for model inference.

        Args:
            engine_path (str): Path to TensorRT engine file.
            input_shape (tuple): (batch, T, C, H, W) for dynamic engines.
            batch_size (int): Batch size for dynamic engines.
            data_format (str): channel_first or channel_last.
        """
        super().__init__(
            engine_path,
            input_shape=input_shape,
            batch_size=batch_size,
            data_format=data_format,
        )
        self._output_names = [t.tensor_name for t in self.output_tensors]

    @property
    def image_input_shape(self):
        """Image input shape (T, C, H, W)."""
        return tuple(self.input_tensors[0].tensor_shape[1:])

    @property
    def image_input_dtype(self):
        """Numpy dtype for image input."""
        return trt.nptype(self.input_tensors[0].tensor_dtype)

    @property
    def num_frames(self):
        """Number of frames T from the engine's image input."""
        return int(self.input_tensors[0].tensor_shape[1])

    @property
    def context_length(self):
        """Text sequence length from engine input."""
        return int(self.input_tensors[1].tensor_shape[1])

    def _run(self, input_list):
        """Low-level engine execution with an ordered input list."""
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

    def infer(self, imgs, input_ids=None, attention_mask=None):  # pylint: disable=arguments-renamed
        """Run inference on a batch of preprocessed videos (and optional text).

        Args:
            imgs (np.ndarray): (B, T, C, H, W) preprocessed video batch.
            input_ids (np.ndarray | None): (B, seq_len) tokenised text. If None,
                zeros are used (text outputs are meaningless).
            attention_mask (np.ndarray | None): (B, seq_len). If None, ones.

        Returns:
            list[np.ndarray]: All engine outputs.
        """
        if input_ids is None:
            seq_len = self.input_tensors[1].tensor_shape[1]
            input_ids = np.zeros((imgs.shape[0], seq_len), dtype=np.int64)
        # attention_mask is inert: the exported ONNX overrides it with all-ones
        # internally (the InternVideo2 text tower is causal + EOT-pooled), so the
        # value passed here does not affect the output. Kept as a graph input for
        # interface compatibility with the combined engine's 3 inputs.
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        return self._run([imgs, input_ids, attention_mask])

    def get_image_embeddings(self, imgs):
        """Extract L2-normalised video embeddings.

        Args:
            imgs (np.ndarray): (B, T, C, H, W) preprocessed videos.

        Returns:
            np.ndarray: (B, D) L2-normalised video features.
        """
        outputs = self.infer(imgs)
        # The engine's output buffer is sized to max_batch_size; a partial final
        # batch returns padded (zero-input) rows. Keep only the real ones.
        feats = outputs[self._output_index("image_embedding")][:imgs.shape[0]]
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)

    def get_text_embeddings(self, input_ids):
        """Extract L2-normalised text embeddings.

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
        # Trim padded rows from a partial final batch (see get_image_embeddings).
        feats = outputs[self._output_index("text_embedding")][:input_ids.shape[0]]
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)


def create_video_clip_inferencer(trt_engine, batch_size=None,
                                 data_format="channel_first"):
    """Return a VideoCLIPInferencer for a combined engine file or directory.

    Args:
        trt_engine (str): Path to a combined ``.engine`` file, or a directory
            containing exactly one ``.engine`` file.
        batch_size (int | None): Batch size for dynamic engines.
        data_format (str): Input data format.

    Returns:
        VideoCLIPInferencer.

    Raises:
        FileNotFoundError: If no engine file is found.
    """
    kwargs = {"batch_size": batch_size, "data_format": data_format}
    if os.path.isdir(trt_engine):
        engines = sorted(glob.glob(os.path.join(trt_engine, "*.engine")))
        if not engines:
            raise FileNotFoundError(
                f"No .engine files found in directory: {trt_engine}"
            )
        logger.info("Combined engine found in dir: %s", engines[0])
        return VideoCLIPInferencer(engines[0], **kwargs)
    if os.path.isfile(trt_engine):
        return VideoCLIPInferencer(trt_engine, **kwargs)
    raise FileNotFoundError(f"TRT engine not found: {trt_engine}")
