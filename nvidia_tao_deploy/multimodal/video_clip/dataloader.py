# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP dataloader for TensorRT retrieval evaluation.

The frame sampling (decord + uniform ``linspace`` over the chunk's time window),
the resolution logic, and the frame preprocessing (Resize 224 BICUBIC ->
ToTensor -> ImageNet Normalize) are ported 1:1 from the tao-pytorch trainer
(``dataloader/video_text_loader.py`` + ``model/adapters/internvideo2clip.py``)
so serving preprocessing is byte-identical to training. Do not "improve" the
resize / normalization / channel order here — that is a silent-accuracy contract.
"""

import json
import logging
import string
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_TRANS_PUNCTUATION = str.maketrans("", "", string.punctuation)


def canonicalize_text(text: str) -> str:
    """Lowercase + strip punctuation (big_vision/SigLIP canonicalization).

    Applied to query text before tokenization only when the exported model
    config has ``canonicalize_text: true`` — must match the training-time
    setting or the tokens (and thus the embeddings) diverge. Mirrors the image
    ``clip`` deploy module's ``canonicalize_text``.
    """
    text = text.replace("_", " ").translate(_TRANS_PUNCTUATION).lower()
    return " ".join(text.split()).strip()


# InternVideo2-CLIP L14 frame normalization (ImageNet stats, NOT CLIP stats).
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _resolve_video_path(video_path, data_root=None, path_prefix_mapping=None):
    """Resolve original dataset paths to local paths (verbatim from trainer)."""
    if video_path is None:
        return ""
    video_path = str(video_path)
    mapping = path_prefix_mapping or {}

    for prefix, replacement in mapping.items():
        if prefix and video_path.startswith(prefix):
            return video_path.replace(prefix, str(replacement), 1)

    path = Path(video_path)
    if path.is_absolute():
        if data_root and video_path.startswith("/media/wbf/"):
            return str(Path(data_root) / video_path[len("/media/wbf/"):])
        return str(path)
    if data_root:
        return str(Path(data_root) / path)
    return video_path


def linspace_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Select ``num_frames`` frame indices (verbatim from trainer).

    Uniform over ``[0, total_frames-1]`` when there are enough frames, else all
    frames padded by repeating the last one.
    """
    if total_frames <= 0:
        raise ValueError("Cannot sample from an empty video clip")
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    indices = np.full((num_frames,), total_frames - 1, dtype=int)
    indices[:total_frames] = np.arange(total_frames)
    return indices


def load_video_frames(video_path, num_frames, start_time_sec=None,
                      end_time_sec=None, start_frame=None, end_frame=None):
    """Load ``num_frames`` PIL RGB frames from a clip via decord (verbatim).

    Mirrors the trainer's decord path: resolve the frame window (by frame range
    if given, else by time * avg_fps), then uniform-sample ``num_frames`` inside
    that window.
    """
    import decord  # pylint: disable=import-outside-toplevel

    reader = decord.VideoReader(str(video_path))
    actual_frames = len(reader)
    start = 0
    end = actual_frames
    if start_frame is not None and end_frame is not None:
        start = max(0, min(int(start_frame), actual_frames - 1))
        end = min(int(end_frame), actual_frames)
    elif start_time_sec is not None and end_time_sec is not None:
        fps = reader.get_avg_fps()
        start = max(0, min(int(round(float(start_time_sec) * fps)), actual_frames - 1))
        end = min(int(round(float(end_time_sec) * fps)), actual_frames)
    if end <= start:
        raise ValueError(f"Invalid video range for {video_path}: [{start}, {end})")
    frame_indices = linspace_indices(end - start, num_frames) + start
    frames = reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def preprocess_frames(frames: List[Image.Image], image_size: int) -> np.ndarray:
    """Resize (BICUBIC) -> /255 -> ImageNet-normalize -> (T, C, H, W) float32.

    Matches ``InternVideo2FrameTransform``: torchvision Resize((s,s), BICUBIC) on
    a PIL image is PIL's own BICUBIC resize; ToTensor scales to [0,1] and moves
    channels first; Normalize subtracts/divides the ImageNet stats.
    """
    out = np.empty((len(frames), 3, image_size, image_size), dtype=np.float32)
    for i, img in enumerate(frames):
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3), RGB
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        out[i] = np.transpose(arr, (2, 0, 1))                    # (3, H, W)
    return out


def build_chunk_index(
    metadata_path: str,
    data_root: Optional[str] = None,
    path_prefix_mapping: Optional[dict] = None,
) -> Dict[str, dict]:
    """Map ``sample_id`` -> clip location from a vadr1_chunks metadata file.

    ``sample_id`` is ``f"{dataset}/{video_id}#{chunk_index}"`` — the same id the
    eval GT uses for gallery ``chunk_id`` / ``relevant_clip_ids``.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # vadr1_chunks is a list of video records, optionally wrapped in {"data": [...]}.
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        records = data["data"]
    else:
        raise ValueError(
            f"Unrecognized chunk-metadata format in {metadata_path}: expected a "
            f"list of video records (or {{'data': [...]}}), got {type(data).__name__}."
        )

    index: Dict[str, dict] = {}
    for record in records:
        video_path = _resolve_video_path(
            record.get("video_path"), data_root=data_root,
            path_prefix_mapping=path_prefix_mapping,
        )
        for chunk in record.get("chunks", []):
            chunk_index = chunk.get("chunk_index", 0)
            sample_id = f"{record.get('dataset', '')}/{record.get('video_id', '')}#{chunk_index}"
            index[sample_id] = {
                "video_path": video_path,
                "start_time_sec": chunk.get("start_time_sec"),
                "end_time_sec": chunk.get("end_time_sec"),
                "start_frame": chunk.get("start_frame"),
                "end_frame": chunk.get("end_frame"),
            }
    return index


class VideoCLIPGalleryLoader:
    """Batch loader for a fixed, ordered gallery of video chunks.

    Yields ``(videos, chunk_ids)`` where ``videos`` is ``(B, T, C, H, W)`` float32
    and ``chunk_ids`` preserves gallery order — the row-to-id alignment the
    evaluator relies on to map ``relevant_clip_ids`` to gallery indices.
    """

    def __init__(
        self,
        chunk_ids: List[str],
        chunk_index: Dict[str, dict],
        num_frames: int,
        image_size: int,
        batch_size: int = 8,
        dtype=np.float32,
    ):
        """Initialize the gallery loader.

        Args:
            chunk_ids: Ordered gallery chunk ids (the corpus).
            chunk_index: sample_id -> clip location (from build_chunk_index).
            num_frames: Frames per clip T (from the engine).
            image_size: Square input size H=W (from the engine).
            batch_size: Clips per batch.
            dtype: Output numpy dtype for the video tensor.
        """
        self.chunk_ids = list(chunk_ids)
        self.chunk_index = chunk_index
        self.num_frames = num_frames
        self.image_size = image_size
        self.batch_size = batch_size
        self.dtype = dtype

        missing = [c for c in self.chunk_ids if c not in chunk_index]
        if missing:
            raise KeyError(
                f"{len(missing)} gallery chunk_ids not found in the chunk "
                f"metadata (e.g. {missing[:3]}). Check dataset.val.metadata / "
                f"video_root."
            )
        self.n_samples = len(self.chunk_ids)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def _load_one(self, chunk_id: str) -> np.ndarray:
        loc = self.chunk_index[chunk_id]
        frames = load_video_frames(
            loc["video_path"], self.num_frames,
            start_time_sec=loc["start_time_sec"], end_time_sec=loc["end_time_sec"],
            start_frame=loc["start_frame"], end_frame=loc["end_frame"],
        )
        return preprocess_frames(frames, self.image_size).astype(self.dtype)

    def __len__(self) -> int:
        """Number of batches."""
        return self.n_batches

    def __iter__(self):
        """Iterate over (videos, chunk_ids) batches in gallery order."""
        for start in range(0, self.n_samples, self.batch_size):
            batch_ids = self.chunk_ids[start:start + self.batch_size]
            videos = np.stack([self._load_one(c) for c in batch_ids], axis=0)
            yield videos, batch_ids
