# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP dataloaders for TensorRT evaluation."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from nvidia_tao_deploy.cv.common.constants import VALID_IMAGE_EXTENSIONS

# Normalization constants for different backbones
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def _resize_shortest_edge(img: Image.Image, target_size: int,
                          resampling) -> Image.Image:
    """Resize so the shortest edge equals target_size (preserve aspect ratio)."""
    w, h = img.size
    if w == h == target_size:
        return img
    if h < w:
        new_h = target_size
        new_w = int(round(w * target_size / h))
    else:
        new_w = target_size
        new_h = int(round(h * target_size / w))
    return img.resize((new_w, new_h), resampling)


def _center_crop(img: Image.Image, crop_w: int, crop_h: int) -> Image.Image:
    """Center crop to (crop_w, crop_h)."""
    w, h = img.size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def get_preprocessing_params(model_type: str, adaptor_name: Optional[str] = None):
    """Get preprocessing parameters based on model type.

    Args:
        model_type: Model type string (e.g., 'siglip2-so400m-patch16-256', 'c-radio_v3-l').
        adaptor_name: For RADIO models, the adaptor name (e.g., 'clip', 'siglip2').

    Returns:
        Tuple of (mean, std, resampling_method, center_crop).
        - mean, std: Normalization constants
        - resampling_method: PIL resampling method
        - center_crop: If True, resize shortest edge + center crop (OpenCLIP).
                       If False, direct resize to target (SigLIP2 HF processor).
    """
    model_lower = model_type.lower()

    # SigLIP2: direct resize, BILINEAR, (0.5, 0.5, 0.5) — matches HF processor
    if model_lower.startswith('siglip'):
        return SIGLIP_MEAN, SIGLIP_STD, Image.Resampling.BILINEAR, False

    # RADIO: OpenAI CLIP normalization regardless of adaptor, with center crop
    if 'radio' in model_lower:
        return OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, Image.Resampling.BICUBIC, True

    # OpenCLIP / NV-CLIP: OpenAI CLIP normalization with center crop
    return OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, Image.Resampling.BICUBIC, True


class CLIPRetrievalLoader:
    """Dataloader for retrieval evaluation with TRT.

    Loads image-caption pairs for computing retrieval metrics.
    Returns preprocessed images and raw caption strings.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        image_dir: str,
        model_type: str,
        adaptor_name: Optional[str] = None,
        caption_dir: Optional[str] = None,
        caption_file_suffix: str = ".txt",
        image_list_file: Optional[str] = None,
        batch_size: int = 32,
        dtype=None,
    ):
        """Initialize the retrieval dataloader.

        Args:
            shape: (C, H, W) input shape from engine.
            image_dir: Directory containing images.
            model_type: Model type string from exported config. Required for
                determining correct preprocessing (normalization, resampling).
            adaptor_name: For RADIO models, the adaptor name (e.g., 'clip', 'siglip2').
            caption_dir: Directory containing caption files.
                If None, captions are expected in image_dir.
            caption_file_suffix: File extension for captions (default ".txt").
            image_list_file: Text file listing image filenames.
                If None, globs image_dir for supported image extensions.
            batch_size: Batch size for iteration.
            dtype: Numpy dtype for output images.
        """
        self.num_channels, self.height, self.width = shape
        self.batch_size = batch_size
        self.dtype = dtype

        mean, std, resampling, center_crop = get_preprocessing_params(
            model_type, adaptor_name
        )
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resampling = resampling
        self.center_crop = center_crop

        self.image_paths: List[str] = []
        self.captions: List[str] = []
        self._load_pairs(
            image_dir, caption_dir, caption_file_suffix, image_list_file
        )

        self.n_samples = len(self.image_paths)
        n_full_batches = self.n_samples // self.batch_size
        has_remainder = 1 if self.n_samples % self.batch_size else 0
        self.n_batches = n_full_batches + has_remainder
        self._current_batch = 0

    def _load_pairs(
        self,
        image_dir: str,
        caption_dir: Optional[str],
        caption_file_suffix: str,
        image_list_file: Optional[str]
    ) -> None:
        """Load image paths and corresponding captions."""
        image_root = Path(image_dir)
        caption_root = Path(caption_dir) if caption_dir else image_root

        if image_list_file:
            with open(image_list_file, 'r', encoding='utf-8') as f:
                image_names = [ln.strip() for ln in f if ln.strip()]
        else:
            image_names = sorted(
                p.name for p in image_root.iterdir()
                if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )

        for name in image_names:
            img_path = image_root / name
            caption_fname = Path(name).stem + caption_file_suffix
            caption_path = caption_root / caption_fname

            if not img_path.exists():
                continue
            if not caption_path.exists():
                continue

            caption = caption_path.read_text(encoding='utf-8').strip()
            if not caption:
                continue

            self.image_paths.append(str(img_path))
            self.captions.append(caption)

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        img = Image.open(image_path).convert('RGB')
        if self.center_crop:
            img = _resize_shortest_edge(img, self.height, self.resampling)
            img = _center_crop(img, self.width, self.height)
        else:
            img = img.resize((self.width, self.height), self.resampling)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        if self.dtype is not None:
            img = img.astype(self.dtype)
        return img

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches

    def __iter__(self):
        """Iterate over batches."""
        self._current_batch = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, List[str]]:
        """Return next batch of (images, captions).

        Returns:
            images: np.ndarray of shape (B, C, H, W).
            captions: List[str] of length B.
        """
        if self._current_batch >= self.n_batches:
            raise StopIteration

        start = self._current_batch * self.batch_size
        end = min(start + self.batch_size, self.n_samples)

        images = [
            self._preprocess_image(self.image_paths[idx])
            for idx in range(start, end)
        ]
        captions = self.captions[start:end]

        self._current_batch += 1
        return np.array(images), captions

    def get_all_captions(self) -> List[str]:
        """Return all captions in dataset order."""
        return self.captions.copy()

    def get_all_image_paths(self) -> List[str]:
        """Return all image paths in dataset order."""
        return self.image_paths.copy()
