# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Dataloader Unit Tests."""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from nvidia_tao_deploy.multimodal.clip.dataloader import CLIPRetrievalLoader


def _create_retrieval_dataset(tmpdir, samples):
    """Create retrieval dataset with images and caption files.

    Args:
        tmpdir: Root directory.
        samples: List of (image_name, caption_text) tuples.

    Returns:
        (image_dir, caption_dir) tuple.
    """
    image_dir = os.path.join(tmpdir, "images")
    caption_dir = os.path.join(tmpdir, "captions")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)

    for img_name, caption in samples:
        img_path = os.path.join(image_dir, img_name)
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(img_path)

        stem = os.path.splitext(img_name)[0]
        caption_path = os.path.join(caption_dir, f"{stem}.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

    return image_dir, caption_dir


class TestCLIPRetrievalLoaderInit:
    """Tests for CLIPRetrievalLoader initialization."""

    def test_basic_init(self):
        """Test basic initialization with images and captions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = [
                ("img_0001.jpg", "a photo of a cat"),
                ("img_0002.jpg", "a photo of a dog"),
                ("img_0003.jpg", "a photo of a bird"),
                ("img_0004.jpg", "a photo of a fish"),
            ]
            image_dir, caption_dir = _create_retrieval_dataset(tmpdir, samples)

            dl = CLIPRetrievalLoader(
                shape=(3, 224, 224),
                image_dir=image_dir,
                caption_dir=caption_dir,
                batch_size=2,
                dtype=np.float32,
                model_type="clip",
            )
            assert dl.height == 224
            assert dl.width == 224
            assert dl.num_channels == 3
            assert dl.n_samples == 4
            assert dl.n_batches == 2
            assert dl.batch_size == 2

    def test_caption_in_same_dir(self):
        """Test when captions are in the same directory as images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = tmpdir

            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(os.path.join(image_dir, f"img_{i:04d}.jpg"))
                with open(
                    os.path.join(image_dir, f"img_{i:04d}.txt"), "w",
                    encoding="utf-8"
                ) as f:
                    f.write(f"caption {i}")

            dl = CLIPRetrievalLoader(
                shape=(3, 224, 224),
                image_dir=image_dir,
                caption_dir=None,
                batch_size=2,
                dtype=np.float32,
                model_type="clip",
            )
            assert dl.n_samples == 4

    def test_handles_partial_batch(self):
        """Test that partial batches are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = [
                ("img_0001.jpg", "caption 1"),
                ("img_0002.jpg", "caption 2"),
                ("img_0003.jpg", "caption 3"),
            ]
            image_dir, caption_dir = _create_retrieval_dataset(tmpdir, samples)

            dl = CLIPRetrievalLoader(
                shape=(3, 224, 224),
                image_dir=image_dir,
                caption_dir=caption_dir,
                batch_size=2,
                dtype=np.float32,
                model_type="clip",
            )
            assert dl.n_samples == 3
            assert dl.n_batches == 2

    def test_missing_caption_skipped(self):
        """Test that images without captions are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = [
                ("img_0001.jpg", "caption 1"),
                ("img_0002.jpg", "caption 2"),
                ("img_0003.jpg", "caption 3"),
                ("img_0004.jpg", "caption 4"),
            ]
            image_dir, caption_dir = _create_retrieval_dataset(tmpdir, samples)

            os.remove(os.path.join(caption_dir, "img_0001.txt"))

            dl = CLIPRetrievalLoader(
                shape=(3, 224, 224),
                image_dir=image_dir,
                caption_dir=caption_dir,
                batch_size=1,
                dtype=np.float32,
                model_type="clip",
            )
            assert dl.n_samples == 3

    def test_empty_caption_skipped(self):
        """Test that images with empty captions are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = [
                ("img_0001.jpg", "caption 1"),
                ("img_0002.jpg", ""),
                ("img_0003.jpg", "caption 3"),
                ("img_0004.jpg", "caption 4"),
            ]
            image_dir, caption_dir = _create_retrieval_dataset(tmpdir, samples)

            dl = CLIPRetrievalLoader(
                shape=(3, 224, 224),
                image_dir=image_dir,
                caption_dir=caption_dir,
                batch_size=1,
                dtype=np.float32,
                model_type="clip",
            )
            assert dl.n_samples == 3


class TestCLIPRetrievalLoaderIteration:
    """Tests for CLIPRetrievalLoader __iter__ / __next__."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create a test loader."""
        samples = [
            ("img_0001.jpg", "a photo of a cat"),
            ("img_0002.jpg", "a photo of a dog"),
            ("img_0003.jpg", "a photo of a bird"),
            ("img_0004.jpg", "a photo of a fish"),
            ("img_0005.jpg", "a photo of a horse"),
            ("img_0006.jpg", "a photo of a cow"),
            ("img_0007.jpg", "a photo of a sheep"),
            ("img_0008.jpg", "a photo of a pig"),
        ]
        image_dir, caption_dir = _create_retrieval_dataset(str(tmp_path), samples)

        return CLIPRetrievalLoader(
            shape=(3, 32, 32),
            image_dir=image_dir,
            caption_dir=caption_dir,
            batch_size=4,
            dtype=np.float32,
            model_type="clip",
        )

    def test_len(self, loader):
        """Test __len__ returns correct number of batches."""
        assert len(loader) == 2

    def test_iter_returns_self(self, loader):
        """Test __iter__ returns self."""
        assert iter(loader) is loader

    def test_batch_shapes(self, loader):
        """Test that batches have correct shapes."""
        for imgs, captions in loader:
            assert imgs.shape == (4, 3, 32, 32)
            assert len(captions) == 4
            assert all(isinstance(c, str) for c in captions)

    def test_pixel_range(self, loader):
        """Test that pixel values are normalized (mean/std applied)."""
        for imgs, _ in loader:
            # After normalization, values can be negative and > 1
            # Just verify they're in a reasonable range for normalized data
            assert imgs.min() >= -3.0
            assert imgs.max() <= 3.0

    def test_captions_are_strings(self, loader):
        """Test that captions are returned as strings."""
        for _, captions in loader:
            for caption in captions:
                assert isinstance(caption, str)
                assert len(caption) > 0

    def test_iteration_count(self, loader):
        """Test correct number of iterations."""
        count = sum(1 for _ in loader)
        assert count == 2

    def test_reiter(self, loader):
        """Verify loader can be iterated multiple times."""
        first = sum(1 for _ in loader)
        second = sum(1 for _ in loader)
        assert first == second == 2

    def test_partial_final_batch(self, tmp_path):
        """Test that partial final batch is returned."""
        samples = [
            ("img_0001.jpg", "caption 1"),
            ("img_0002.jpg", "caption 2"),
            ("img_0003.jpg", "caption 3"),
            ("img_0004.jpg", "caption 4"),
            ("img_0005.jpg", "caption 5"),
        ]
        image_dir, caption_dir = _create_retrieval_dataset(str(tmp_path), samples)

        dl = CLIPRetrievalLoader(
            shape=(3, 32, 32),
            image_dir=image_dir,
            caption_dir=caption_dir,
            batch_size=4,
            dtype=np.float32,
            model_type="clip",
        )
        assert dl.n_batches == 2

        batches = list(dl)
        assert len(batches) == 2
        assert batches[0][0].shape[0] == 4
        assert batches[1][0].shape[0] == 1

    def test_resize_to_target(self, tmp_path):
        """Images created at 64x64 should be resized to target shape."""
        samples = [
            ("img_0001.jpg", "caption 1"),
            ("img_0002.jpg", "caption 2"),
        ]
        image_dir, caption_dir = _create_retrieval_dataset(str(tmp_path), samples)

        dl = CLIPRetrievalLoader(
            shape=(3, 128, 128),
            image_dir=image_dir,
            caption_dir=caption_dir,
            batch_size=2,
            dtype=np.float32,
            model_type="clip",
        )
        imgs, _ = next(iter(dl))
        assert imgs.shape == (2, 3, 128, 128)


class TestCLIPRetrievalLoaderHelpers:
    """Tests for CLIPRetrievalLoader helper methods."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create a test loader."""
        samples = [
            ("img_0001.jpg", "caption 1"),
            ("img_0002.jpg", "caption 2"),
            ("img_0003.jpg", "caption 3"),
            ("img_0004.jpg", "caption 4"),
        ]
        image_dir, caption_dir = _create_retrieval_dataset(str(tmp_path), samples)

        return CLIPRetrievalLoader(
            shape=(3, 32, 32),
            image_dir=image_dir,
            caption_dir=caption_dir,
            batch_size=2,
            dtype=np.float32,
            model_type="clip",
        )

    def test_get_all_captions(self, loader):
        """Test get_all_captions returns all captions."""
        captions = loader.get_all_captions()
        assert len(captions) == 4
        assert all(isinstance(c, str) for c in captions)

    def test_get_all_image_paths(self, loader):
        """Test get_all_image_paths returns all paths."""
        paths = loader.get_all_image_paths()
        assert len(paths) == 4
        assert all(os.path.exists(p) for p in paths)

    def test_helpers_return_copies(self, loader):
        """Test that helper methods return copies, not references."""
        captions1 = loader.get_all_captions()
        captions2 = loader.get_all_captions()
        assert captions1 is not captions2

        paths1 = loader.get_all_image_paths()
        paths2 = loader.get_all_image_paths()
        assert paths1 is not paths2


class TestCLIPRetrievalLoaderCustomSuffix:
    """Tests for custom caption file suffix."""

    def test_custom_suffix(self, tmp_path):
        """Test custom caption file suffix."""
        image_dir = tmp_path / "images"
        caption_dir = tmp_path / "captions"
        image_dir.mkdir()
        caption_dir.mkdir()

        for i in range(4):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(image_dir / f"img_{i:04d}.png")

            with open(
                caption_dir / f"img_{i:04d}.caption", "w", encoding="utf-8"
            ) as f:
                f.write(f"caption {i}")

        dl = CLIPRetrievalLoader(
            shape=(3, 32, 32),
            image_dir=str(image_dir),
            caption_dir=str(caption_dir),
            caption_file_suffix=".caption",
            batch_size=2,
            dtype=np.float32,
            model_type="clip",
        )
        assert dl.n_samples == 4
