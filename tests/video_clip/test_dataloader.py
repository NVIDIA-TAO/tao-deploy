# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP dataloader unit tests (frame preprocessing + chunk resolution).

These are the train/serve preprocessing-parity guards in unit form.
"""

import json

import numpy as np
import pytest
from PIL import Image

from nvidia_tao_deploy.multimodal.video_clip.dataloader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VideoCLIPGalleryLoader,
    build_chunk_index,
    canonicalize_text,
    linspace_indices,
    preprocess_frames,
)


class TestLinspaceIndices:
    def test_enough_frames_uniform(self):
        idx = linspace_indices(100, 8)
        assert idx.tolist() == [0, 14, 28, 42, 56, 70, 84, 99]

    def test_fewer_frames_repeat_last(self):
        idx = linspace_indices(3, 8)
        assert idx.tolist() == [0, 1, 2, 2, 2, 2, 2, 2]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            linspace_indices(0, 8)


class TestPreprocessFrames:
    def test_shape_and_dtype(self):
        frames = [Image.new("RGB", (64, 48), (10, 20, 30)) for _ in range(8)]
        out = preprocess_frames(frames, 224)
        assert out.shape == (8, 3, 224, 224)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_imagenet_normalization_and_channel_order(self):
        # Solid colour -> every pixel equals (c/255 - mean)/std, per channel.
        color = (255, 0, 128)
        out = preprocess_frames([Image.new("RGB", (32, 32), color)], 224)
        expected = (np.array(color, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        for c in range(3):
            np.testing.assert_allclose(out[0, c], expected[c], rtol=1e-4, atol=1e-4)

    def test_not_clip_normalization(self):
        # Guard against accidentally using CLIP mean/std (0.481.., 0.268..).
        # Use black, where ImageNet (-2.118) and CLIP (-1.792) norms differ sharply.
        out = preprocess_frames([Image.new("RGB", (16, 16), (0, 0, 0))], 224)
        clip_expected = (0.0 - 0.48145466) / 0.26862954   # channel 0, CLIP norm
        assert abs(float(out[0, 0, 0, 0]) - clip_expected) > 0.1


class TestChunkIndex:
    def _write_meta(self, tmp_path):
        data = [{
            "dataset": "CHAD", "video_id": "1_086_1", "video_path": "/data/a.mp4",
            "chunks": [
                {"chunk_index": 0, "start_time_sec": 0.0, "end_time_sec": 4.0,
                 "start_frame": 0, "end_frame": 120},
                {"chunk_index": 1, "start_time_sec": 4.0, "end_time_sec": 8.0,
                 "start_frame": 120, "end_frame": 240},
            ],
        }]
        p = tmp_path / "meta.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_sample_id_format_and_fields(self, tmp_path):
        idx = build_chunk_index(self._write_meta(tmp_path))
        assert "CHAD/1_086_1#0" in idx and "CHAD/1_086_1#1" in idx
        rec = idx["CHAD/1_086_1#0"]
        assert rec["video_path"] == "/data/a.mp4"
        assert rec["end_frame"] == 120

    def test_path_prefix_mapping(self, tmp_path):
        idx = build_chunk_index(
            self._write_meta(tmp_path),
            path_prefix_mapping={"/data": "/mnt/local"},
        )
        assert idx["CHAD/1_086_1#0"]["video_path"] == "/mnt/local/a.mp4"

    def test_gallery_loader_missing_id_raises(self, tmp_path):
        idx = build_chunk_index(self._write_meta(tmp_path))
        with pytest.raises(KeyError):
            VideoCLIPGalleryLoader(
                ["CHAD/1_086_1#0", "MISSING/x#0"], idx,
                num_frames=8, image_size=224,
            )

    def test_wrapped_data_key(self, tmp_path):
        # {"data": [...]} form is accepted.
        recs = json.loads(open(self._write_meta(tmp_path)).read())
        p = tmp_path / "wrapped.json"
        p.write_text(json.dumps({"data": recs}))
        idx = build_chunk_index(str(p))
        assert "CHAD/1_086_1#0" in idx

    def test_bad_format_raises_clear_error(self, tmp_path):
        # A dict without a list 'data' is a clear ValueError, not an AttributeError.
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"CHAD/1_086_1#0": {"video_path": "/x.mp4"}}))
        with pytest.raises(ValueError, match="Unrecognized chunk-metadata format"):
            build_chunk_index(str(p))


class TestCanonicalizeText:
    def test_lowercase_and_strip_punctuation(self):
        assert canonicalize_text("A Black_Car, running!") == "a black car running"

    def test_collapses_whitespace(self):
        assert canonicalize_text("  two   people  ") == "two people"
