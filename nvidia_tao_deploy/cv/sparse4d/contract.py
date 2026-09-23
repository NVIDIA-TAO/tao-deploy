# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of the public Sparse4D export and frame contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np


OUTPUTS = ("classification1", "classification2", *(f"prediction{i}" for i in range(1, 7)),
           "quality1", "quality2", "output_cached_feature", "output_cached_anchor")


def sha256(path):
    """Hash large artifacts without reading them all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_shapes(model):
    """Return the supported profile; timestamp may be pruned by ONNX export."""
    values = (model.num_cameras, model.input_height, model.input_width,
              model.num_queries, model.num_temp_instances, model.embed_dims, model.num_classes)
    if any(v <= 0 for v in values) or model.num_temp_instances > model.num_queries:
        raise ValueError("Model dimensions must be positive and cache size <= query count")
    if model.anchor_dims != 11:
        raise ValueError("This Sparse4D backend requires 11D anchors (including 3D velocity)")
    return {
        "img": (1, model.num_cameras, 3, model.input_height, model.input_width),
        "projection_mat": (1, model.num_cameras, 4, 4),
        "image_wh": (1, model.num_cameras, 2),
        "input_cached_feature": (1, model.num_temp_instances, model.embed_dims),
        "input_cached_anchor": (1, model.num_temp_instances, model.anchor_dims),
        "prev_exists": (1,), "interval_mask": (1, 1, 1), "timestamp": (1,),
    }


def validate_names(names):
    """Reject unknown exports instead of guessing tensor order."""
    expected = {"img", "projection_mat", "image_wh", "input_cached_feature",
                "input_cached_anchor", "prev_exists", "interval_mask"}
    if set(names) not in (expected, expected | {"timestamp"}):
        raise ValueError(f"Unsupported Sparse4D inputs: {sorted(names)}")


def validate_array(name, value, shape, dtype=np.float32):
    """Reject accidental dtype, layout, shape and nonfinite input changes."""
    if value.shape != tuple(shape) or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name}: expected {tuple(shape)} {np.dtype(dtype)}, got {value.shape} {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(value)


def read_manifest(path):
    """Read ordered frame records. No pickle deserialization is supported."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("frames"), list) or not data["frames"]:
        raise ValueError("Expected manifest version: 1 with a nonempty frames list")
    for frame in data["frames"]:
        if not isinstance(frame.get("scene"), str) or not frame["scene"]:
            raise ValueError("Each frame needs a nonempty scene string")
        timestamp = frame.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
            raise ValueError("Each frame needs a finite numeric timestamp in seconds")
        frame_path = frame.get("tensors")
        if not isinstance(frame_path, str) or not frame_path:
            raise ValueError("Each frame needs a tensors NPZ path")
        yield frame, path.parent / frame_path


def read_frame(path, model):
    """Load already-normalized images and matching augmented calibration."""
    shapes = input_shapes(model)
    with np.load(path, allow_pickle=False) as data:
        if not {"img", "projection_mat", "image_wh"}.issubset(data.files):
            raise ValueError("Frame NPZ requires img, projection_mat and image_wh")
        if set(data.files) - {"img", "projection_mat", "image_wh", "T_global"}:
            raise ValueError("Frame NPZ contains unexpected fields")
        frame = {name: validate_array(name, data[name], shapes[name])
                 for name in ("img", "projection_mat", "image_wh")}
        expected_wh = np.array([model.input_width, model.input_height], np.float32)
        if not np.all(frame["image_wh"] == expected_wh):
            raise ValueError("image_wh must contain the network image width, height (in that order)")
        pose = data["T_global"] if "T_global" in data else np.eye(4, dtype=np.float32)
        pose = validate_array("T_global", pose, (4, 4))
        rotation = pose[:3, :3]
        if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError("T_global must be a rigid local-to-global transform")
    return frame, pose
