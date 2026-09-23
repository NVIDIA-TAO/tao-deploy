# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained deployment-contract tests: no external models/data/GPU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from nvidia_tao_deploy.config.sparse4d.default_config import ExperimentConfig, ModelConfig
from nvidia_tao_deploy.cv.sparse4d.contract import (
    input_shapes, read_frame, read_manifest, sha256, validate_array, validate_names,
)
from nvidia_tao_deploy.cv.sparse4d.inferencer import check_metadata
from nvidia_tao_deploy.cv.sparse4d.runtime import load_plugin


def test_schema_independent_mutable_defaults():
    """Configs are independently owned and compose without training dependencies."""
    first, second = ExperimentConfig(), ExperimentConfig()
    first.inference.gpu_ids.append(1)
    assert second.inference.gpu_ids == [0]
    cfg = OmegaConf.structured(second)
    assert cfg.model_name == "sparse4d"
    assert cfg.gen_trt_engine.data_type == "fp32"


@pytest.mark.parametrize("timestamp", [False, True])
def test_optional_timestamp(timestamp):
    """Export may prune timestamp; no other tensor is optional."""
    names = set(input_shapes(ModelConfig()))
    if not timestamp:
        names.remove("timestamp")
    validate_names(names)
    with pytest.raises(ValueError):
        validate_names(names - {"interval_mask"})
    with pytest.raises(ValueError):
        validate_names(names | {"unknown"})


@pytest.mark.parametrize("field,value", [
    ("num_cameras", 0), ("input_width", -1), ("anchor_dims", 10), ("num_temp_instances", 901)])
def test_invalid_dimensions(field, value):
    """Fail before runtime allocation."""
    model = ModelConfig()
    setattr(model, field, value)
    with pytest.raises(ValueError):
        input_shapes(model)


@pytest.mark.parametrize("value", [
    np.zeros((2,), np.float32), np.zeros((1,), np.float64),
    np.array([np.nan], np.float32), np.array([np.inf], np.float32)])
def test_invalid_array(value):
    """Shape, dtype and nonfinite inputs cannot be silently coerced."""
    with pytest.raises(ValueError):
        validate_array("input", value, (1,))


def test_read_frame_and_pose(tmp_path):
    """Camera layout, width/height convention and rigid pose are explicit."""
    model = ModelConfig(num_cameras=2, input_height=4, input_width=8)
    shape = input_shapes(model)
    arrays = {key: np.zeros(shape[key], np.float32) for key in ("img", "projection_mat", "image_wh")}
    arrays["image_wh"][:] = [8, 4]
    path = tmp_path / "frame.npz"
    np.savez(path, **arrays)
    frame, pose = read_frame(path, model)
    np.testing.assert_array_equal(pose, np.eye(4))
    assert frame["img"].shape == (1, 2, 3, 4, 8)
    arrays["image_wh"][:] = [4, 8]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="width, height"):
        read_frame(path, model)
    arrays["image_wh"][:] = [8, 4]
    arrays["T_global"] = np.zeros((4, 4), np.float32)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="rigid"):
        read_frame(path, model)


def test_frame_rejects_pickle(tmp_path):
    """A deploy-side frame must never unpickle Python objects."""
    path = tmp_path / "unsafe.npz"
    np.savez(path, img=np.array([{}], dtype=object), projection_mat=np.zeros(1), image_wh=np.zeros(1))
    with pytest.raises(ValueError, match="Object arrays"):
        read_frame(path, ModelConfig())


@pytest.mark.parametrize("change", [
    {"version": 2}, {"frames": []},
    {"frames": [{"scene": "a", "timestamp": float("nan"), "tensors": "a.npz"}]},
    {"frames": [{"scene": "", "timestamp": 0, "tensors": "a.npz"}]},
    {"frames": [{"scene": "a", "timestamp": 0}]}])
def test_bad_manifest(tmp_path, change):
    """Malformed metadata is rejected before inference."""
    manifest = {"version": 1, "frames": [{"scene": "a", "timestamp": 0, "tensors": "a.npz"}]}
    manifest.update(change)
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        list(read_manifest(path))


def test_manifest_preserves_order(tmp_path):
    """Input order is part of the temporal API; timestamps aren't silently sorted."""
    path = tmp_path / "frames.json"
    path.write_text(json.dumps({"version": 1, "frames": [
        {"scene": "a", "timestamp": t, "tensors": f"{t}.npz"} for t in (2, 0, 1)]}))
    records = list(read_manifest(path))
    assert [r[0]["timestamp"] for r in records] == [2, 0, 1]
    assert records[0][1] == tmp_path / "2.npz"


def test_plugin_pinning_before_loading(tmp_path):
    """Never dlopen an absent/unpinned binary."""
    path = tmp_path / "plugin.so"
    path.write_bytes(b"not a shared library")
    with pytest.raises(ValueError, match="sha256"):
        load_plugin(SimpleNamespace(path=str(path), sha256="0" * 64))
    with pytest.raises(ValueError, match="absolute"):
        load_plugin(SimpleNamespace(path="plugin.so", sha256=sha256(path)))


@pytest.mark.parametrize("field,value", [
    ("engine_sha256", "wrong"), ("plugin_sha256", "wrong"),
    ("tensorrt", "10.other"), ("plugin_conformance", {})])
def test_metadata_tampering(tmp_path, field, value):
    """Serialized engine, plugin and tested runtime version remain bound together."""
    cfg = ExperimentConfig()
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"fake engine")
    cfg.inference.trt_engine = str(engine)
    cfg.plugin.sha256 = "a" * 64
    metadata = {"version": 1, "model_name": "sparse4d", "engine_sha256": sha256(engine),
                "plugin_sha256": cfg.plugin.sha256, "tensorrt": "10.test",
                "plugin_conformance": {"float32": "passed", "float16": "passed"},
                "model": vars(cfg.model),
                "input_shapes": {k: list(v) for k, v in input_shapes(cfg.model).items()}}
    sidecar = tmp_path / "model.engine.json"
    sidecar.write_text(json.dumps(metadata))
    assert check_metadata(cfg, "10.test") == metadata
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        check_metadata(cfg, "10.test")
