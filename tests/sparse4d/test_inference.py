# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the complete file/state/output loop with a CPU-only engine double."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from nvidia_tao_deploy.config.sparse4d.default_config import ExperimentConfig, ModelConfig
from nvidia_tao_deploy.cv.sparse4d.contract import OUTPUTS, input_shapes, sha256
from nvidia_tao_deploy.cv.sparse4d import inferencer


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Generate a complete three-frame test deployment without external files."""
    cfg = ExperimentConfig()
    cfg.model = ModelConfig(num_queries=3, num_temp_instances=2, embed_dims=4,
                            num_classes=2, num_cameras=1, input_height=4, input_width=8)
    cfg.inference.num_output = 3
    cfg.inference.save_raw = True
    cfg.results_dir = str(tmp_path / "results")
    shapes = input_shapes(cfg.model)
    shapes.pop("timestamp")
    arrays = {k: np.zeros(shapes[k], np.float32) for k in ("img", "projection_mat", "image_wh")}
    arrays["image_wh"][:] = [8, 4]
    np.savez(tmp_path / "frame.npz", **arrays)
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({"version": 1, "frames": [
        {"scene": scene, "timestamp": t, "tensors": "frame.npz"}
        for scene, t in (("a", 0), ("a", 0.1), ("b", 0))]}))
    cfg.inference.manifest = str(manifest)
    engine = tmp_path / "fake.engine"
    engine.write_bytes(b"engine double")
    cfg.inference.trt_engine = str(engine)
    cfg.plugin.sha256 = "a" * 64
    metadata = {"version": 1, "model_name": "sparse4d", "engine_sha256": sha256(engine),
                "plugin_sha256": cfg.plugin.sha256, "tensorrt": "10.test",
                "plugin_conformance": {"float32": "passed", "float16": "passed"},
                "model": vars(cfg.model), "input_shapes": {k: list(v) for k, v in shapes.items()}}
    (tmp_path / "fake.engine.json").write_text(json.dumps(metadata))
    observed = {"inputs": [], "closed": False}

    class Runner:
        """Minimal exact engine API for testing the orchestration layer."""

        input_names = list(shapes)

        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            observed["closed"] = True

        def __call__(self, inputs):
            observed["inputs"].append(inputs)
            result = {}
            for name in OUTPUTS:
                width = 2 if name.startswith(("classification", "quality")) else 4 if name == "output_cached_feature" else 11
                result[name] = np.ones((1, 3, width), np.float32)
            if observed.get("poison"):
                result["quality1"][0, 0, 0] = np.nan
            return result

    monkeypatch.setattr(inferencer, "EngineRunner", Runner)
    monkeypatch.setattr(inferencer, "trt_modules", lambda: (SimpleNamespace(__version__="10.test"), None))
    monkeypatch.setattr(inferencer, "load_plugin", lambda _: object())
    return cfg, observed


def test_inference_end_to_end(deployment, tmp_path):
    """Run cold start, continuation and scene reset; verify IDs/artifacts."""
    cfg, observed = deployment
    result = inferencer.run_inference(cfg)
    assert result["frames"] == 3
    assert result["detections"] == 9
    assert result["temporal_resets"] == 2
    assert observed["closed"]
    assert [int(x["prev_exists"].item()) for x in observed["inputs"]] == [0, 1, 0]
    predictions = [json.loads(line) for line in (tmp_path / "results/predictions.jsonl").read_text().splitlines()]
    assert predictions[0]["instance_ids"] == [0, 1, 2]
    assert predictions[1]["instance_ids"] == [0, 1, 3]
    assert predictions[2]["instance_ids"] == [4, 5, 6]
    with np.load(tmp_path / "results/raw/000002.npz") as data:
        assert set(data.files) == set(OUTPUTS)
    with pytest.raises(ValueError, match="already exist"):
        inferencer.run_inference(cfg)


def test_inference_nonfinite_failure_cleanup(deployment, tmp_path):
    """A poisoned unused stage cannot produce a success summary."""
    cfg, observed = deployment
    observed["poison"] = True
    with pytest.raises(ValueError, match="NaN"):
        inferencer.run_inference(cfg)
    assert observed["closed"]
    assert not (tmp_path / "results/summary.json").exists()


def test_model_and_profile_mismatch(deployment, tmp_path):
    """Reject changed model settings and profiles, not just engine bytes."""
    cfg, _ = deployment
    cfg.model.num_cameras = 2
    with pytest.raises(ValueError, match="model settings"):
        inferencer.check_metadata(cfg, "10.test")
    cfg.model.num_cameras = 1
    path = tmp_path / "fake.engine.json"
    metadata = json.loads(path.read_text())
    metadata["input_shapes"]["img"][1] = 2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="profile"):
        inferencer.check_metadata(cfg, "10.test")
