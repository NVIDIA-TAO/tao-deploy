# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered Sparse4D inference with hash-bound engine/plugin provenance."""

import json
from pathlib import Path

import numpy as np

from nvidia_tao_deploy.cv.sparse4d.contract import (
    input_shapes, read_frame, read_manifest, sha256, validate_array, validate_names,
)
from nvidia_tao_deploy.cv.sparse4d.postprocessing import TemporalState
from nvidia_tao_deploy.cv.sparse4d.runtime import EngineRunner, load_plugin, trt_modules


def check_metadata(cfg, trt_version):
    """Refuse a stale engine, changed plugin, or incompatible deployment config."""
    engine = Path(cfg.inference.trt_engine)
    metadata = json.loads(Path(str(engine) + ".json").read_text(encoding="utf-8"))
    # Exact TensorRT version matching (including patch/build) is intentional:
    # these engines are not built with version compatibility; rebuild on upgrade.
    expected = {"version": 1, "model_name": "sparse4d", "engine_sha256": sha256(engine),
                "plugin_sha256": cfg.plugin.sha256.lower(), "tensorrt": trt_version,
                "plugin_conformance": {"float32": "passed", "float16": "passed"}}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Engine provenance mismatch; regenerate with this TensorRT and MSDA library")
    from nvidia_tao_deploy.config.sparse4d.default_config import ModelConfig  # pylint: disable=import-outside-toplevel
    configured = {key: getattr(cfg.model, key) for key in ModelConfig.__dataclass_fields__}
    if configured != metadata["model"]:
        raise ValueError("Inference model settings differ from the engine build")
    validate_names(metadata["input_shapes"])
    shapes = input_shapes(cfg.model)
    if any(shape != list(shapes[name]) for name, shape in metadata["input_shapes"].items()):
        raise ValueError("Engine profile in sidecar disagrees with model settings")
    return metadata


def run_inference(cfg):
    """Write decoded JSON lines and optional raw tensors; do not overwrite results."""
    shapes = input_shapes(cfg.model)
    state = TemporalState(cfg.model, cfg.inference)
    records = list(read_manifest(cfg.inference.manifest))
    root = Path(cfg.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_file = root / "predictions.jsonl"
    summary_file = root / "summary.json"
    if output_file.exists() or summary_file.exists() or (root / "raw").exists():
        raise ValueError("Inference artifacts already exist; choose a new results_dir")
    trt, _ = trt_modules()
    metadata = check_metadata(cfg, trt.__version__)
    plugin_handle = load_plugin(cfg.plugin)
    if cfg.inference.save_raw:
        (root / "raw").mkdir()
    frames, detections, resets = 0, 0, 0
    with EngineRunner(Path(cfg.inference.trt_engine).read_bytes()) as runner, output_file.open("x", encoding="utf-8") as stream:
        validate_names(runner.input_names)
        if set(runner.input_names) != set(metadata["input_shapes"]):
            raise ValueError("Engine inputs disagree with its sidecar")
        for index, (record, tensor_path) in enumerate(records):
            inputs, pose = read_frame(tensor_path, cfg.model)
            inputs.update(state.prepare(record["scene"], record["timestamp"], pose))
            resets += int(inputs["prev_exists"][0] == 0)
            if "timestamp" in runner.input_names:
                inputs["timestamp"] = np.array([record["timestamp"]], np.float32)
            for name, value in inputs.items():
                validate_array(name, value, shapes[name], np.bool_ if name == "interval_mask" else np.float32)
            outputs = runner(inputs)
            decoded = state.update(outputs)
            result = {"frame_index": index, "scene": record["scene"], "timestamp": record["timestamp"],
                      "temporal_reset": bool(inputs["prev_exists"][0] == 0),
                      **{name: value.tolist() for name, value in decoded.items()}}
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            if cfg.inference.save_raw:
                np.savez_compressed(root / "raw" / f"{index:06d}.npz", **outputs)
            frames += 1
            detections += len(decoded["scores_3d"])
    summary = {"frames": frames, "detections": detections, "temporal_resets": resets,
               "all_outputs_finite": True, "engine_sha256": metadata["engine_sha256"],
               "plugin_sha256": metadata["plugin_sha256"], "tensorrt": trt.__version__,
               "manifest_sha256": sha256(cfg.inference.manifest)}
    with summary_file.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    del plugin_handle
    return summary
