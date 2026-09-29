# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated fixtures exercise recurrent cache and SparseBox3D decoding."""

import numpy as np
import pytest

from nvidia_tao_deploy.config.sparse4d.default_config import ModelConfig, InferenceConfig
from nvidia_tao_deploy.cv.sparse4d.contract import OUTPUTS
from nvidia_tao_deploy.cv.sparse4d.postprocessing import TemporalState, decode, sigmoid, validate_outputs


pytestmark = pytest.mark.sparse4d


@pytest.fixture
def sample():
    """Tiny pure NumPy ABI fixture; no weights, images or annotation files."""
    model = ModelConfig(num_queries=3, num_temp_instances=2, embed_dims=4, num_classes=2)
    config = InferenceConfig(num_output=3)
    outputs = {}
    for name in OUTPUTS:
        width = 2 if name.startswith(("classification", "quality")) else 4 if name == "output_cached_feature" else 11
        outputs[name] = np.zeros((1, 3, width), np.float32)
    outputs["classification2"][0] = [[3, -2], [2, -1], [-2, 1]]
    outputs["output_cached_feature"][0] = np.arange(12).reshape(3, 4)
    outputs["output_cached_anchor"][0, :, 7] = 1
    outputs["output_cached_anchor"][0, :, 8] = 2
    return model, config, outputs


def test_first_frame_cache_and_projection(sample):
    """Top-k feedback, velocity and ego-motion match InstanceBank convention."""
    model, config, outputs = sample
    state = TemporalState(model, config)
    first = state.prepare("a", 0, np.eye(4, dtype=np.float32))
    assert not first["prev_exists"].any()
    decoded = state.update(outputs)
    assert decoded["instance_ids"].tolist() == [0, 1, 2]
    assert state.feature.shape == (1, 2, 4)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 0.25
    second = state.prepare("a", 0.5, pose)
    assert second["prev_exists"].item() == 1
    np.testing.assert_allclose(second["input_cached_anchor"][0, :, 0], 0.75)
    np.testing.assert_array_equal(second["input_cached_feature"], outputs["output_cached_feature"][:, :2])


def test_rotation_projection(sample):
    """Yaw and velocity rotate along with positions."""
    model, config, outputs = sample
    state = TemporalState(model, config)
    state.prepare("a", 0, np.eye(4, dtype=np.float32))
    state.update(outputs)
    pose = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    anchor = state.prepare("a", 0.5, pose)["input_cached_anchor"][0]
    np.testing.assert_allclose(anchor[:, 6:8], [[-1, 0], [-1, 0]])
    np.testing.assert_allclose(anchor[:, 8:], [[0, -2, 0], [0, -2, 0]])


@pytest.mark.parametrize("scene,timestamp", [("b", 0.1), ("a", 0), ("a", -1), ("a", 3)])
def test_boundaries_do_not_reuse_ids(sample, scene, timestamp):
    """Cache miss/reset forgets tensors and confidence, but keeps allocated IDs."""
    model, config, outputs = sample
    state = TemporalState(model, config)
    state.prepare("a", 0, np.eye(4, dtype=np.float32))
    state.update(outputs)
    inputs = state.prepare(scene, timestamp, np.eye(4, dtype=np.float32))
    assert not inputs["prev_exists"].item()
    assert not inputs["input_cached_feature"].any()
    assert state.confidence is None
    assert state.update(outputs)["instance_ids"].tolist() == [3, 4, 5]


def test_confidence_decay_and_ids(sample):
    """A previously confident track can survive one low-confidence frame."""
    model, config, outputs = sample
    state = TemporalState(model, config)
    state.prepare("a", 0, np.eye(4, dtype=np.float32))
    state.update(outputs)
    state.prepare("a", 1, np.eye(4, dtype=np.float32))
    outputs["classification2"][0] = [[-10, -10], [-10, -10], [0, 0]]
    state.update(outputs)
    np.testing.assert_array_equal(state.ids, [0, 1])
    np.testing.assert_allclose(state.confidence, sigmoid(np.array([3, 2])) * 0.8, rtol=1e-6)


def test_quality_reorders_after_raw_threshold(sample):
    """Raw class threshold must not be incorrectly applied to weighted score."""
    _, _, outputs = sample
    outputs["quality2"][0, :, 0] = [-8, 8, 0]
    outputs["prediction6"][0, :, 3:6] = np.log(2)
    outputs["prediction6"][0, :, 6] = 1
    result = decode(outputs, np.array([10, 20, 30]), 3, 0.8)
    assert result["instance_ids"].tolist() == [20, 10]
    assert result["scores_3d"][1] < 0.01
    assert result["cls_scores"].min() >= 0.8
    np.testing.assert_allclose(result["boxes_3d"][:, 3:6], 2)
    np.testing.assert_allclose(result["boxes_3d"][:, 6], np.pi / 2)


def test_nonfinite_outputs_and_box_overflow(sample):
    """Check even unused decoder stages, then check decoded dimensions."""
    model, _, outputs = sample
    outputs["prediction1"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_outputs(outputs, model)
    outputs["prediction1"][0, 0, 0] = 0
    outputs["prediction6"][0, :, 3] = 1000
    with pytest.raises(ValueError, match="nonfinite"):
        decode(outputs, np.arange(3), 3, 0)


@pytest.mark.parametrize("field,value", [
    ("num_output", 4), ("confidence_decay", -1), ("max_time_interval", 0),
    ("max_time_interval", np.nan), ("score_threshold", 2), ("tracking_threshold", -1)])
def test_bad_runtime_config(sample, field, value):
    """Reject invalid runtime controls before consuming frames."""
    model, config, _ = sample
    setattr(config, field, value)
    with pytest.raises(ValueError):
        TemporalState(model, config)
