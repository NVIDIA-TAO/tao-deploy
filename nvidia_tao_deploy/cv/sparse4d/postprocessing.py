# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy recurrent state and decoding for the batch-one Sparse4D export ABI."""

import numpy as np

from nvidia_tao_deploy.cv.sparse4d.contract import OUTPUTS, validate_array


def sigmoid(value):
    """Stable sigmoid for logits of either sign."""
    return np.exp(-np.logaddexp(0, -value))


def top_indices(scores, count):
    """Descending top-k with deterministic query-index tie breaking."""
    return np.argsort(-scores, kind="stable")[:count]


def validate_outputs(outputs, model):
    """Check all stages, including tensors not used by final decoding."""
    if set(outputs) != set(OUTPUTS):
        raise ValueError("Engine outputs do not match the six-stage Sparse4D ABI")
    for name, value in outputs.items():
        width = (model.num_classes if name.startswith("classification") else
                 2 if name.startswith("quality") else
                 model.embed_dims if name == "output_cached_feature" else model.anchor_dims)
        validate_array(name, value, (1, model.num_queries, width))


def decode(outputs, instance_ids, num_output, score_threshold):
    """Match tracking-mode SparseBox3DDecoder: threshold BEFORE quality weighting."""
    probabilities = sigmoid(outputs["classification2"][0])
    labels = probabilities.argmax(axis=-1)
    raw_scores = probabilities.max(axis=-1)
    indices = top_indices(raw_scores, num_output)
    scores = raw_scores[indices] * sigmoid(outputs["quality2"][0, indices, 0])
    order = top_indices(scores, len(scores))
    indices, scores = indices[order], scores[order]
    keep = raw_scores[indices] >= score_threshold
    indices, scores = indices[keep], scores[keep]
    anchors = outputs["prediction6"][0, indices]
    with np.errstate(over="ignore"):
        boxes = np.concatenate((anchors[:, :3], np.exp(anchors[:, 3:6]),
                                np.arctan2(anchors[:, 6:7], anchors[:, 7:8]), anchors[:, 8:]), axis=-1)
    if not np.isfinite(boxes).all():
        raise ValueError("Decoded boxes contain nonfinite dimensions")
    return {"boxes_3d": boxes, "scores_3d": scores, "labels_3d": labels[indices],
            "cls_scores": raw_scores[indices], "instance_ids": instance_ids[indices]}


class TemporalState:
    """One ordered stream; scene/gap resets clear cache but never reuse allocated IDs.

    The exported graph handles query fusion, not cache selection or projection.
    Reset on non-increasing timestamps as well as large gaps; batch > 1 is not
    supported. The graph's refinement interval may be baked in by its exporter.
    """

    def __init__(self, model, config):
        """Initialize empty state and validate runtime settings."""
        self.model, self.config = model, config
        if any(not 0 <= value <= 1 for value in
               (config.confidence_decay, config.tracking_threshold, config.score_threshold)):
            raise ValueError("Thresholds and confidence_decay must be finite values in [0, 1]")
        if (config.max_time_interval <= 0 or not np.isfinite(config.max_time_interval) or
                config.num_output <= 0 or config.num_output > model.num_queries):
            raise ValueError("Invalid cache, threshold, time interval or output count settings")
        self.next_id = 0
        self.reset()

    def reset(self):
        """Clear temporal tensors without restarting the global ID counter."""
        self.feature = self.anchor = self.confidence = self.ids = None
        self.timestamp = self.scene = self.pose = None

    def prepare(self, scene, timestamp, pose):
        """Project the previous top-k anchors into this frame's coordinates."""
        m = self.model
        valid = (self.feature is not None and scene == self.scene and
                 0 < timestamp - self.timestamp <= self.config.max_time_interval)
        if not valid:
            self.reset()
            feature = np.zeros((1, m.num_temp_instances, m.embed_dims), np.float32)
            anchor = np.zeros((1, m.num_temp_instances, m.anchor_dims), np.float32)
        else:
            feature = self.feature
            anchor = self.anchor.copy()
            delta = np.float32(timestamp - self.timestamp)
            transform = np.linalg.inv(pose) @ self.pose
            centers = anchor[0, :, :3] + anchor[0, :, 8:] * delta
            anchor[0, :, :3] = centers @ transform[:3, :3].T + transform[:3, 3]
            yaw = anchor[0][:, [7, 6]] @ transform[:2, :2].T
            anchor[0, :, 6:8] = yaw[:, [1, 0]]
            anchor[0, :, 8:] = anchor[0, :, 8:] @ transform[:3, :3].T
        self.scene, self.timestamp, self.pose = scene, timestamp, pose.copy()
        return {"input_cached_feature": feature, "input_cached_anchor": anchor,
                "prev_exists": np.array([valid], np.float32),
                "interval_mask": np.array([[[valid]]], np.bool_)}

    def update(self, outputs):
        """Select recurrent top-k with decayed confidence and propagate query IDs."""
        validate_outputs(outputs, self.model)
        raw_confidence = sigmoid(outputs["classification2"][0]).max(axis=-1)
        ids = np.full(self.model.num_queries, -1, np.int64)
        if self.ids is not None:
            ids[:self.model.num_temp_instances] = self.ids
        new = (ids < 0) & (raw_confidence >= self.config.tracking_threshold)
        count = int(new.sum())
        ids[new] = np.arange(self.next_id, self.next_id + count)
        self.next_id += count
        confidence = raw_confidence.copy()
        if self.confidence is not None:
            confidence[:self.model.num_temp_instances] = np.maximum(
                confidence[:self.model.num_temp_instances], self.confidence * self.config.confidence_decay)
        selected = top_indices(confidence, self.model.num_temp_instances)
        self.feature = np.ascontiguousarray(outputs["output_cached_feature"][:, selected])
        self.anchor = np.ascontiguousarray(outputs["output_cached_anchor"][:, selected])
        self.confidence, self.ids = confidence[selected], ids[selected]
        return decode(outputs, ids, self.config.num_output, self.config.score_threshold)
