# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP retrieval evaluation module for TensorRT inference."""

from nvidia_tao_deploy.multimodal.video_clip.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_ndcg,
)
from nvidia_tao_deploy.multimodal.video_clip.evaluation.retrieval import (
    RetrievalEvaluator,
    RetrievalMetrics,
    evaluate_by_slice,
    log_retrieval_metrics,
)

__all__ = [
    "compute_ap",
    "compute_auc",
    "compute_ndcg",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "evaluate_by_slice",
    "log_retrieval_metrics",
]
