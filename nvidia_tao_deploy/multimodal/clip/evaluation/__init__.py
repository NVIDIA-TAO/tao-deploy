# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP evaluation module for TensorRT inference."""

from nvidia_tao_deploy.multimodal.clip.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_ndcg,
)
from nvidia_tao_deploy.multimodal.clip.evaluation.retrieval import (
    RetrievalEvaluator,
    RetrievalMetrics,
    log_retrieval_metrics,
)

__all__ = [
    "compute_ap",
    "compute_auc",
    "compute_ndcg",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "log_retrieval_metrics",
]
