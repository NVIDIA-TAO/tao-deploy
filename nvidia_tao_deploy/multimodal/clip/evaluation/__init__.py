# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
