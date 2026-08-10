# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-relevance text->video retrieval evaluation for Video-CLIP TRT.

Reproduces the tao-pytorch trainer's retrieval evaluator
(``model/evaluation/retrieval.py``) with a numpy-only similarity path (no torch)
so the deploy-side headline number matches the reference validation curve. Each
query carries an explicit ``relevant_clip_ids`` set (verbatim, not derived from
idx grouping); mAP is the mean of per-query Average Precision over the cosine-
ranked gallery, with an optional per-slice breakdown.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from nvidia_tao_deploy.multimodal.video_clip.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_ndcg,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    median_rank: float = 0.0
    mean_rank: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    auc: float = 0.0
    num_queries: int = 0
    gallery_size: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a flat dictionary."""
        result = {
            'mAP': self.map_score,
            'median_rank': self.median_rank,
            'mean_rank': self.mean_rank,
            'auc': self.auc,
            'num_queries': self.num_queries,
            'gallery_size': self.gallery_size,
        }
        for k, v in self.recall_at_k.items():
            result[f'recall@{k}'] = v
        for k, v in self.hit_at_k.items():
            result[f'hit@{k}'] = v
        for k, v in self.ndcg_at_k.items():
            result[f'ndcg@{k}'] = v
        return result


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (matches the trainer's F.normalize)."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class RetrievalEvaluator:
    """Text->video retrieval evaluator (numpy).

    Args:
        k_values: Tuple of k values for Recall@k / Hit@k / NDCG@k.
        compute_auc: Whether to compute AUC.
    """

    def __init__(self, k_values: Tuple[int, ...] = (1, 5, 10),
                 compute_auc: bool = True):
        """Initialize the retrieval evaluator."""
        self.k_values = k_values
        self._compute_auc = compute_auc

    def evaluate(
        self,
        query_embs: np.ndarray,
        gallery_embs: np.ndarray,
        ground_truth: List[List[int]],
    ) -> RetrievalMetrics:
        """Compute retrieval metrics.

        Args:
            query_embs: Query embeddings (N, D).
            gallery_embs: Gallery embeddings (M, D).
            ground_truth: ``gt[i]`` = list of relevant gallery indices for query i.

        Returns:
            RetrievalMetrics.
        """
        query_embs = _l2_normalize(np.asarray(query_embs, dtype=np.float32))
        gallery_embs = _l2_normalize(np.asarray(gallery_embs, dtype=np.float32))
        n_queries = query_embs.shape[0]
        n_gallery = gallery_embs.shape[0]

        similarity_matrix = query_embs @ gallery_embs.T

        ap_scores: List[float] = []
        ranks: List[int] = []
        recall_scores = {k: [] for k in self.k_values}
        hit_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        auc_scores: List[float] = []

        for i in range(n_queries):
            sims = similarity_matrix[i]
            relevant_indices = set(ground_truth[i])
            n_pos = len(relevant_indices)
            if n_pos == 0:
                continue

            sorted_idx = np.argsort(-sims)
            sorted_labels = np.array(
                [1.0 if idx in relevant_indices else 0.0 for idx in sorted_idx]
            )

            ap_scores.append(compute_ap(sorted_labels))

            first_pos_rank = np.where(sorted_labels == 1)[0]
            if len(first_pos_rank) > 0:
                ranks.append(first_pos_rank[0] + 1)

            for k in self.k_values:
                hits_at_k = np.sum(sorted_labels[:k])
                recall_scores[k].append(hits_at_k / n_pos)
                hit_scores[k].append(1.0 if hits_at_k > 0 else 0.0)
                ndcg_scores[k].append(compute_ndcg(sorted_labels, k))

            if self._compute_auc:
                auc_scores.append(compute_auc(sims, sorted_labels))

        return RetrievalMetrics(
            recall_at_k={k: float(np.mean(v)) for k, v in recall_scores.items() if v},
            hit_at_k={k: float(np.mean(v)) for k, v in hit_scores.items() if v},
            map_score=float(np.mean(ap_scores)) if ap_scores else 0.0,
            median_rank=float(np.median(ranks)) if ranks else 0.0,
            mean_rank=float(np.mean(ranks)) if ranks else 0.0,
            ndcg_at_k={k: float(np.mean(v)) for k, v in ndcg_scores.items() if v},
            auc=float(np.mean(auc_scores)) if auc_scores else 0.0,
            num_queries=len(ap_scores),
            gallery_size=n_gallery,
        )


def evaluate_by_slice(
    evaluator: RetrievalEvaluator,
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    ground_truth: List[List[int]],
    slices: List[str],
) -> Dict[str, RetrievalMetrics]:
    """Compute overall metrics plus one RetrievalMetrics per slice.

    Args:
        evaluator: A RetrievalEvaluator.
        query_embs: (N, D) query embeddings.
        gallery_embs: (M, D) gallery embeddings.
        ground_truth: Per-query relevant gallery-index lists.
        slices: Per-query slice label (same length/order as query_embs).

    Returns:
        Dict mapping 'overall' and each slice name to RetrievalMetrics.
    """
    results = {'overall': evaluator.evaluate(query_embs, gallery_embs, ground_truth)}
    for name in sorted(set(slices)):
        idx = [i for i, s in enumerate(slices) if s == name]
        results[name] = evaluator.evaluate(
            query_embs[idx], gallery_embs, [ground_truth[i] for i in idx]
        )
    return results


def log_retrieval_metrics(results: Dict[str, RetrievalMetrics], prefix: str = "") -> None:
    """Log a compact table of retrieval metrics per slice."""
    header = f"{prefix}Retrieval metrics (slice: mAP / R@1 / R@5 / R@10 / nDCG@10 / n):"
    lines = [header]
    for name, m in results.items():
        lines.append(
            f"  {name:<20} "
            f"mAP={m.map_score:.4f} "
            f"R@1={m.recall_at_k.get(1, 0):.4f} "
            f"R@5={m.recall_at_k.get(5, 0):.4f} "
            f"R@10={m.recall_at_k.get(10, 0):.4f} "
            f"nDCG@10={m.ndcg_at_k.get(10, 0):.4f} "
            f"n={m.num_queries}"
        )
    logger.info("\n".join(lines))
