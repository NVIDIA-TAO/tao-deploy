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

"""Retrieval evaluation for CLIP TensorRT models.

This module provides text-to-image and image-to-text retrieval evaluation
using CuPy for GPU-accelerated similarity computation.

Classes:
    RetrievalMetrics: Container for retrieval evaluation results.
    RetrievalEvaluator: Main evaluator class for retrieval tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging
from nvidia_tao_deploy.multimodal.clip.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_ndcg,
)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics.

    Attributes:
        recall_at_k: Dictionary mapping k values to recall scores.
        map_score: Mean Average Precision.
        median_rank: Median rank of first correct match.
        mean_rank: Mean rank of first correct match.
        ndcg_at_k: Dictionary mapping k values to NDCG scores.
        auc: Area Under ROC Curve (separability).
        num_queries: Number of queries evaluated.
        gallery_size: Size of the retrieval gallery.
    """

    recall_at_k: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    median_rank: float = 0.0
    mean_rank: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    auc: float = 0.0
    num_queries: int = 0
    gallery_size: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to flat dictionary."""
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
        for k, v in self.ndcg_at_k.items():
            result[f'ndcg@{k}'] = v
        return result

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = [
            f"Retrieval Metrics (queries={self.num_queries}, "
            f"gallery={self.gallery_size}):",
            f"  mAP: {self.map_score:.4f}",
        ]
        for k in sorted(self.recall_at_k.keys()):
            lines.append(f"  R@{k}: {self.recall_at_k[k]:.4f}")
        lines.append(f"  MedR: {self.median_rank:.1f}")
        lines.append(f"  MeanR: {self.mean_rank:.1f}")
        if self.auc > 0:
            lines.append(f"  AUC: {self.auc:.4f}")
        return '\n'.join(lines)


class RetrievalEvaluator:
    """Evaluator for text-to-image and image-to-text retrieval.

    Uses NumPy for similarity computation.

    Args:
        k_values: Tuple of k values for Recall@k and NDCG@k.
        compute_auc: Whether to compute AUC metric.
        batch_size: Batch size for batched similarity computation.
    """

    def __init__(
        self,
        k_values: Tuple[int, ...] = (1, 5, 10),
        compute_auc: bool = True,
        batch_size: int = 1024,
        **kwargs
    ):
        """Initialize the retrieval evaluator."""
        self.k_values = k_values
        self._compute_auc = compute_auc
        self.batch_size = batch_size

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def _compute_similarity_matrix(
        self,
        query_embs: np.ndarray,
        gallery_embs: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity matrix.

        Args:
            query_embs: Query embeddings of shape (N, D).
            gallery_embs: Gallery embeddings of shape (M, D).

        Returns:
            Similarity matrix of shape (N, M).
        """
        query_embs = self._normalize(query_embs)
        gallery_embs = self._normalize(gallery_embs)

        n_queries = query_embs.shape[0]
        similarity_rows = []

        for start_idx in range(0, n_queries, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_queries)
            query_batch = query_embs[start_idx:end_idx]
            sim_batch = np.dot(query_batch, gallery_embs.T)
            similarity_rows.append(sim_batch)

        return np.concatenate(similarity_rows, axis=0)

    def evaluate(
        self,
        query_embs: np.ndarray,
        gallery_embs: np.ndarray,
        ground_truth: Union[np.ndarray, List[List[int]]]
    ) -> RetrievalMetrics:
        """Compute retrieval metrics.

        Args:
            query_embs: Query embeddings of shape (N, D).
            gallery_embs: Gallery embeddings of shape (M, D).
            ground_truth: Ground truth matches. Either:
                - Binary matrix (N, M) where gt[i,j]=1 if j relevant to i
                - List of lists where gt[i] contains relevant indices

        Returns:
            RetrievalMetrics object with all computed metrics.
        """
        n_queries = query_embs.shape[0]
        n_gallery = gallery_embs.shape[0]

        logging.info(
            "Computing similarity matrix (%d x %d)...",
            n_queries, n_gallery
        )

        similarity_matrix = self._compute_similarity_matrix(
            query_embs, gallery_embs
        )

        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            gt_matrix = np.asarray(ground_truth, dtype=np.float32)
            gt_list = [
                np.where(gt_matrix[i] == 1)[0].tolist()
                for i in range(n_queries)
            ]

        ap_scores = []
        ranks = []
        recall_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        auc_scores = []

        desc = "Computing retrieval metrics"
        for i in tqdm(range(n_queries), desc=desc, leave=False):
            sims = similarity_matrix[i]
            relevant_indices = set(gt_list[i])
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

            for k in self.k_values:
                ndcg_scores[k].append(compute_ndcg(sorted_labels, k))

            if self._compute_auc:
                auc_scores.append(compute_auc(sims, sorted_labels))

        return RetrievalMetrics(
            recall_at_k={k: np.mean(v) for k, v in recall_scores.items() if v},
            map_score=np.mean(ap_scores) if ap_scores else 0.0,
            median_rank=np.median(ranks) if ranks else 0.0,
            mean_rank=np.mean(ranks) if ranks else 0.0,
            ndcg_at_k={k: np.mean(v) for k, v in ndcg_scores.items() if v},
            auc=np.mean(auc_scores) if auc_scores else 0.0,
            num_queries=len(ap_scores),
            gallery_size=n_gallery,
        )

    def evaluate_bidirectional(
        self,
        image_embs: np.ndarray,
        text_embs: np.ndarray,
        image_to_text_gt: Optional[Union[np.ndarray, List[List[int]]]] = None,
        text_to_image_gt: Optional[Union[np.ndarray, List[List[int]]]] = None,
    ) -> Dict[str, RetrievalMetrics]:
        """Evaluate bidirectional retrieval (image-to-text and text-to-image).

        For paired datasets where image i matches text i, ground truth is
        automatically inferred if not provided.

        Args:
            image_embs: Image embeddings of shape (N, D).
            text_embs: Text embeddings of shape (M, D).
            image_to_text_gt: Ground truth for image queries.
            text_to_image_gt: Ground truth for text queries.

        Returns:
            Dictionary with 'image_to_text' and 'text_to_image' metrics.
        """
        n_images = image_embs.shape[0]
        n_texts = text_embs.shape[0]

        if image_to_text_gt is None and n_images == n_texts:
            image_to_text_gt = [[i] for i in range(n_images)]
        if text_to_image_gt is None and n_images == n_texts:
            text_to_image_gt = [[i] for i in range(n_texts)]

        results = {}

        if image_to_text_gt is not None:
            logging.info("Evaluating image-to-text retrieval...")
            results['image_to_text'] = self.evaluate(
                image_embs, text_embs, image_to_text_gt
            )

        if text_to_image_gt is not None:
            logging.info("Evaluating text-to-image retrieval...")
            results['text_to_image'] = self.evaluate(
                text_embs, image_embs, text_to_image_gt
            )

        return results


def log_retrieval_metrics(
    metrics: Union[RetrievalMetrics, Dict[str, RetrievalMetrics]],
    prefix: str = ""
) -> None:
    """Log retrieval metrics in a formatted table.

    Args:
        metrics: Single RetrievalMetrics or dict with direction keys.
        prefix: Optional prefix for log messages.
    """
    if isinstance(metrics, RetrievalMetrics):
        metrics = {'retrieval': metrics}

    table_data = []
    headers = [
        'Direction', 'mAP', 'R@1', 'R@5', 'R@10', 'MedR', 'MeanR', 'AUC'
    ]

    for direction, m in metrics.items():
        row = [
            direction,
            f"{m.map_score:.4f}",
            f"{m.recall_at_k.get(1, 0):.4f}",
            f"{m.recall_at_k.get(5, 0):.4f}",
            f"{m.recall_at_k.get(10, 0):.4f}",
            f"{m.median_rank:.1f}",
            f"{m.mean_rank:.1f}",
            f"{m.auc:.4f}",
        ]
        table_data.append(row)

    col_widths = [max(len(str(row[i])) for row in [headers] + table_data)
                  for i in range(len(headers))]

    def format_row(row):
        return "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))

    lines = [format_row(headers), "-" * (sum(col_widths) + 2 * (len(headers) - 1))]
    for row in table_data:
        lines.append(format_row(row))

    table = "\n".join(lines)
    log_msg = f"{prefix}Retrieval Metrics:\n{table}" if prefix else f"Retrieval Metrics:\n{table}"
    logging.info(log_msg)
