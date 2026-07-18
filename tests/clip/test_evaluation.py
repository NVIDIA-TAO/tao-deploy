# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Evaluation Module Unit Tests."""

import numpy as np
import pytest

from nvidia_tao_deploy.multimodal.clip.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_ndcg,
)
from nvidia_tao_deploy.multimodal.clip.evaluation.retrieval import (
    RetrievalEvaluator,
    RetrievalMetrics,
)


class TestComputeAP:
    """Tests for compute_ap function."""

    def test_perfect_ranking(self):
        """All positives ranked first should give AP=1.0."""
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert compute_ap(labels) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """All positives ranked last should give low AP."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        ap = compute_ap(labels)
        assert ap < 0.5

    def test_single_positive_first(self):
        """Single positive ranked first."""
        labels = np.array([1, 0, 0, 0, 0])
        assert compute_ap(labels) == pytest.approx(1.0)

    def test_single_positive_last(self):
        """Single positive ranked last."""
        labels = np.array([0, 0, 0, 0, 1])
        assert compute_ap(labels) == pytest.approx(0.2)

    def test_no_positives(self):
        """No positives should return 0."""
        labels = np.array([0, 0, 0, 0, 0])
        assert compute_ap(labels) == 0.0

    def test_all_positives(self):
        """All positives should return 1.0."""
        labels = np.array([1, 1, 1, 1, 1])
        assert compute_ap(labels) == pytest.approx(1.0)

    def test_mixed_ranking(self):
        """Test with mixed ranking."""
        labels = np.array([1, 0, 1, 0, 1, 0])
        expected = (1/1 + 2/3 + 3/5) / 3
        assert compute_ap(labels) == pytest.approx(expected)


class TestComputeNDCG:
    """Tests for compute_ndcg function."""

    def test_perfect_ranking_k1(self):
        """Perfect ranking at k=1."""
        labels = np.array([1, 1, 0, 0, 0])
        assert compute_ndcg(labels, k=1) == pytest.approx(1.0)

    def test_perfect_ranking_k5(self):
        """Perfect ranking at k=5."""
        labels = np.array([1, 1, 1, 0, 0])
        assert compute_ndcg(labels, k=5) == pytest.approx(1.0)

    def test_no_positives(self):
        """No positives should return 0."""
        labels = np.array([0, 0, 0, 0, 0])
        assert compute_ndcg(labels, k=5) == 0.0

    def test_k_larger_than_array(self):
        """k larger than array length should handle gracefully."""
        labels = np.array([1, 0, 1])
        ndcg = compute_ndcg(labels, k=10)
        assert 0.0 <= ndcg <= 1.0

    def test_positive_at_position_2(self):
        """Single positive at position 2."""
        labels = np.array([0, 1, 0, 0, 0])
        ndcg = compute_ndcg(labels, k=5)
        assert ndcg < 1.0
        assert ndcg > 0.0


class TestComputeAUC:
    """Tests for compute_auc function."""

    def test_perfect_separation(self):
        """Perfect separation should give AUC=1.0."""
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert compute_auc(scores, labels) == pytest.approx(1.0)

    def test_worst_separation(self):
        """Worst separation should give AUC=0.0."""
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert compute_auc(scores, labels) == pytest.approx(0.0)

    def test_random_separation(self):
        """Random separation should give AUC around 0.5."""
        np.random.seed(42)
        scores = np.random.rand(100)
        labels = np.array([1] * 50 + [0] * 50)
        auc = compute_auc(scores, labels)
        assert 0.3 < auc < 0.7

    def test_no_positives(self):
        """No positives should return 0.0."""
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0])
        assert compute_auc(scores, labels) == 0.0

    def test_no_negatives(self):
        """No negatives should return 1.0."""
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])
        assert compute_auc(scores, labels) == 1.0


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metrics = RetrievalMetrics()
        assert metrics.map_score == 0.0
        assert metrics.median_rank == 0.0
        assert metrics.mean_rank == 0.0
        assert metrics.auc == 0.0
        assert metrics.num_queries == 0
        assert metrics.gallery_size == 0
        assert len(metrics.recall_at_k) == 0
        assert len(metrics.ndcg_at_k) == 0

    def test_to_dict(self):
        """Test to_dict method."""
        metrics = RetrievalMetrics(
            recall_at_k={1: 0.5, 5: 0.8},
            map_score=0.75,
            median_rank=2.0,
            mean_rank=3.5,
            auc=0.9,
            num_queries=100,
            gallery_size=1000,
        )
        d = metrics.to_dict()
        assert d['mAP'] == 0.75
        assert d['median_rank'] == 2.0
        assert d['mean_rank'] == 3.5
        assert d['auc'] == 0.9
        assert d['recall@1'] == 0.5
        assert d['recall@5'] == 0.8

    def test_str_representation(self):
        """Test string representation."""
        metrics = RetrievalMetrics(
            recall_at_k={1: 0.5, 5: 0.8},
            map_score=0.75,
            median_rank=2.0,
            mean_rank=3.5,
            auc=0.9,
            num_queries=100,
            gallery_size=1000,
        )
        s = str(metrics)
        assert "mAP: 0.7500" in s
        assert "R@1: 0.5000" in s
        assert "MedR: 2.0" in s


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator class."""

    def test_init_default(self):
        """Test default initialization."""
        evaluator = RetrievalEvaluator()
        assert evaluator.k_values == (1, 5, 10)
        assert evaluator._compute_auc is True
        assert evaluator.batch_size == 1024

    def test_init_custom(self):
        """Test custom initialization."""
        evaluator = RetrievalEvaluator(
            k_values=(1, 3, 5),
            compute_auc=False,
            batch_size=512,
        )
        assert evaluator.k_values == (1, 3, 5)
        assert evaluator._compute_auc is False
        assert evaluator.batch_size == 512

    def test_evaluate_perfect_matching(self):
        """Test with perfect 1:1 matching."""
        evaluator = RetrievalEvaluator()

        n = 10
        d = 64
        embeddings = np.random.randn(n, d).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        gt = [[i] for i in range(n)]

        metrics = evaluator.evaluate(embeddings, embeddings, gt)

        assert metrics.recall_at_k[1] == pytest.approx(1.0)
        assert metrics.map_score == pytest.approx(1.0)
        assert metrics.median_rank == pytest.approx(1.0)

    def test_evaluate_bidirectional(self):
        """Test bidirectional evaluation."""
        evaluator = RetrievalEvaluator()

        n = 10
        d = 64
        image_embs = np.random.randn(n, d).astype(np.float32)
        text_embs = image_embs.copy()

        results = evaluator.evaluate_bidirectional(image_embs, text_embs)

        assert 'image_to_text' in results
        assert 'text_to_image' in results
        assert results['image_to_text'].recall_at_k[1] == pytest.approx(1.0)
        assert results['text_to_image'].recall_at_k[1] == pytest.approx(1.0)

    def test_evaluate_with_ground_truth_matrix(self):
        """Test with ground truth as matrix."""
        evaluator = RetrievalEvaluator(k_values=(1, 2))

        n = 5
        d = 32
        query_embs = np.random.randn(n, d).astype(np.float32)
        gallery_embs = np.random.randn(n, d).astype(np.float32)

        gt_matrix = np.eye(n)

        metrics = evaluator.evaluate(query_embs, gallery_embs, gt_matrix)

        assert metrics.num_queries == n
        assert metrics.gallery_size == n

    def test_evaluate_with_no_auc(self):
        """Test evaluation without AUC computation."""
        evaluator = RetrievalEvaluator(compute_auc=False)

        n = 5
        d = 32
        embeddings = np.random.randn(n, d).astype(np.float32)
        gt = [[i] for i in range(n)]

        metrics = evaluator.evaluate(embeddings, embeddings, gt)

        assert metrics.auc == 0.0
