# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP retrieval metric unit tests (pure numpy, deterministic)."""

import numpy as np

from nvidia_tao_deploy.multimodal.video_clip.evaluation import (
    RetrievalEvaluator,
    compute_ap,
    evaluate_by_slice,
)


class TestComputeAP:
    def test_all_relevant_first(self):
        assert compute_ap(np.array([1.0, 1.0, 0.0, 0.0])) == 1.0

    def test_textbook_interleaved(self):
        # relevant at ranks 2 and 4 -> (1/2 + 2/4) / 2 = 0.5
        assert compute_ap(np.array([0.0, 1.0, 0.0, 1.0])) == 0.5

    def test_no_relevant_is_zero(self):
        assert compute_ap(np.array([0.0, 0.0, 0.0])) == 0.0


class TestRetrievalEvaluator:
    """Deterministic cases: gallery = identity basis, ties broken by index."""

    def setup_method(self):
        self.gallery = np.eye(4, dtype=np.float32)          # 4 clips, D=4
        self.ev = RetrievalEvaluator(k_values=(1, 2), compute_auc=False)

    def test_perfect_single_relevant(self):
        q = np.array([[1, 0, 0, 0]], dtype=np.float32)      # ranks clip0 first
        m = self.ev.evaluate(q, self.gallery, [[0]])
        assert m.map_score == 1.0
        assert m.recall_at_k[1] == 1.0

    def test_relevant_at_rank_two(self):
        # sims=[1,0,0,0]; stable argsort -> order [0,1,2,3]; relevant={1} at rank2
        q = np.array([[1, 0, 0, 0]], dtype=np.float32)
        m = self.ev.evaluate(q, self.gallery, [[1]])
        assert abs(m.map_score - 0.5) < 1e-6

    def test_multi_relevance_average(self):
        # query favours clips 0 and 1 equally; relevant={0,2}
        q = np.array([[1, 1, 0, 0]], dtype=np.float32) / np.sqrt(2)
        m = self.ev.evaluate(q, self.gallery, [[0, 2]])
        # order [0,1,2,3]; labels [1,0,1,0] -> (1/1 + 2/3)/2 = 0.8333
        assert abs(m.map_score - (1.0 + 2.0 / 3.0) / 2.0) < 1e-6

    def test_empty_relevant_skipped(self):
        q = np.array([[1, 0, 0, 0]], dtype=np.float32)
        m = self.ev.evaluate(q, self.gallery, [[]])
        assert m.num_queries == 0
        assert m.map_score == 0.0


class TestRetrievalAUC:
    """AUC needs scores and labels in the SAME order; ranked labels alone are not enough."""

    def setup_method(self):
        self.gallery = np.eye(4, dtype=np.float32)
        # sims = [1,2,3,4]/||q||, so the ranking order [3,2,1,0] is NOT the
        # gallery order -- a misaligned call gives the exact opposite answer.
        self.q = np.array([[1, 2, 3, 4]], dtype=np.float32)
        self.ev = RetrievalEvaluator(k_values=(1,), compute_auc=True)

    def test_only_lowest_scoring_clip_relevant_is_auc_zero(self):
        m = self.ev.evaluate(self.q, self.gallery, [[0]])
        assert abs(m.auc - 0.0) < 1e-6

    def test_only_highest_scoring_clip_relevant_is_auc_one(self):
        m = self.ev.evaluate(self.q, self.gallery, [[3]])
        assert abs(m.auc - 1.0) < 1e-6


class TestEvaluateBySlice:
    def test_slice_split_and_overall(self):
        gallery = np.eye(3, dtype=np.float32)
        q = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        gt = [[0], [1]]  # both perfect
        slices = ["anomaly", "specific"]
        ev = RetrievalEvaluator(k_values=(1,), compute_auc=False)
        res = evaluate_by_slice(ev, q, gallery, gt, slices)
        assert set(res) == {"overall", "anomaly", "specific"}
        assert res["overall"].map_score == 1.0
        assert res["anomaly"].num_queries == 1
        assert res["specific"].num_queries == 1
