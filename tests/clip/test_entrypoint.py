# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Entrypoint Unit Tests."""

from nvidia_tao_deploy.multimodal.clip.entrypoint.clip import get_subtask_list


class TestGetSubtaskList:
    """Tests for the CLIP entrypoint subtask discovery."""

    def test_returns_dict(self):
        subtasks = get_subtask_list()
        assert isinstance(subtasks, dict)

    def test_contains_gen_trt_engine(self):
        subtasks = get_subtask_list()
        assert "gen_trt_engine" in subtasks

    def test_contains_evaluate(self):
        subtasks = get_subtask_list()
        assert "evaluate" in subtasks

    def test_subtask_values_have_runner_path(self):
        subtasks = get_subtask_list()
        for details in subtasks.values():
            assert isinstance(details, dict)
            assert "runner_path" in details
            assert details["runner_path"].endswith(".py")
