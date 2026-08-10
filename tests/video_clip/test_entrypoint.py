# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP entrypoint unit test — subtask discovery."""

from nvidia_tao_deploy.multimodal.video_clip.entrypoint.video_clip import (
    get_subtask_list,
)


def test_subtasks_present():
    subtasks = get_subtask_list()
    for name in ("gen_trt_engine", "evaluate", "inference"):
        assert name in subtasks
