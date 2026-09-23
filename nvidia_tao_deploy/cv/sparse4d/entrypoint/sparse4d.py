# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse4D module entrypoint; also available through model_agnostic."""

import argparse

from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra import command_line_parser, get_subtasks, launch
from nvidia_tao_deploy.cv.sparse4d import scripts


def main():
    """Dispatch Sparse4D gen_trt_engine and inference actions."""
    parser = argparse.ArgumentParser(description="TAO Deploy Sparse4D")
    subtasks = get_subtasks(scripts)
    args, unknown = command_line_parser(parser, subtasks)
    launch(vars(args), unknown, subtasks, network="sparse4d")


if __name__ == "__main__":
    main()
