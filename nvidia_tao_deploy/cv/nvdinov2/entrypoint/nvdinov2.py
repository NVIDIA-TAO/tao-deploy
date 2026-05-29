# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO Deploy command line wrapper to invoke CLI scripts."""

import argparse
from nvidia_tao_deploy.cv.nvdinov2 import scripts

from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra import get_subtasks, launch, command_line_parser


def get_subtask_list():
    """Return the list of subtasks by inspecting the scripts package."""
    return get_subtasks(scripts)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "nvdinov2",
        add_help=True,
        description="Train Adapt Optimize Deploy entrypoint for NVDinoV2"
    )

    # Obtain the list of substasks
    subtasks = get_subtask_list()

    args, unknown_args = command_line_parser(parser, subtasks)

    # Parse the arguments and launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="nvdinov2")


if __name__ == '__main__':
    main()
