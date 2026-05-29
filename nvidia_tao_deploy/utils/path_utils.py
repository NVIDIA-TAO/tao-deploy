# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO common path utils used across all apps."""

import os


def expand_path(path):
    """This function takes in a path and returns the absolute path of that path after expanding the tilde (~) character to the user's home directory.

    This is done to prevent any path traversal vulnerability.

    Args:
        path (str): The path to expand and make absolute.

    Returns:
        str: The absolute path with expanded tilde.
    """
    return os.path.abspath(os.path.expanduser(path))
