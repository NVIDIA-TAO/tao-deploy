# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version string for the TAO Toolkit Deploy AI models/tasks."""

MAJOR = "6"
MINOR = "25"
PATCH = "10"
PRE_RELEASE = ''


# Getting the build number.
def get_build_info():
    """Get the build version number."""
    # required since setup.py runs a version string and global imports aren't executed.
    import os  # noqa pylint: disable=import-outside-toplevel
    build_file = "build.info"
    if not os.path.exists(build_file):
        raise FileNotFoundError("Build file doesn't exist.")
    patch = 0
    with open(build_file, 'r') as bfile:
        patch = bfile.read().strip()
    assert bfile.closed, "Build file wasn't closed properly."
    return patch


try:
    PATCH = get_build_info()
except FileNotFoundError:
    pass

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

# Version of the library.
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

# Version of the file format.
__format_version__ = 2

# Other package info.
__package_name__ = "nvidia-tao-deploy"
__description__ = "NVIDIA's package for deploying models from TAO Toolkit."
__keywords__ = "nvidia, tao, tensorrt"

__contact_names__ = "Sean Cha"
__contact_emails__ = "scha@nvidia.com"

__license__ = "NVIDIA Proprietary Software"