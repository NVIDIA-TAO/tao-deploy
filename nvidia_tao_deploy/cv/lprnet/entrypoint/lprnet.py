# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO Deploy command line wrapper to invoke CLI scripts."""

import sys
from nvidia_tao_deploy.cv.common.entrypoint.entrypoint_proto import launch_job
import nvidia_tao_deploy.cv.lprnet.scripts


def main():
    """Function to launch the job."""
    launch_job(nvidia_tao_deploy.cv.lprnet.scripts, "lprnet", sys.argv[1:])


if __name__ == "__main__":
    main()
