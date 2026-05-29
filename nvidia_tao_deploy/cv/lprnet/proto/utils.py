# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO Deploy Config Base Utilities."""

import os
import logging

from google.protobuf.text_format import Merge as merge_text_proto
from nvidia_tao_deploy.cv.lprnet.proto.experiment_pb2 import Experiment

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def load_proto(config):
    """Load the experiment proto."""
    proto = Experiment()

    def _load_from_file(filename, pb2):
        if not os.path.exists(filename):
            raise IOError(f"Specfile not found at: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            merge_text_proto(f.read(), pb2)
    _load_from_file(config, proto)

    return proto
