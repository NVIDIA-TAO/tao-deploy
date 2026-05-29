# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confidence config class that holds parameters for postprocessing confidence."""

from nvidia_tao_deploy.cv.detectnet_v2.proto.postprocessing_config_pb2 import ConfidenceConfig as ConfidenceProto


def build_confidence_config(confidence_config):
    """Build ConfidenceConfig from a proto.

    Args:
        confidence_config: confidence_config proto message.

    Returns:
        ConfidenceConfig object.
    """
    return ConfidenceConfig(confidence_config.confidence_model_filename,
                            confidence_config.confidence_threshold)


def build_confidence_proto(confidence_config):
    """Build proto from ConfidenceConfig.

    Args:
        confidence_config: ConfidenceConfig object.

    Returns:
        confidence_config: confidence_config proto.
    """
    proto = ConfidenceProto()
    proto.confidence_model_filename = confidence_config.confidence_model_filename
    proto.confidence_threshold = confidence_config.confidence_threshold
    return proto


class ConfidenceConfig(object):
    """Hold the parameters for postprocessing confidence."""

    def __init__(self, confidence_model_filename, confidence_threshold):
        """Constructor.

        Args:
            confidence_model_filename (str): Absolute path to the confidence model hdf5.
            confidence_threshold (float): Confidence threshold value. Must be >= 0.
        Raises:
            ValueError: If the input arg is not within the accepted range.
        """
        if confidence_threshold < 0.0:
            raise ValueError("ConfidenceConfig.confidence_threshold must be >= 0")

        self.confidence_model_filename = confidence_model_filename
        self.confidence_threshold = confidence_threshold
