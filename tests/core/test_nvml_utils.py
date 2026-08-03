# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NVML telemetry utilities."""

import json

import pynvml
import pytest

from nvidia_tao_deploy.cv.common.telemetry.nvml_utils import GPUDevice

DEVICE_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_CONFIG_NAME = "NVIDIA-A100-SXM4-80GB"


def _make_device(device_name, pci_bus_id="00000000:07:00.0"):
    return GPUDevice(
        pci_bus_id=pci_bus_id,
        device_name=device_name,
        device_brand=pynvml.NVML_BRAND_NVIDIA,
        memory=85899345920,
        cuda_compute_capability=(8, 0),
    )


@pytest.mark.core
@pytest.mark.parametrize(
    "device_name",
    [DEVICE_NAME.encode(), DEVICE_NAME],
    ids=["bytes_name", "str_name"],
)
def test_get_config_name(device_name):
    """get_config() must handle both bytes (old pynvml) and str (new pynvml) names."""
    device = _make_device(device_name)
    config = device.get_config()
    assert config["name"] == EXPECTED_CONFIG_NAME


@pytest.mark.core
@pytest.mark.parametrize(
    "device_name",
    [DEVICE_NAME.encode(), DEVICE_NAME],
    ids=["bytes_name", "str_name"],
)
@pytest.mark.parametrize(
    "pci_bus_id",
    [b"00000000:07:00.0", "00000000:07:00.0"],
    ids=["bytes_bus_id", "str_bus_id"],
)
def test_config_is_json_serializable(device_name, pci_bus_id):
    """No bytes may leak into the config dict: str(device) json.dumps the config."""
    device = _make_device(device_name, pci_bus_id=pci_bus_id)
    config = device.get_config()
    assert json.dumps(config)
    assert str(device)
    assert config["pci_bus_id"] == "00000000:07:00.0"
