# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action-boundary failures reach the real status logger without a GPU."""

import importlib
import json

import onnx
import pytest
from omegaconf import OmegaConf

from nvidia_tao_deploy.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.logging import status_logging, tlt_logging


pytestmark = pytest.mark.sparse4d


@pytest.fixture(autouse=True)
def isolate_status_logger(monkeypatch):
    """Keep actual file logging while disabling external callbacks and leaks."""
    previous = status_logging.get_status_logger()
    monkeypatch.setattr(status_logging, "_STATUS_LOGGER", previous)
    monkeypatch.setattr(status_logging, "status_callback", lambda _: None)
    monkeypatch.setattr(tlt_logging.logger, "handlers", tlt_logging.logger.handlers.copy())
    yield
    current = status_logging.get_status_logger()
    if current is not previous and hasattr(current, "l_file"):
        current.l_file.close()


@pytest.fixture(params=[("gen_trt_engine", "build_engine"), ("inference", "run_inference")])
def action(request, tmp_path):
    """Use the production Hydra passthrough and monitor_status wrappers."""
    name, runner = request.param
    module = importlib.import_module(f"nvidia_tao_deploy.cv.sparse4d.scripts.{name}")
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg[name].results_dir = str(tmp_path)
    return module, runner, cfg


def read_status(tmp_path):
    """Read the real JSON-lines log produced by monitor_status."""
    return [json.loads(line) for line in (tmp_path / "status.json").read_text().splitlines()]


@pytest.mark.parametrize("error_kind", ["runtime", "onnx", "pycuda", "other"])
def test_action_failure_records_failure_and_preserves_cause(action, error_kind, monkeypatch, tmp_path):
    """Native exception classes are not necessarily RuntimeError subclasses."""
    if error_kind == "pycuda":
        # Importing the driver exception type does not create a CUDA context.
        error_type = pytest.importorskip("pycuda.driver").Error
    elif error_kind == "onnx":
        error_type = onnx.checker.ValidationError
    else:
        error_type = RuntimeError if error_kind == "runtime" else Exception
    error = error_type(f"injected {error_kind} failure")
    module, runner, cfg = action

    def fail(_):
        raise error

    monkeypatch.setattr(module, runner, fail)
    with pytest.raises(ValueError, match=f"injected {error_kind} failure") as raised:
        module.main(cfg)
    assert raised.value.__cause__ is error
    records = read_status(tmp_path)
    assert records[0]["status"] == "STARTED"
    assert records[-1]["status"] == "FAILURE"
    assert f"injected {error_kind} failure" in records[-1]["message"]
    assert not any(record["status"] == "SUCCESS" for record in records)


def test_successful_action_records_success(action, monkeypatch, tmp_path):
    """The error adapter does not change the normal action lifecycle."""
    module, runner, cfg = action
    observed = []
    monkeypatch.setattr(module, runner, lambda received: observed.append(received))
    module.main(cfg)
    assert observed == [cfg]
    assert read_status(tmp_path)[-1]["status"] == "SUCCESS"


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_action_adapter_does_not_translate_process_control(action, error_type, monkeypatch):
    """Only ordinary exceptions are translated at the action boundary."""
    module, runner, cfg = action
    error = error_type("stop")

    def fail(_):
        raise error

    monkeypatch.setattr(module, runner, fail)
    # Bypass the shared monitor, whose interruption policy is outside this adapter.
    with pytest.raises(error_type) as raised:
        module.main.__wrapped__.__wrapped__(cfg)
    assert raised.value is error


def test_invalid_onnx_records_failure_before_gpu_initialization(tmp_path):
    """A generated invalid graph exercises the actual ONNX checker and builder."""
    from nvidia_tao_deploy.cv.sparse4d.scripts import gen_trt_engine

    model = onnx.helper.make_model(onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["missing_input"], ["output"])],
        "invalid", [],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    ))
    path = tmp_path / "invalid.onnx"
    onnx.save(model, path)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.gen_trt_engine.results_dir = str(tmp_path)
    cfg.gen_trt_engine.onnx_file = str(path)
    cfg.gen_trt_engine.save_engine = str(tmp_path / "model.engine")
    with pytest.raises(ValueError) as raised:
        gen_trt_engine.main(cfg)
    assert isinstance(raised.value.__cause__, onnx.checker.ValidationError)
    assert read_status(tmp_path)[-1]["status"] == "FAILURE"
    assert not (tmp_path / "model.engine").exists()
