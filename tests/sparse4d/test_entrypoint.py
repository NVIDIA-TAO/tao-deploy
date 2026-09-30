# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the installed command registration, discovery and documentation aligned."""

import ast
from pathlib import Path
import runpy
import sys

import pytest
from setuptools import find_packages

from nvidia_tao_deploy.cv.sparse4d.entrypoint import sparse4d


pytestmark = pytest.mark.sparse4d
REPO_ROOT = Path(__file__).resolve().parents[2]
SUBTASKS = {"gen_trt_engine", "inference", "default_specs"}


@pytest.fixture
def command_docs():
    """Use the production setup.py parser without executing package setup."""
    return runpy.run_path(str(REPO_ROOT / "tools/update_docs_supported_commands.py"))


def test_console_script_registration_and_generated_documentation(command_docs):
    """The packaged console script and generated command table cannot drift."""
    commands = [command for command in command_docs["discover_commands"]() if command.name == "sparse4d"]
    assert len(commands) == 1
    command = commands[0]
    assert command.target == "nvidia_tao_deploy.cv.sparse4d.entrypoint.sparse4d:main"
    assert set(command.subtasks.split(", ")) == SUBTASKS
    assert command_docs["TARGET"].read_text() == command_docs["desired_content"]()


def test_spec_templates_use_generic_package_data(command_docs):
    """Discover the specs package and include YAML with the shared data rule."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    setup_call = command_docs["_find_setup_call"](tree)
    package_data = ast.literal_eval(next(
        keyword.value for keyword in setup_call.keywords if keyword.arg == "package_data"
    ))
    assert "cv.sparse4d.specs" in find_packages(str(REPO_ROOT / "nvidia_tao_deploy"))
    assert not any(name.startswith("nvidia_tao_deploy.cv.sparse4d") for name in package_data)
    package = REPO_ROOT / "nvidia_tao_deploy/cv/sparse4d/specs"
    patterns = package_data.get("", [])
    included = {path.relative_to(package).as_posix() for pattern in patterns for path in package.glob(pattern)}
    assert {"gen_trt_engine.yaml", "inference.yaml"} <= included


def test_console_help_discovers_all_actions(monkeypatch, capsys):
    """Help resolves the real action modules without a CUDA context."""
    monkeypatch.setattr(sys, "argv", ["sparse4d", "--help"])
    with pytest.raises(SystemExit) as raised:
        sparse4d.main()
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "TAO Deploy Sparse4D" in output
    for task in SUBTASKS:
        assert task in output


@pytest.mark.parametrize("task", sorted(SUBTASKS))
def test_console_dispatch_uses_sparse4d_network(task, monkeypatch, tmp_path):
    """The registered entrypoint forwards each action to the shared launcher."""
    args = ["sparse4d", task]
    spec = tmp_path / "experiment.yaml"
    if task != "default_specs":
        args += ["-e", str(spec)]
    args.append("results_dir=/tmp/sparse4d-command-test")
    monkeypatch.setattr(sys, "argv", args)
    calls = []

    def capture_launch(parsed, overrides, subtasks, network):
        calls.append((parsed, overrides, subtasks, network))

    monkeypatch.setattr(sparse4d, "launch", capture_launch)
    sparse4d.main()
    assert len(calls) == 1
    parsed, overrides, subtasks, network = calls[0]
    assert parsed["subtask"] == task
    assert parsed["experiment_spec_file"] == (None if task == "default_specs" else str(spec))
    assert overrides == ["results_dir=/tmp/sparse4d-command-test"]
    assert set(subtasks) == SUBTASKS
    assert network == "sparse4d"
