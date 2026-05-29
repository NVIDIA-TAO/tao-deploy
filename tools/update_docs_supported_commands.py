# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the supported TAO Deploy command documentation."""

from __future__ import annotations

import argparse
import ast
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = REPO_ROOT / "setup.py"
TARGET = REPO_ROOT / "docs" / "supported_commands.md"
BEGIN = "<!-- BEGIN GENERATED: supported-commands -->"
END = "<!-- END GENERATED: supported-commands -->"


@dataclass(frozen=True)
class Command:
    """A console command discovered from setup.py."""

    name: str
    target: str
    domain: str
    implementation: str
    subtasks: str


def _find_setup_call(tree: ast.AST) -> ast.Call:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "setup":
            return node
        if isinstance(func, ast.Name) and func.id == "setup":
            return node
    raise ValueError("Could not find setup() call in setup.py")


def _console_script_specs() -> list[str]:
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"), filename=str(SETUP_PY))
    setup_call = _find_setup_call(tree)
    for keyword in setup_call.keywords:
        if keyword.arg == "entry_points":
            entry_points = ast.literal_eval(keyword.value)
            return list(entry_points["console_scripts"])
    raise ValueError("Could not find entry_points['console_scripts'] in setup.py")


def _domain_for(module_path: str) -> str:
    parts = module_path.split(".")
    if "multimodal" in parts:
        return "multimodal"
    if "cv" in parts:
        return "cv"
    return "common"


def _implementation_root(module_path: str) -> str:
    if ".entrypoint." in module_path:
        return module_path.split(".entrypoint.", maxsplit=1)[0]
    return module_path.rsplit(".", maxsplit=1)[0]


def _module_source(module_path: str) -> Path:
    return REPO_ROOT / Path(*module_path.split(".")).with_suffix(".py")


def _scripts_dir(implementation: str) -> Path:
    return REPO_ROOT / Path(*implementation.split(".")) / "scripts"


def _uses_hydra_entrypoint(module_path: str) -> bool:
    source_path = _module_source(module_path)
    if not source_path.exists():
        return False
    source = source_path.read_text(encoding="utf-8")
    return "entrypoint_hydra" in source or "get_subtasks" in source


def _discover_subtasks(command_name: str, module_path: str, implementation: str) -> str:
    if module_path.endswith("entrypoint_agnostic"):
        return "model-selected from spec; default_specs requires -m/--model_name"

    scripts_dir = _scripts_dir(implementation)
    subtasks = []
    if scripts_dir.exists():
        subtasks = sorted(
            path.stem
            for path in scripts_dir.glob("*.py")
            if path.stem != "__init__"
        )

    if _uses_hydra_entrypoint(module_path):
        subtasks.append("default_specs")

    if not subtasks:
        return f"see {command_name} entrypoint"
    return ", ".join(subtasks)


def discover_commands() -> list[Command]:
    commands = []
    for spec in _console_script_specs():
        name, target = spec.split("=", maxsplit=1)
        module_path = target.split(":", maxsplit=1)[0]
        implementation = _implementation_root(module_path)
        commands.append(
            Command(
                name=name,
                target=target,
                domain=_domain_for(module_path),
                implementation=implementation,
                subtasks=_discover_subtasks(name, module_path, implementation),
            )
        )
    return sorted(commands, key=lambda command: command.name)


def _render_table(commands: Iterable[Command]) -> list[str]:
    lines = [
        BEGIN,
        "",
        "_Source: `setup.py` console scripts plus each implementation's `scripts/` package. "
        "Regenerate with `python tools/update_docs_supported_commands.py`._",
        "",
        "| Command | Domain | Implementation | Discovered subtasks |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for command in commands:
        lines.append(
            f"| `{command.name}` | {command.domain} | "
            f"`{command.implementation}` | {command.subtasks} |"
        )
    lines.extend(["", END])
    return lines


def render_document(commands: Iterable[Command]) -> str:
    lines = [
        "# Supported Deploy Commands",
        "",
        "This file is generated. Edit `setup.py`, model entrypoints, or model "
        "`scripts/` packages, then regenerate this file.",
        "",
    ]
    lines.extend(_render_table(commands))
    lines.append("")
    return "\n".join(lines)


def _replace_generated_block(existing: str, generated_block: str) -> str:
    if BEGIN not in existing or END not in existing:
        raise ValueError(f"{TARGET} exists but does not contain generated markers")

    before, rest = existing.split(BEGIN, maxsplit=1)
    _, after = rest.split(END, maxsplit=1)
    return before + generated_block + after


def desired_content() -> str:
    commands = discover_commands()
    generated_block = "\n".join(_render_table(commands))
    if TARGET.exists():
        return _replace_generated_block(TARGET.read_text(encoding="utf-8"), generated_block)
    return render_document(commands)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated docs are stale.")
    args = parser.parse_args()

    desired = desired_content()
    existing = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""

    if args.check:
        if existing != desired:
            diff = difflib.unified_diff(
                existing.splitlines(),
                desired.splitlines(),
                fromfile=str(TARGET),
                tofile=f"{TARGET} (generated)",
                lineterm="",
            )
            print("\n".join(diff))
            return 1
        print(f"{TARGET.relative_to(REPO_ROOT)} is up to date.")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(desired, encoding="utf-8")
    print(f"Updated {TARGET.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
