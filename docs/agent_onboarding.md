# Agent Onboarding

This guide is the fast path for coding agents and new maintainers. It favors
source-backed orientation over assumptions.

## First Audit

Run these before editing:

```sh
pwd
git remote -v
git branch -vv
git -c filter.lfs.process= -c filter.lfs.required=false status --short --branch
find . -maxdepth 2 -type d
sed -n '1,220p' README.md
sed -n '1,220p' .gitlab-ci.yml
rg -n "console_scripts|entry_points" setup.py
find runner docker release tests ci -maxdepth 2 -type f
```

Use the LFS-disabled `git status` form when the local checkout cannot write to
`.git/lfs/tmp`.

## Runtime Trace

When a task touches command behavior, gather the real flow with:

```sh
rg -n "ArgumentParser|docker/manifest|manifest.json|docker run|--gpus|--tag|--run_as_user" runner scripts docker release
rg -n "hydra_runner|default_specs|get_subtasks|entrypoint|console_scripts" nvidia_tao_deploy setup.py
rg -n "pytest|run_static|pre-commit|flake8|pylint" .gitlab-ci.yml ci tests
```

Then inspect the specific model package under `nvidia_tao_deploy/cv/` or
`nvidia_tao_deploy/multimodal/`.

## Mental Model

TAO Deploy is a deployment runtime repository, not a training repository. Most
commands accept an exported model, build or load a TensorRT engine, run
inference or evaluation, and write outputs under a results directory.

The important layers are:

| Layer | What to inspect |
| :--- | :--- |
| Launcher | `scripts/envsetup.sh`, `runner/tao_deploy.py`, `docker/manifest.json` |
| Installed commands | `setup.py` console scripts |
| Model dispatch | `MODEL_NAME/entrypoint/*.py` and `MODEL_NAME/scripts/*.py` |
| Config | `nvidia_tao_deploy/config/MODEL_NAME/` and `MODEL_NAME/specs/` |
| TensorRT runtime | `nvidia_tao_deploy/engine/`, `nvidia_tao_deploy/inferencer/`, model-specific builders |
| Validation | `ci/`, `tests/`, `.gitlab-ci.yml` |

## Dirty Worktree Safety

Treat untracked files and unrelated edits as user-owned. Check status before and
after edits, and do not remove local files unless the task explicitly requires
it.

Generated documentation has one source of truth:

```sh
python tools/update_docs_supported_commands.py
python tools/update_docs_supported_commands.py --check
```

If the check fails, regenerate the file rather than hand-editing the generated
block.

## Targeted Checks

For documentation-only changes:

```sh
python tools/update_docs_supported_commands.py --check
python -m py_compile tools/update_docs_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml .gitlab-ci.yml
```

For code changes, add the nearest static or pytest check from
[Testing and debugging](testing_and_debugging.md). GPU, Docker, TensorRT,
private checkpoint, and dataset-heavy tests should be called out explicitly
when they are outside the change's blast radius.
