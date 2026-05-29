# Testing And Debugging

## Static Checks

GitLab runs `static_tests` from `.gitlab-ci.yml`. The job first checks generated
command documentation and then runs the static test launcher:

```sh
python tools/update_docs_supported_commands.py --check
python ci/run_static_tests.py
```

`ci/run_static_tests.py` runs `pylint`, `pydocstyle`, and `flake8` against the
configured source modules. Outside CI it launches those checks inside the base
container from `docker/manifest.json`.

## Functional Tests

Run all functional tests:

```sh
python ci/run_functional_tests.py
```

Run a model subset:

```sh
python ci/run_functional_tests.py --module clip
```

Run a focused pytest directly when the environment already has dependencies:

```sh
pytest --color=yes -v tests/core/test_dual_logging.py
pytest --color=yes -v tests/clip/test_evaluation.py
```

## Test Map

| Area | Examples |
| :--- | :--- |
| Core utilities | `tests/core/test_decrypt.py`, `tests/core/test_dual_logging.py` |
| TensorRT engine flows | `tests/*/test_engine.py`, `tests/*/test_engine_builder.py` |
| Dataloaders | `tests/*/test_dataloader.py`, `tests/depth_net/test_loader.py` |
| Inference helpers | `tests/clip/test_inferencer.py`, model-specific inferencer tests |
| Metrics and evaluation | `tests/clip/test_evaluation.py`, model `evaluate` tests |

Many engine tests require GPUs, TensorRT, model artifacts, and data mounted from
private paths such as `/home/scratch.metropolis2`. Prefer targeted tests for
small source changes and state clearly when those heavier checks were not run.

## Common Failures

| Symptom | Likely source | Check |
| :--- | :--- | :--- |
| `git status` fails on LFS clean filter | Read-only `.git/lfs/tmp` | Use `git -c filter.lfs.process= -c filter.lfs.required=false status --short --branch`. |
| Base image pull fails | NGC login or network access | `docker login nvcr.io` and inspect `docker/manifest.json`. |
| Static tests run differently locally | `ci/run_static_tests.py` launches Docker outside CI | Check `CI_PROJECT_DIR` and `NV_TAO_DEPLOY_TOP`. |
| Import errors for `nvidia_tao_core` | Local container does not have the core wheel | `ci/run_functional_tests.py` adds `/workspace/tao-deploy/tao-core` to `PYTHONPATH`. |
| TensorRT or CUDA errors | Host/container mismatch or missing GPU | Check `--gpus`, driver, CUDA, TensorRT, and `nvidia-container-toolkit`. |
| Generated docs check fails | `setup.py` or `scripts/` changed | Run `python tools/update_docs_supported_commands.py`. |

## Documentation Checks

For docs-only changes:

```sh
python tools/update_docs_supported_commands.py --check
python -m py_compile tools/update_docs_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml .gitlab-ci.yml
```

GPU, Docker, TensorRT, dataset, and private-checkpoint tests are normally out of
scope for documentation-only updates unless the docs changed runnable behavior.
