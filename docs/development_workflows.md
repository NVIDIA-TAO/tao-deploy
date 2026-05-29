# Development Workflows

## Source Setup

```sh
source scripts/envsetup.sh
```

This sets `NV_TAO_DEPLOY_TOP` and defines the `tao_deploy` shell function.

Build and install a wheel locally:

```sh
make build
make install
```

Build the L4T wheel path:

```sh
make build_l4t
```

Clean generated package artifacts:

```sh
make clean
```

## Run Inside The Base Container

Use `--` to split launcher arguments from the command that runs inside the
container:

```sh
tao_deploy --gpus all -- python3 -m pytest tests/core/test_dual_logging.py
```

Useful launcher options are documented in
[Container power users](container_power_users.md).

## Update The Base Image

The base image source is under `docker/`.

```sh
bash docker/build.sh --build --x86
bash docker/build.sh --build --arm
bash docker/build.sh --build --l4t
```

Push and record the new digest only after validation:

```sh
bash docker/build.sh --build --x86 --push
```

Update the matching platform digest in `docker/manifest.json`.

## Build A Release Image

The release image installs a wheel built from this repository.

```sh
source scripts/envsetup.sh
cd release/docker
./deploy.sh --build --wheel
```

Release image tags are assembled in `release/docker/deploy.sh`; package version
metadata comes from `release/python/version.py`.

## Update A Model Deploy Flow

For a Hydra-style model:

1. Update the dataclasses in `nvidia_tao_deploy/config/MODEL_NAME/`.
2. Update the matching templates in `nvidia_tao_deploy/cv/MODEL_NAME/specs/` or
   `nvidia_tao_deploy/multimodal/MODEL_NAME/specs/`.
3. Update `scripts/gen_trt_engine.py`, `scripts/inference.py`, or
   `scripts/evaluate.py`.
4. Update any model-specific builder, inferencer, dataloader, metric, or
   post-processing code.
5. Add or update focused tests under `tests/MODEL_NAME/`.

For a proto-style model, follow the existing `build_command_line_parser()` and
proto loader pattern in that model package.

## Update Generated Command Docs

When `setup.py` console scripts or model `scripts/` packages change:

```sh
python tools/update_docs_supported_commands.py
python tools/update_docs_supported_commands.py --check
```

The generated file is `docs/supported_commands.md`.

## Add A New Command

1. Add the model package under `nvidia_tao_deploy/cv/` or
   `nvidia_tao_deploy/multimodal/`.
2. Add an `entrypoint/` wrapper and `scripts/` package.
3. Add a `console_scripts` entry in `setup.py`.
4. Add config dataclasses under `nvidia_tao_deploy/config/` when the command
   supports default spec generation.
5. Regenerate supported-command docs.
6. Add tests under `tests/MODEL_NAME/`.

See [Deploy backend integration](deploy_backend_integration.md) for the
source-backed checklist.
