# Sparse4D MSDA TensorRT plugin

This directory builds `libmsda_fp16.so`, which registers **MSDA, version 1,
empty plugin namespace** for the `nv::MSDA` nodes exported by TAO Sparse4D.
The library supports both FP32 and FP16 despite its filename. It is an
experimental source build for TensorRT 10.x. The TAO 7.3 Deploy image contains
the build tools and backend, but does not contain this compiled library.

## Build

From the root of this checkout, build with the same image used for engine
creation and inference:

```bash
docker run --rm --runtime=runc --network=none \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/source" -w /source/nvidia_tao_deploy/cv/sparse4d/msda_fp16 \
  --entrypoint /bin/bash nvcr.io/nvidia/tao/tao-toolkit:7.3.0-deploy \
  -lc 'make -j2 && make test-host'
sha256sum nvidia_tao_deploy/cv/sparse4d/msda_fp16/build/libmsda_fp16.so
```

Compilation and `test-host` do not require a GPU. The 7.3.0 Deploy image provides
CUDA 13.2 and TensorRT 10.16.0.72. The tested image manifest digest is
`sha256:2b5093e24116b4846b287a96d1ae9477feffcc55673b06804bc0045523b3b475`.
Use the image by digest when reproducing a build.

The default build emits native code for the listed architectures supported by
`nvcc`. To build only for a selected GPU, use, for example,
`make -j2 CUDA_ARCHS=120` for compute capability 12.0. This is an architecture
selection, not a statement of validation on every GPU. Run `make clean` before
changing the compiler, architecture list, or build flags. You can override
`CUDA_HOME`, `NVCC`, `TENSORRT_INCLUDE`, `TENSORRT_LIB`, and `BUILD_DIR` for another
TensorRT 10 installation. Keep CUDA/TensorRT and the GPU architecture compatible
with the runtime that will load the library. No PyTorch or Python extension
build is needed.

The result is `nvidia_tao_deploy/cv/sparse4d/msda_fp16/build/libmsda_fp16.so`. Mount that directory
into the runtime container and set `plugin.path` to the **container's absolute
path** to the library. Set `plugin.sha256` to the SHA-256 printed above in both
the engine-build and inference specifications. TAO Deploy loads it using these
fields; `LD_PRELOAD` is not required. Rebuild engines after changing the plugin.
For model export, prepared frames, engine generation, and inference, refer to
the canonical TAO documentation, **Sparse4D with TAO Deploy**.

## Contract and validation

The production tensor layout is:

| Input | Shape | Type |
| --- | --- | --- |
| Features | `(1, F, 256)` | FP32 or FP16 |
| Spatial shapes | `(C, 4, 2)` in height/width order | INT32 or INT64 |
| Scale start offsets | `(C, 4)` in the flattened feature tensor | Same integer type |
| Sampling locations | `(1, 900, 13, C, 2)` in x/y order | Same type as features |
| Weights | `(1, 900, 13, C, 4, 8)` | Same type as features |
| Output | `(1, 900, 256)` | Same type as features |

All tensors use linear storage. Shape values and offsets must describe valid
regions in the flattened feature tensor; use the constants from the trusted TAO
export. The wrapper rejects unsupported tensor dimensions and types before
launching a kernel. Sampling is bilinear with zero padding, and only locations
with **both coordinates strictly inside `(0, 1)`** contribute. Boundary values,
NaN, and infinity contribute zero. Feature and weight arithmetic is unchanged
from the maintained CUDA implementation; arbitrary nonfinite features or
weights are not sanitized.

`make test-host` checks registry identity, serialization and cloning, malformed
serialized data and creator fields, format negotiation, and layout rejection.
The GPU suite additionally exercises FP32/FP16 boundary/NaN/Inf behavior and
compares nonuniform bilinear samples against an independent NumPy reference,
with four and six cameras and both INT32 and INT64 indices:

```bash
# Run inside a compatible GPU-enabled TAO Deploy container with this checkout
# mounted at /source and the library mounted at /plugins.
cd /source
export PYTHONPATH=/source
export SPARSE4D_PLUGIN_PATH=/plugins/libmsda_fp16.so
export SPARSE4D_PLUGIN_SHA256="$(sha256sum "$SPARSE4D_PLUGIN_PATH" | cut -d' ' -f1)"
python3 -m pytest tests/sparse4d/test_plugin_gpu.py \
  tests/sparse4d/test_msda_reference.py -q
```

The hash records the trusted local artifact; it does not establish its origin.
Plugin tests do not establish model accuracy. Before deployment, compare full
engine outputs with the TAO PyTorch export wrapper on identical inputs and
validate task accuracy on representative data. GPU conformance and full-model
validation are required release checks, separate from successful compilation.

## Attribution

The CUDA implementation and TensorRT wrapper come from NVIDIA's maintained
Sparse4D `deploy/msda_fp16` implementation. The CUDA source preserves the upstream
kernel, including its FP32/FP16 checks that accept only coordinates strictly
inside `(0, 1)`. The import adds SPDX attribution and removes trailing whitespace;
it does not change the kernel arithmetic. The wrapper was originally generated
by [TensorRT Plugin Generator](https://github.com/NVIDIA-AI-IOT/tensorrt_plugin_generator).
The imported files retain their MIT license and Horizon Robotics/NVIDIA
attribution in [LICENSE](LICENSE) and source headers. The TAO integration adapts
the wrapper to validate the supported layout, formats, creator fields, and
serialized data, and to propagate CUDA errors. It also supplies a standalone
build and host/GPU tests. The TensorRT library build needs neither the upstream
Python debug binding nor its vendored pybind11 headers.
