# Sparse4D deployment

This backend adds `gen_trt_engine` and `inference` to TAO Deploy's existing
model-agnostic dispatcher. It consumes the **six-stage tao-pytorch Sparse4D
export ABI**, not a training checkpoint. This is source-branch functionality;
it does not imply that an already-released TAO image includes Sparse4D Deploy.

## Requirements and scope

- TensorRT 10.x, PyCUDA and a compatible NVIDIA GPU/runtime.
- A trusted, locally installed `nv::MSDA` plugin library built for that runtime.
  Supply its absolute path and SHA256. No plugin source or binary is bundled.
  Before every model build, generated FP32/FP16 probes check that the registered
  plugin rejects NaN, infinity, zero, one and out-of-bounds sampling coordinates.
  Valid sampling requires **both coordinates strictly inside (0, 1)**. This is
  also required when ONNX export sanitizes invalid coordinates to zero.
- Batch size **one**, one ordered scene stream, a fixed camera count and image
  resolution per engine. The supported production layout is 900 queries,
  600 cached queries, 256 feature channels and 11 anchor components; six decoder
  stages, two classification/quality outputs, and all twelve output names are
  required. There is no INT8, multi-GPU, multi-stream or evaluation action.
- FP32 is the initial correctness baseline (TF32 disabled). FP16 availability
  does not establish FP16 task accuracy; validate it on your own checkpoint/data.
- These are deployment-only specs; do not pass a full tao-pytorch training spec
  to engine generation or inference.

`setup.py`, dependency files and containers are unchanged. A separately installed
`sparse4d` console alias requires TAO Infra's entrypoint registration. Until then,
use the already-registered `model_agnostic` command or the module form below.

## Commands

Start from `specs/gen_trt_engine.yaml` and `specs/inference.yaml`. Replace paths,
plugin hash, image dimensions and camera count with your actual inputs. Keep
model settings identical between build and inference. Use fresh output paths.

```bash
model_agnostic gen_trt_engine -e /path/to/build.yaml
model_agnostic inference -e /path/to/infer.yaml

# Equivalent when running this checkout via PYTHONPATH:
python -m nvidia_tao_deploy.cv.common.entrypoint.entrypoint_agnostic gen_trt_engine -e /path/to/build.yaml
python -m nvidia_tao_deploy.cv.sparse4d.entrypoint.sparse4d inference -e /path/to/infer.yaml
```

Engine generation writes the engine and `<engine>.json`, binding ONNX/engine/
plugin hashes, TensorRT version, profile and plugin conformance results. Keep the
sidecar with the engine. Inference refuses changed engines, plugin hashes,
runtime versions or model settings. Treat engines and plugin libraries as
trusted executable artifacts, not safe inputs from unknown parties.

## Prepared frame interface

Inference accepts a JSON manifest and non-pickle NPZ files. This avoids a second,
potentially divergent implementation of image normalization and camera geometry.
To bridge an existing prepared Sparse4D dataset, run this optional utility **in
the tao-pytorch environment**, with both source packages available:

```bash
python -m nvidia_tao_deploy.cv.sparse4d.prepare_frames \
  --spec /path/to/resolved-training-experiment.yaml \
  --output /data/prepared-frames --max-frames 10
```

The utility uses tao-pytorch's prediction dataloader, including its validation
image resize/crop/normalization and projection transforms. Only use trusted
training annotations: that upstream loader can deserialize pickle. This bridge
is not a replacement for `annotations sparse4d_prepare` in tao-data-services.
It transforms that dataset pipeline's **inference inputs** into portable tensors.
No tao-pytorch or tao-core installation is needed to consume those tensors.

Alternatively, produce this explicit interface yourself:

```json
{"version": 1, "frames": [
  {"scene": "scene-a", "timestamp": 0.0, "tensors": "000000.npz"},
  {"scene": "scene-a", "timestamp": 0.033333, "tensors": "000001.npz"}
]}
```

Paths are relative to the manifest (absolute paths also work). Timestamps are
seconds. Preserve frame order and camera ordering. Each NPZ contains float32:

| Key | Shape | Meaning |
| --- | --- | --- |
| `img` | `(1, C, 3, H, W)` | Already normalized, resized/cropped network input |
| `projection_mat` | `(1, C, 4, 4)` | Local 3D to image projection after the same image transforms |
| `image_wh` | `(1, C, 2)` | `[W, H]`, not `[H, W]` |
| `T_global` (optional) | `(4, 4)` | Rigid local-to-global pose; omission explicitly means identity |

Images, matrices and poses must be finite. Invalid projected sampling locations
inside the model are handled by the hardened MSDA plugin. Input sizes are checked
against the engine profile; cameras are never silently padded, dropped or sorted.

## Temporal and output semantics

The caller maintains top-600 cache selection with confidence decay, projects
cached anchors using object velocity and ego motion, and assigns query IDs.
Scene changes, non-increasing timestamps and gaps above `max_time_interval`
clear cache/confidence/assignments; allocated IDs remain unique across scenes.
This is a batch-one deployment policy, not a claim of parity with every optional
training `InstanceBank` reset policy. Tied top-k scores use ascending query index;
PyTorch's tie ordering is not guaranteed to match.

Current exports can prune `timestamp` and bake their refinement interval into
the graph. Real timestamps still control host-side cache projection and resets;
this backend **does not turn a baked refinement interval into a dynamic one**.
Cold-start behavior follows the exported zero-cache graph, not a claim that
zero-filled cache attention equals native PyTorch's absent-cache branch.

Inference writes `predictions.jsonl`, `summary.json`, TAO `status.json` and the
resolved experiment spec. `save_raw: true` additionally writes all twelve engine
outputs per frame. Boxes are `[x,y,z,w,l,h,yaw,vx,vy,vz]` in the current local
coordinate frame. Tracking decoding selects one class per query, thresholds raw
class probability, then multiplies by centerness and reorders, matching
`SparseBox3DDecoder` tracking mode. IDs below the assignment threshold may be -1.
These JSON results are **not NVSchema or HOTA metrics**; no new evaluation or
DeepStream integration is implied.

## Tests

```bash
python -m pytest tests/sparse4d -q
SPARSE4D_PLUGIN_PATH=/plugins/libmsda.so \
SPARSE4D_PLUGIN_SHA256=<trusted-library-sha256> \
  python -m pytest tests/sparse4d/test_plugin_gpu.py -q
```

CPU tests generate their own arrays, JSON and NPZ fixtures. The opt-in GPU test
generates its own ONNX probe graph; it needs a supplied compatible plugin but no
dataset or checkpoint. Full-model validation should additionally compare all
engine outputs with the tao-pytorch **export wrapper on identical inputs**, then
separately assess native-model accuracy on a representative dataset.
