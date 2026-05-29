# Supported Deploy Commands

This file is generated. Edit `setup.py`, model entrypoints, or model `scripts/` packages, then regenerate this file.

<!-- BEGIN GENERATED: supported-commands -->

_Source: `setup.py` console scripts plus each implementation's `scripts/` package. Regenerate with `python tools/update_docs_supported_commands.py`._

| Command | Domain | Implementation | Discovered subtasks |
| :--- | :--- | :--- | :--- |
| `centerpose` | cv | `nvidia_tao_deploy.cv.centerpose` | evaluate, gen_trt_engine, inference, default_specs |
| `classification_pyt` | cv | `nvidia_tao_deploy.cv.classification_pyt` | evaluate, gen_trt_engine, inference, default_specs |
| `classification_tf1` | cv | `nvidia_tao_deploy.cv.classification_tf1` | evaluate, gen_trt_engine, inference |
| `classification_tf2` | cv | `nvidia_tao_deploy.cv.classification_tf2` | evaluate, gen_trt_engine, inference, default_specs |
| `clip` | multimodal | `nvidia_tao_deploy.multimodal.clip` | evaluate, gen_trt_engine, inference, default_specs |
| `deformable_detr` | cv | `nvidia_tao_deploy.cv.deformable_detr` | evaluate, gen_trt_engine, inference, default_specs |
| `depth_net` | cv | `nvidia_tao_deploy.cv.depth_net` | evaluate, gen_trt_engine, inference, default_specs |
| `detectnet_v2` | cv | `nvidia_tao_deploy.cv.detectnet_v2` | evaluate, gen_trt_engine, inference |
| `dino` | cv | `nvidia_tao_deploy.cv.dino` | evaluate, gen_trt_engine, inference, default_specs |
| `dssd` | cv | `nvidia_tao_deploy.cv.ssd` | evaluate, gen_trt_engine, inference |
| `efficientdet_tf1` | cv | `nvidia_tao_deploy.cv.efficientdet_tf1` | evaluate, gen_trt_engine, inference |
| `efficientdet_tf2` | cv | `nvidia_tao_deploy.cv.efficientdet_tf2` | evaluate, gen_trt_engine, inference, default_specs |
| `faster_rcnn` | cv | `nvidia_tao_deploy.cv.faster_rcnn` | evaluate, gen_trt_engine, inference |
| `grounding_dino` | cv | `nvidia_tao_deploy.cv.grounding_dino` | evaluate, gen_trt_engine, inference, default_specs |
| `lprnet` | cv | `nvidia_tao_deploy.cv.lprnet` | evaluate, gen_trt_engine, inference |
| `mae` | cv | `nvidia_tao_deploy.cv.mae` | gen_trt_engine, default_specs |
| `mask2former` | cv | `nvidia_tao_deploy.cv.mask2former` | evaluate, gen_trt_engine, inference, default_specs |
| `mask_grounding_dino` | cv | `nvidia_tao_deploy.cv.mask_grounding_dino` | evaluate, gen_trt_engine, inference, default_specs |
| `mask_rcnn` | cv | `nvidia_tao_deploy.cv.mask_rcnn` | evaluate, gen_trt_engine, inference |
| `ml_recog` | cv | `nvidia_tao_deploy.cv.ml_recog` | evaluate, gen_trt_engine, inference, default_specs |
| `model_agnostic` | cv | `nvidia_tao_deploy.cv.common` | model-selected from spec; default_specs requires -m/--model_name |
| `multitask_classification` | cv | `nvidia_tao_deploy.cv.multitask_classification` | evaluate, gen_trt_engine, inference |
| `nvdinov2` | cv | `nvidia_tao_deploy.cv.nvdinov2` | gen_trt_engine, default_specs |
| `ocdnet` | cv | `nvidia_tao_deploy.cv.ocdnet` | evaluate, gen_trt_engine, inference, default_specs |
| `ocrnet` | cv | `nvidia_tao_deploy.cv.ocrnet` | evaluate, gen_trt_engine, inference, default_specs |
| `oneformer` | cv | `nvidia_tao_deploy.cv.oneformer` | evaluate, gen_trt_engine, inference, default_specs |
| `optical_inspection` | cv | `nvidia_tao_deploy.cv.optical_inspection` | evaluate, gen_trt_engine, inference, default_specs |
| `pointpillars` | cv | `nvidia_tao_deploy.cv.pointpillars` | evaluate, gen_trt_engine, inference, default_specs |
| `retinanet` | cv | `nvidia_tao_deploy.cv.retinanet` | evaluate, gen_trt_engine, inference |
| `rtdetr` | cv | `nvidia_tao_deploy.cv.rtdetr` | evaluate, gen_trt_engine, inference, default_specs |
| `segformer` | cv | `nvidia_tao_deploy.cv.segformer` | evaluate, gen_trt_engine, inference, default_specs |
| `ssd` | cv | `nvidia_tao_deploy.cv.ssd` | evaluate, gen_trt_engine, inference |
| `unet` | cv | `nvidia_tao_deploy.cv.unet` | evaluate, gen_trt_engine, inference |
| `visual_changenet` | cv | `nvidia_tao_deploy.cv.visual_changenet` | evaluate, gen_trt_engine, inference, default_specs |
| `yolo_v3` | cv | `nvidia_tao_deploy.cv.yolo_v3` | evaluate, gen_trt_engine, inference |
| `yolo_v4` | cv | `nvidia_tao_deploy.cv.yolo_v4` | evaluate, gen_trt_engine, inference |
| `yolo_v4_tiny` | cv | `nvidia_tao_deploy.cv.yolo_v4` | evaluate, gen_trt_engine, inference |

<!-- END GENERATED: supported-commands -->
