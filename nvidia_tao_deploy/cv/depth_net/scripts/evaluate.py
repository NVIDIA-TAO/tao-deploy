# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet TensorRT evaluation script.

This script provides comprehensive evaluation capabilities for DepthNet models using
TensorRT-optimized inference. It supports evaluation of both monocular and stereo
depth estimation models with industry-standard metrics.

The evaluation process includes:
- TensorRT engine loading and inference
- Batch processing of test images
- Ground truth comparison and metric computation
- Results storage in JSON format
- Support for both monocular and stereo depth estimation
"""

import os
import logging
import tensorrt as trt
import json
from tqdm.auto import tqdm
import operator

from nvidia_tao_deploy.config.depth_net.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.depth_net.inferencer import DepthNetInferencer
from nvidia_tao_deploy.cv.depth_net.evaluation import MonoDepthEvaluator, StereoDepthEvaluator
from nvidia_tao_deploy.cv.depth_net.dataloader import DepthNetDataLoader
from nvidia_tao_deploy.cv.depth_net.utils import check_batch_sizes

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.logging import status_logging

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name='depth_net', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """
    Evaluate DepthNet model using TensorRT engine with comprehensive metrics.

    This function performs end-to-end evaluation of a DepthNet model using a TensorRT
    engine. It processes test images in batches, computes depth predictions, and
    compares them against ground truth to calculate various evaluation metrics.

    The evaluation supports:
    - Monocular depth estimation (single image input)
    - Stereo depth estimation (left-right image pairs)
    - Configurable preprocessing parameters
    - Multiple evaluation metrics (abs_rel, d1, bp1, bp2, bp3, epe)
    - Results storage in JSON format

    Args:
        cfg (ExperimentConfig): Configuration object containing all evaluation parameters
            including model paths, dataset configuration, and evaluation settings.

    Raises:
        FileNotFoundError: If the TensorRT engine file does not exist.
        ValueError: If configuration parameters are invalid.
        RuntimeError: If evaluation process fails.

    Example:
        The function is typically called through the command line interface:
        ```bash
        python evaluate.py evaluate.trt_engine=/path/to/model.trt
        ```

    Output:
        - Evaluation results are saved to `cfg.results_dir/results.json`
        - Console output includes progress information and final metrics
    """
    if not os.path.exists(cfg.evaluate.trt_engine):
        raise FileNotFoundError(f"Provided evaluate.trt_engine at {cfg.evaluate.trt_engine} does not exist!")

    trt_infer = DepthNetInferencer(cfg.evaluate.trt_engine,
                                   batch_size=cfg.dataset.test_dataset.batch_size)

    c, h, w = trt_infer.input_tensors[0].shape
    # Dynamic engine: override H/W from augmentation.crop_size.
    is_dynamic_engine = any(
        getattr(t, "optimization_profile", None) is not None
        for t in trt_infer.input_tensors
    )
    if is_dynamic_engine:
        try:
            _aug = cfg.dataset.test_dataset.augmentation
            _cs = _aug.get("crop_size", None) if _aug is not None else None
            if _cs is not None and len(_cs) == 2:
                h, w = int(_cs[0]), int(_cs[1])
        except Exception:
            pass
    dataset_name = cfg.dataset.dataset_name
    is_stereo = dataset_name.lower() == "stereodataset"

    if is_stereo:
        evaluator = StereoDepthEvaluator.from_cfg(cfg)
    else:
        evaluator = MonoDepthEvaluator.from_cfg(cfg)

    loader = DepthNetDataLoader(
        cfg.dataset.test_dataset.data_sources,
        (cfg.dataset.test_dataset.batch_size, c, h, w),
        trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        preprocessor="DepthNet",
        evaluation=True,
    )

    per_sample_stereo = []

    for batches, img_paths, scales, gt_depths in tqdm(loader.get_batch(), total=loader.num_batches, desc="Producing predictions"):
        left_images, batches = check_batch_sizes(batches, img_paths)

        pred_depths = trt_infer.infer(batches)

        # FoundationStereo emits (B, 1, H, W); squeeze to (B, H, W).
        if pred_depths.ndim == 4:
            pred_depths = pred_depths.squeeze(1)

        pred_dict = []

        for img_path, batch, scale, pred_depth, gt_depth in zip(img_paths, left_images, scales, pred_depths, gt_depths):
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)

            if is_stereo:
                result = evaluator.prepare_sample(
                    pred_depth, gt_depth, new_h, new_w, orig_h, orig_w,
                )
                pred_dict.append(result["sample_dict"])
                per_sample_stereo.append({
                    "scene": os.path.basename(os.path.dirname(img_path)),
                    "path": img_path,
                    **result["per_sample"],
                })
            else:
                pred_dict.append(evaluator.prepare_sample(  # pylint: disable=unexpected-keyword-arg
                    pred_depth, gt_depth,
                    eval_crop_box=loader.eval_crop_box(gt_depth.shape),
                    orig_h=orig_h, orig_w=orig_w,
                ))
        evaluator.update(pred_dict)

    # Computing the final evaluation metrics and store evaluation results into JSON
    eval_results = evaluator.compute()
    logging.info("logging evaluation results.")
    status_logging_dict = {}
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        eval_results[key] = float(value)
        status_logging_dict[key] = float(value)
        logging.info("%s: %.9f", key, value)

    status_logging.get_status_logger().kpi = status_logging_dict
    status_logging.get_status_logger().write(
        message="Eval metrics generated.",
        status_level=status_logging.Status.SUCCESS
    )

    with open(os.path.join(cfg.results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)

    if per_sample_stereo:
        with open(os.path.join(cfg.results_dir, "per_sample_results.json"), "w", encoding="utf-8") as f:
            json.dump(per_sample_stereo, f, indent=4)
        logging.info("Per-sample stereo results saved.")

    logging.info("Finished Evaluation.")


if __name__ == '__main__':
    main()
