# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet TensorRT inference script.

This script provides high-performance inference capabilities for DepthNet models using
TensorRT optimization. It supports both monocular and stereo depth estimation with
efficient batch processing and visualization capabilities.

The inference process includes:
- TensorRT engine loading and initialization
- Batch processing of input images
- Depth map generation and post-processing
- Colorized visualization of depth maps
- Support for both monocular and stereo depth estimation
"""

import os
import logging
import cv2
import numpy as np
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_deploy.config.depth_net.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.depth_net.inferencer import DepthNetInferencer
from nvidia_tao_deploy.cv.depth_net.utils import vis_disparity, check_batch_sizes
from nvidia_tao_deploy.cv.depth_net.dataloader import DepthNetDataLoader

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='depth_net', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """
    Perform DepthNet inference using TensorRT engine with visualization.

    This function performs high-performance inference on input images using a
    TensorRT-optimized DepthNet model. It processes images in batches, generates
    depth maps, and creates colorized visualizations for easy interpretation.

    The inference supports:
    - Monocular depth estimation (single image input)
    - Stereo depth estimation (left-right image pairs)
    - Configurable preprocessing parameters
    - Batch processing for improved throughput
    - Colorized depth map visualization
    - Automatic output directory creation

    Args:
        cfg (ExperimentConfig): Configuration object containing all inference parameters
            including model paths, dataset configuration, and output settings.

    Raises:
        FileNotFoundError: If the TensorRT engine file does not exist.
        ValueError: If configuration parameters are invalid.
        RuntimeError: If inference process fails.

    Example:
        The function is typically called through the command line interface:
        ```bash
        python inference.py inference.trt_engine=/path/to/model.trt
        ```

    Output:
        - Colorized depth maps are saved to `cfg.results_dir/predicted_depth/`
        - Each output image corresponds to an input image with the same filename
        - Console output includes progress information and completion status
    """
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    trt_infer = DepthNetInferencer(cfg.inference.trt_engine,
                                   batch_size=cfg.dataset.infer_dataset.batch_size)

    c, h, w = trt_infer.input_tensors[0].shape
    # Dynamic engine: override H/W from augmentation.crop_size.
    is_dynamic_engine = any(
        getattr(t, "optimization_profile", None) is not None
        for t in trt_infer.input_tensors
    )
    if is_dynamic_engine:
        try:
            _aug = cfg.dataset.infer_dataset.augmentation
            _cs = _aug.get("crop_size", None) if _aug is not None else None
            if _cs is not None and len(_cs) == 2:
                h, w = int(_cs[0]), int(_cs[1])
        except Exception:
            pass

    loader = DepthNetDataLoader(
        cfg.dataset.infer_dataset.data_sources,
        (cfg.dataset.infer_dataset.batch_size, c, h, w),
        trt.nptype(trt_infer.input_tensors[0].tensor_dtype),
        preprocessor="DepthNet",
        evaluation=False,
    )

    # Create results directories
    output_annotate_root = os.path.join(cfg.results_dir, "predicted_depth")

    os.makedirs(output_annotate_root, exist_ok=True)

    for batches, img_paths, scales in tqdm(loader.get_batch(), total=loader.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        left_images, batches = check_batch_sizes(batches, img_paths)

        pred_depths = trt_infer.infer(batches)

        # squeeze depth tensor to 3D for FoundationStereo (B, 1, H, W) -> (B, H, W)
        if pred_depths.ndim == 4:
            pred_depths = pred_depths.squeeze(1)

        for batch, scale, pred_depth, img_path in zip(left_images, scales, pred_depths, img_paths):
            _, new_h, new_w = batch.shape
            orig_h, orig_w = int(scale[0] * new_h), int(scale[1] * new_w)
            pred_depth_engine = pred_depth  # at engine resolution
            pred_depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            pred_depth_vis = vis_disparity(pred_depth_resized, normalize_depth=False)
            # Use last two path components to keep filenames injective.
            parts = img_path.replace("\\", "/").split("/")
            fname = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            cv2.imwrite(os.path.join(output_annotate_root, fname), pred_depth_vis)
            # Optional raw fp32 disparity save (engine resolution, pre-resize).
            try:
                save_raw_npy = bool(getattr(cfg.inference, 'save_raw_npy', False))
            except Exception:
                save_raw_npy = False
            if save_raw_npy:
                npy_name = os.path.splitext(fname)[0] + '.npy'
                np.save(os.path.join(output_annotate_root, npy_name),
                        pred_depth_engine.astype(np.float32))

    logging.info("Finished inference.")


if __name__ == '__main__':
    main()
