# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone TensorRT inference."""

import os
import logging
import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_deploy.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.centerpose.inferencer import CenterPoseInferencer
from nvidia_tao_deploy.cv.centerpose.dataloader import CPPredictDataset
from nvidia_tao_deploy.cv.centerpose.utils import transform_outputs, merge_outputs, save_inference_prediction, PnPProcess

from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="infer", schema=ExperimentConfig
)
@monitor_status(name='centerpose', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """CenterPose TRT Inference."""
    if not os.path.exists(cfg.inference.trt_engine):
        raise FileNotFoundError(f"Provided inference.trt_engine at {cfg.inference.trt_engine} does not exist!")

    trt_infer = CenterPoseInferencer(cfg.inference.trt_engine,
                                     batch_size=cfg.dataset.batch_size)
    c, h, w = trt_infer.input_tensors[0].shape

    batcher = CPPredictDataset(cfg.dataset, cfg.dataset.inference_data, (cfg.dataset.batch_size, c, h, w),
                               trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    pnp_solver = PnPProcess(cfg.inference)

    for batches, img_paths, (cxcy, max_axis) in tqdm(batcher.get_batch(), total=batcher.num_batches, desc="Producing predictions"):
        # Handle last batch as we artifically pad images for the last batch idx
        if len(img_paths) != len(batches):
            batches = batches[:len(img_paths)]

        det = trt_infer.infer(batches)

        # Post-processing
        transformed_det = transform_outputs(det, cxcy, max_axis, cfg.dataset.output_res)
        merged_det = merge_outputs(transformed_det)
        final_output = pnp_solver.get_process(merged_det)

        # Save the final results
        save_inference_prediction(final_output, cfg.results_dir, img_paths, cfg.inference)

    logging.info("Finished inference.")


if __name__ == '__main__':
    main()
