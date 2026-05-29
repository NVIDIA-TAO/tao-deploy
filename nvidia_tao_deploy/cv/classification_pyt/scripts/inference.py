# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone TensorRT inference."""

import logging

import os
import pandas as pd
import numpy as np

import tensorrt as trt
from tqdm.auto import tqdm

from nvidia_tao_deploy.config.classification_pyt.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.classification_tf1.inferencer import ClassificationInferencer
from nvidia_tao_deploy.cv.classification_tf1.dataloader import ClassificationLoader
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="inference", schema=ExperimentConfig
)
@monitor_status(name='classification_pyt', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Classification TRT inference."""
    classmap = os.path.join(cfg.dataset.root_dir, 'classes.txt')

    if os.path.exists(classmap):
        # if classmap is provided, we explicitly set the mapping from the text file

        with open(classmap, "r", encoding="utf-8") as f:
            mapping_dict = {line.rstrip(): idx for idx, line in enumerate(sorted(f.readlines()))}
    else:
        # If not, the order of the classes are alphanumeric as defined by Keras
        # Ref: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/preprocessing/image.py#L507
        mapping_dict = {}
        for idx, subdir in enumerate(sorted(os.listdir(cfg.dataset.test_dataset.images_dir))):
            if os.path.isdir(os.path.join(cfg.dataset.test_dataset.images_dir, subdir)):
                mapping_dict[subdir] = idx

    image_mean = list(cfg.dataset.augmentation.mean)
    img_std = list(cfg.dataset.augmentation.std)
    batch_size = cfg.dataset.batch_size

    trt_infer = ClassificationInferencer(cfg.inference.trt_engine, data_format="channel_first", batch_size=batch_size)

    dl = ClassificationLoader(
        trt_infer.input_tensors[0].shape,
        [cfg.dataset.test_dataset.images_dir],
        mapping_dict,
        is_inference=True,
        data_format="channels_first",
        mode="torch",
        batch_size=batch_size,
        image_mean=image_mean,
        image_std=img_std,
        dtype=trt.nptype(trt_infer.input_tensors[0].tensor_dtype))

    result_csv_path = os.path.join(cfg.results_dir, 'result.csv')
    with open(result_csv_path, 'w', encoding="utf-8") as csv_f:
        for i, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
            image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]

            y_pred = trt_infer.infer(imgs)
            # Class output from softmax layer
            class_indices = np.argmax(y_pred, axis=1)
            # Map label index to label name
            class_labels = map(lambda i: list(mapping_dict.keys())
                               [list(mapping_dict.values()).index(i)],
                               class_indices)
            conf = np.max(y_pred, axis=1)
            # Write predictions to file
            df = pd.DataFrame(zip(image_paths, class_labels, conf))
            df.to_csv(csv_f, header=False, index=False)
    logging.info("Finished inference.")


if __name__ == '__main__':

    main()
