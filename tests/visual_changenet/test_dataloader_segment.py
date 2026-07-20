# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet-Segmentation Data Loading Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from nvidia_tao_deploy.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_deploy.cv.visual_changenet.segmentation.dataloader import ChangeNetDataLoader as ChangeNetSegmentDataLoader


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
SAMPLES = 10
BATCH_SIZE = 2
OUTPUT_SHAPE = 256
LABEL_TRANSFORM = 'norm'
DATASET = 'CNDataset'
IMG_FOLDER_NAME = 'A'
CHANGE_FOLDER_NAME = 'B'
LABEL_FOLDER_NAME = 'label'
LIST_FOLDER_NAME = 'list'


@pytest.fixture
def _test_dir():

    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_test_dir = os.path.join(tmp_top_dir, IMG_FOLDER_NAME)
    tmp_golden_dir = os.path.join(tmp_top_dir, CHANGE_FOLDER_NAME)
    tmp_list_dir = os.path.join(tmp_top_dir, LIST_FOLDER_NAME)
    tmp_label_dir = os.path.join(tmp_top_dir, LABEL_FOLDER_NAME)

    os.makedirs(tmp_test_dir, exist_ok=True)
    os.makedirs(tmp_golden_dir, exist_ok=True)
    os.makedirs(tmp_list_dir, exist_ok=True)
    os.makedirs(tmp_label_dir, exist_ok=True)

    #Input images
    test_data = np.random.rand(OUTPUT_SHAPE, OUTPUT_SHAPE, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)
    #GT Label image
    label_data = np.zeros((OUTPUT_SHAPE, OUTPUT_SHAPE))
    label_data = label_data.astype(np.uint8)
    im_label = Image.fromarray(label_data)
    
    total_samples = SAMPLES
    txt_data = []
    for sample in range(total_samples):
        im.save(os.path.join(tmp_test_dir, str(sample)+'.png'))
        im.save(os.path.join(tmp_golden_dir, str(sample)+'.png'))
        im_label.save(os.path.join(tmp_label_dir, str(sample)+'.png'))
        txt_data.append(str(sample)+'.png')
    
    splits = ['train', 'val', 'test']
    for split in splits:
        txt_file_name = os.path.join(tmp_list_dir, split+'.txt')
        with open (txt_file_name ,'w') as f:
            for data in txt_data:
                f.write(f"{data}\n")


@pytest.fixture
def _test_exp_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]['segment']["root_dir"] = tmp_top_dir
    experiment_config["dataset"]['segment']["label_transform"] = LABEL_TRANSFORM
    experiment_config["dataset"]['segment']["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]['segment']["dataset"] = DATASET
    experiment_config["dataset"]['segment']["image_folder_name"] = IMG_FOLDER_NAME
    experiment_config["dataset"]['segment']["change_image_folder_name"] = CHANGE_FOLDER_NAME
    experiment_config["dataset"]['segment']["list_folder_name"] = LIST_FOLDER_NAME
    experiment_config["dataset"]['segment']["annotation_folder_name"] = LABEL_FOLDER_NAME
    experiment_config["dataset"]['segment']["label_suffix"] = '.png'
    experiment_config["dataset"]['segment']["batch_size"] = BATCH_SIZE

    experiment_config["results_dir"] = tmp_top_dir

    yield experiment_config


@pytest.mark.parametrize("split", ['train', 'valid', 'test', 'infer'])
@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.cv_unit
def test_build_dataloader(_test_dir, _test_exp_spec, split, sample):

    dataloader = ChangeNetSegmentDataLoader(
        dataset_config=_test_exp_spec.dataset.segment,
        dtype=np.float32,
        mode='test',
        split=_test_exp_spec.dataset.segment.test_split
    )

    total_num_samples = len(dataloader)

    for idx, (img_1, img_2, label) in tqdm(enumerate(dataloader), total=total_num_samples):
        assert img_1.shape[0] == BATCH_SIZE, "Incorrect batch size"
        assert img_1.shape[2] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect height"
        assert img_1.shape[3] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect width"
        assert img_1.shape == img_2.shape, "Input images do not match in size"
        assert label.shape[0] == BATCH_SIZE, "Incorrect batch size"
        assert label.shape[1] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect height"
        assert label.shape[2] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect width"

    tmp_top_obj.cleanup()
