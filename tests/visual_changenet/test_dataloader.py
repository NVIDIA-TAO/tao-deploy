# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet-Classification Data Loading Unit Tests."""

import pytest
import os
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from nvidia_tao_deploy.config.visual_changenet.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.optical_inspection.dataloader import OpticalInspectionDataLoader


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
csv_file = os.path.join(tmp_top_dir, 'test_data.csv')
SAMPLES = 10
BATCH_SIZE = 2
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_INPUT = 4


@pytest.fixture
def _test_dir():
    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_test_dir = os.path.join(tmp_top_dir, "test")
    tmp_golden_dir = os.path.join(tmp_top_dir, "golden")
    os.makedirs(tmp_test_dir, exist_ok=True)
    os.makedirs(tmp_golden_dir, exist_ok=True)

    test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)
    lighting = ['LowAngleLight', 'SolderLight', 'UniformLight', 'WhiteLight']
    labels = ['PASS', 'MISSING']
    total_samples = SAMPLES
    comp_name = 'test'
    csv_data = []
    for sample in range(total_samples):
        for label in labels:
            for light in lighting:
                save_light = comp_name + '_' + light + '.jpg'
                im.save(os.path.join(tmp_test_dir, save_light))
                im.save(os.path.join(tmp_golden_dir, save_light))
                csv_data.append({
                    'input_path': 'test',
                    'golden_path': 'golden',
                    'label': label,
                    'object_name': comp_name
                })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    yield csv_file


@pytest.fixture
def _test_exp_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]['classify']["train_dataset"]["csv_path"] = csv_file
    experiment_config["dataset"]['classify']["validation_dataset"]["csv_path"] = csv_file
    experiment_config["dataset"]['classify']["test_dataset"]["csv_path"] = csv_file
    experiment_config["dataset"]['classify']["infer_dataset"]["csv_path"] = csv_file
    experiment_config["dataset"]['classify']["train_dataset"]["images_dir"] = tmp_top_dir
    experiment_config["dataset"]['classify']["validation_dataset"]["images_dir"] = tmp_top_dir
    experiment_config["dataset"]['classify']["test_dataset"]["images_dir"] = tmp_top_dir
    experiment_config["dataset"]['classify']["infer_dataset"]["images_dir"] = tmp_top_dir
    experiment_config["dataset"]['classify']["num_input"] = NUM_INPUT
    experiment_config["dataset"]['classify']["image_width"] = IMAGE_WIDTH
    experiment_config["dataset"]['classify']["image_height"] = IMAGE_HEIGHT
    experiment_config["dataset"]['classify']["input_map"] = {'LowAngleLight': 0,
                                                        'SolderLight': 1,
                                                        'UniformLight': 2,
                                                        'WhiteLight': 3
                                                        }
    experiment_config["dataset"]['classify']["concat_type"] = 'linear'
    experiment_config["dataset"]['classify']["image_ext"] = '.jpg'
    experiment_config["dataset"]['classify']["batch_size"] = BATCH_SIZE

    experiment_config["results_dir"] = tmp_top_dir

    yield experiment_config


@pytest.mark.parametrize("split", ['train', 'valid', 'test', 'infer'])
@pytest.mark.parametrize("sample", [True, False])
@pytest.mark.cv_unit
def test_build_dataloader(_test_dir, _test_exp_spec, split, sample):

    dataloader = OpticalInspectionDataLoader(
        csv_file=_test_exp_spec.dataset.classify.infer_dataset.csv_path,
        input_data_path=_test_exp_spec.dataset.classify.infer_dataset.images_dir,
        train=False,
        data_config=_test_exp_spec.dataset.classify,
        batch_size=_test_exp_spec.dataset.classify.batch_size
    )
    total_num_samples = len(dataloader)
    for unit_batch, golden_batch in tqdm(dataloader, total=total_num_samples):
        assert unit_batch.shape[0] == BATCH_SIZE, "Incorrect batch size"
        assert unit_batch.shape[2] == _test_exp_spec["dataset"]['classify']["num_input"] * _test_exp_spec["dataset"]['classify']["image_height"], "Incorrect height"
        assert unit_batch.shape[3] == _test_exp_spec["dataset"]['classify']["image_width"], "Incorrect width"
        assert unit_batch.shape == golden_batch.shape, "Input shapes do not match for test and golden sample"

    tmp_top_obj.cleanup()
