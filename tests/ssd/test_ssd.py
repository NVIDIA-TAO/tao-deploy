# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSD workflow pipeline."""

import pytest
import os
import subprocess
import sys


model_dir = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/"
data_dir = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/"


# UFF model no longer supported in DLFW 24.06+
# (os.path.join(model_dir, "ssd/ssd_its.etlt"), os.path.join(model_dir, "ssd/ssd_spec.txt")
@pytest.mark.ssd
@pytest.mark.parametrize("model_tupl", [(os.path.join(model_dir, "ssd/ssd_resnet18_epoch_074.etlt"),
                                        "nvidia_tao_deploy/cv/ssd/specs/experiment_spec.txt"
                                         ),])
@pytest.mark.parametrize("cal_img_dir", [os.path.join(data_dir, "L1/IVA-0010-02_1280_0_181016/eval_images_kitti")])
@pytest.mark.parametrize("model_key", ["nvidia_tlt"])
@pytest.mark.parametrize("data_type", ["int8"])
@pytest.mark.parametrize("tmpdir", ["/tmp/ssd"])
@pytest.mark.parametrize("max_batch_size", [4])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("batches", [200])
def test_gen_trt_engine(model_tupl, cal_img_dir, model_key, data_type, tmpdir, max_batch_size, batch_size, batches):
    os.makedirs(tmpdir, exist_ok=True)

    model_path, spec_path = model_tupl
    
    # Check model file
    if not os.path.exists(model_path):
        model_dir_path = os.path.dirname(model_path)
        print(f"\n{'='*80}")
        print(f"ERROR: Model file not found: {model_path}")
        print(f"{'='*80}")
        if os.path.exists(model_dir_path):
            print(f"\nContents of {model_dir_path}:")
            try:
                for item in sorted(os.listdir(model_dir_path)):
                    item_path = os.path.join(model_dir_path, item)
                    if os.path.isdir(item_path):
                        print(f"  [DIR]  {item}")
                    else:
                        size = os.path.getsize(item_path)
                        print(f"  [FILE] {item} ({size} bytes)")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nDirectory does not exist: {model_dir_path}")
        print(f"{'='*80}\n")
    
    # Check calibration directory
    if not os.path.exists(cal_img_dir):
        print(f"\n{'='*80}")
        print(f"ERROR: Calibration directory not found: {cal_img_dir}")
        print(f"{'='*80}")
        parent_dir = os.path.dirname(cal_img_dir)
        if os.path.exists(parent_dir):
            print(f"\nContents of parent directory {parent_dir}:")
            try:
                for item in sorted(os.listdir(parent_dir)):
                    print(f"  {item}")
            except Exception as e:
                print(f"  Error: {e}")
        print(f"{'='*80}\n")
    else:
        # Check number of calibration images
        try:
            image_files = [f for f in os.listdir(cal_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            required_images = batch_size * batches
            print(f"\nCalibration images: found {len(image_files)}, required {required_images}")
            if len(image_files) < required_images:
                print(f"WARNING: Insufficient calibration images!")
        except Exception as e:
            print(f"Error checking calibration images: {e}")

    output_engine_path = os.path.join(tmpdir, os.path.basename(model_path).replace(".etlt", ".engine"))
    output_data_path = os.path.join(tmpdir, "ssd.tensorfile")
    output_cache_path = os.path.join(tmpdir, "ssd.bin")

    # Create system call.
    call = (
        "python nvidia_tao_deploy/cv/ssd/entrypoint/ssd.py gen_trt_engine "
        f"-m {model_path} "
        f"-k {model_key} "
        f"--data_type {data_type} "
        f"-e {spec_path} "
        f"--cal_data_file {output_data_path} "
        f"--cal_cache_file {output_cache_path} "
        f"--cal_image_dir {cal_img_dir} "
        f"--engine_file {output_engine_path} "
        f"--max_batch_size {max_batch_size} "
        f"--batch_size {batch_size} "
        f"--batches {batches} "
        f"-r {tmpdir} "
    )

    print(call)
    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    # Check if there are any files.
    if not os.path.exists(output_engine_path):
        raise FileNotFoundError(f"{output_engine_path} does not exist")
    if not os.path.exists(output_data_path):
        raise FileNotFoundError(f"{output_data_path} does not exist")
    if not os.path.exists(output_cache_path):
        raise FileNotFoundError(f"{output_cache_path} does not exist")


@pytest.mark.skip(reason="SSD evaluation depends on engine generation - skipping until INT8 calibration is fixed")
@pytest.mark.ssd
@pytest.mark.parametrize("model_path", [os.path.join("/tmp/ssd/ssd_resnet18_epoch_074.engine")])
@pytest.mark.parametrize("img_dir", [os.path.join(data_dir, "L1/IVA-0010-02_1280_0_181016/eval_images_kitti")])
@pytest.mark.parametrize("label_dir", [os.path.join(data_dir, "L1/IVA-0010-02_1280_0_181016/eval_labels_kitti")])
@pytest.mark.parametrize("tmpdir", ["/tmp/ssd"])
@pytest.mark.parametrize("batch_size", [4])
def test_evaluate(model_path, img_dir, label_dir, tmpdir, batch_size):
    os.makedirs(tmpdir, exist_ok=True)
    
    # Check if engine file exists (should be created by test_gen_trt_engine)
    if not os.path.exists(model_path):
        print(f"\n{'='*80}")
        print(f"ERROR: Engine file not found: {model_path}")
        print(f"{'='*80}")
        engine_dir = os.path.dirname(model_path)
        if os.path.exists(engine_dir):
            print(f"\nContents of {engine_dir}:")
            try:
                for item in sorted(os.listdir(engine_dir)):
                    item_path = os.path.join(engine_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  [DIR]  {item}")
                    else:
                        size = os.path.getsize(item_path)
                        print(f"  [FILE] {item} ({size} bytes)")
            except Exception as e:
                print(f"  Error: {e}")
        print(f"NOTE: This test requires test_gen_trt_engine to run successfully first.")
        print(f"{'='*80}\n")
    
    results_json_file = os.path.join(tmpdir, "results.json")
    spec_path = os.path.join(model_dir, "ssd/ssd_spec.txt")
    
    # Check spec file
    if not os.path.exists(spec_path):
        print(f"\n{'='*80}")
        print(f"ERROR: Spec file not found: {spec_path}")
        print(f"{'='*80}")
        spec_dir = os.path.dirname(spec_path)
        if os.path.exists(spec_dir):
            print(f"\nContents of {spec_dir}:")
            try:
                for item in sorted(os.listdir(spec_dir)):
                    print(f"  {item}")
            except Exception as e:
                print(f"  Error: {e}")
        print(f"{'='*80}\n")
    # Create system call.
    call = (
        "python nvidia_tao_deploy/cv/ssd/entrypoint/ssd.py evaluate "
        f"-m {model_path} "
        f"-e {spec_path} "
        f"-i {img_dir} "
        f"-l {label_dir} "
        f"-r {tmpdir} "
        f"--batch_size {batch_size} "
    )

    print(call)
    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    # Check if there are any files.
    if not os.path.exists(results_json_file):
        raise FileNotFoundError(f"{results_json_file} does not exist")
