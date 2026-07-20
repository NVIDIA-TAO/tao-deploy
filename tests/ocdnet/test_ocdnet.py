# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import shutil
import subprocess
import sys

@pytest.fixture
def _test_dir():
#    os.system('tar -xf /home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data.tar.xz --directory tests/ocdnet/')
#    tmp_top_dir = "tests/ocdnet/ocdnet_data/"
    tmp_top_dir = "tests/ocdnet/"
    yield tmp_top_dir

class TestOCDNet():

    @pytest.mark.skip(reason="OCDNet INT8 calibration issues - skipping until TensorRT engine generation is fixed")
    @pytest.mark.ocdnet
    def test_ocdnet_generate_engine(self, tmpdir, _test_dir):
        """ Tests generation for tensorrt engine.
        Args:
            tmpdir: fixture providing a temporary directory unique to the test invocation.
        """
        print("tmpdir is {}".format(tmpdir))
        
        # Check model files and calibration directory
        model_dcn18 = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/ocdnet/model/dcn_resnet18_Uber_finetune_icdar15_best.onnx"
        cal_image_dir = "/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/train/img"
        
        # Check DCN18 model
        if not os.path.exists(model_dcn18):
            model_dir = os.path.dirname(model_dcn18)
            print(f"\n{'='*80}")
            print(f"ERROR: Model file not found: {model_dcn18}")
            print(f"{'='*80}")
            if os.path.exists(model_dir):
                print(f"\nContents of {model_dir}:")
                try:
                    for item in sorted(os.listdir(model_dir)):
                        item_path = os.path.join(model_dir, item)
                        if os.path.isdir(item_path):
                            print(f"  [DIR]  {item}")
                        else:
                            size = os.path.getsize(item_path)
                            print(f"  [FILE] {item} ({size} bytes)")
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print(f"\nDirectory does not exist: {model_dir}")
            print(f"{'='*80}\n")
        
        # Check calibration directory
        if not os.path.exists(cal_image_dir):
            print(f"\n{'='*80}")
            print(f"ERROR: Calibration directory not found: {cal_image_dir}")
            print(f"{'='*80}")
            parent_dir = os.path.dirname(cal_image_dir)
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
                image_files = [f for f in os.listdir(cal_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                required_images = 8 * 2  # batch_size * batches
                print(f"\nCalibration images: found {len(image_files)}, required {required_images}")
                if len(image_files) < required_images:
                    print(f"WARNING: Insufficient calibration images!")
                    print(f"First 10 files in {cal_image_dir}:")
                    for f in sorted(image_files)[:10]:
                        print(f"  {f}")
            except Exception as e:
                print(f"Error checking calibration images: {e}")
        
        # Create system call.
        call = (
            "python3 nvidia_tao_deploy/cv/ocdnet/scripts/gen_trt_engine.py "
            f" gen_trt_engine.onnx_file={model_dcn18} "
            f" gen_trt_engine.width=1280 "
            f" gen_trt_engine.height=736 "
            f" gen_trt_engine.tensorrt.data_type=int8 "
            f" gen_trt_engine.tensorrt.calibration.cal_cache_file=tests/ocdnet/out/cal_dcn18.bin "
            f" gen_trt_engine.tensorrt.calibration.cal_image_dir=[/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/train/img] "
            f" gen_trt_engine.tensorrt.calibration.cal_batch_size=8 "
            f" gen_trt_engine.tensorrt.calibration.cal_batches=2 "
            f" gen_trt_engine.trt_engine=tests/ocdnet/out/output_dcn18.engine "
        )
        # Run the call as subprocess.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Check whether model is present and encrypted.
        assert os.path.exists(f"tests/ocdnet/out/output_dcn18.engine")
        assert os.path.exists(f"tests/ocdnet/out/cal_dcn18.bin")

        # Check DCN50 model
        model_dcn50 = "/home/scratch.metropolis2/tao_ci/tao_deploy/models/ocdnet/model/dcn_resnet50.onnx"
        if not os.path.exists(model_dcn50):
            model_dir = os.path.dirname(model_dcn50)
            print(f"\n{'='*80}")
            print(f"ERROR: Model file not found: {model_dcn50}")
            print(f"{'='*80}")
            if os.path.exists(model_dir):
                print(f"\nContents of {model_dir}:")
                try:
                    for item in sorted(os.listdir(model_dir)):
                        item_path = os.path.join(model_dir, item)
                        if os.path.isdir(item_path):
                            print(f"  [DIR]  {item}")
                        else:
                            size = os.path.getsize(item_path)
                            print(f"  [FILE] {item} ({size} bytes)")
                except Exception as e:
                    print(f"  Error: {e}")
            print(f"{'='*80}\n")
        
        # Create system call.
        call = (
            "python3 nvidia_tao_deploy/cv/ocdnet/scripts/gen_trt_engine.py "
            f" gen_trt_engine.onnx_file={model_dcn50} "
            f" gen_trt_engine.width=1280 "
            f" gen_trt_engine.height=736 "
            f" gen_trt_engine.tensorrt.data_type=int8 "
            f" gen_trt_engine.tensorrt.calibration.cal_cache_file=tests/ocdnet/out/cal_dcn50.bin "
            f" gen_trt_engine.tensorrt.calibration.cal_image_dir=[/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/train/img] "
            f" gen_trt_engine.tensorrt.calibration.cal_batch_size=8 "
            f" gen_trt_engine.tensorrt.calibration.cal_batches=2 "
            f" gen_trt_engine.trt_engine=tests/ocdnet/out/output_dcn50.engine "
        )
        # Run the call as subprocess.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Check whether model is present and encrypted.
        assert os.path.exists(f"tests/ocdnet/out/output_dcn50.engine")
        assert os.path.exists(f"tests/ocdnet/out/cal_dcn50.bin")

    @pytest.mark.skip(reason="Depends on test_ocdnet_generate_engine which is skipped")
    @pytest.mark.ocdnet
    def test_ocdnet_evaluate(self, tmpdir, _test_dir):
        """ Tests evaluate on.
        Args:
            tmpdir: fixture providing a temporary directory unique to the test invocation.
        """
        # Create system call.
        call = (
            "python3 nvidia_tao_deploy/cv/ocdnet/scripts/evaluate.py "
            f" evaluate.trt_engine=tests/ocdnet/out/output_dcn18.engine "
            f" dataset.validate_dataset.data_path=['/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/test'] "
        )
        # Run the call as subprocess.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Create system call.
        call = (
            "python3 nvidia_tao_deploy/cv/ocdnet/scripts/evaluate.py "
            f" evaluate.trt_engine=tests/ocdnet/out/output_dcn50.engine "
            f" dataset.validate_dataset.data_path=['/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/test'] "
        )
        # Run the call as subprocess.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    @pytest.mark.skip(reason="Depends on test_ocdnet_generate_engine which is skipped")
    @pytest.mark.ocdnet
    def test_ocdnet_inference(self, tmpdir, _test_dir):
        """ Tests inference on.
        Args:
            tmpdir: fixture providing a temporary directory unique to the test invocation.
        """
        # Create system call.
        call = (
            "python3 nvidia_tao_deploy/cv/ocdnet/scripts/inference.py "
            f" inference.trt_engine=tests/ocdnet/out/output_dcn18.engine "
            f" inference.input_folder=/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/test_part/img "
            f" inference.results_dir=tests/ocdnet/out/result_ocdnet_inference_dcn18 "
        )
        # Run the call as subprocess.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Check whether infernce result is present.
        assert os.path.exists(f"tests/ocdnet/out/result_ocdnet_inference_dcn18/img_1_result.jpg")
        assert os.path.exists(f"tests/ocdnet/out/result_ocdnet_inference_dcn18/img_1.txt")
        assert os.path.getsize(f"tests/ocdnet/out/result_ocdnet_inference_dcn18/img_1.txt") > 0
        
        # Create system call.
        call = (
            "python3 nvidia_tao_deploy/cv/ocdnet/scripts/inference.py "
            f" inference.trt_engine=tests/ocdnet/out/output_dcn50.engine "
            f" inference.input_folder=/home/scratch.metropolis2/tao_ci/tao_deploy/data/L1/uber/ocdnet_data/test_data/test_part/img "
            f" inference.results_dir=tests/ocdnet/out/result_ocdnet_inference_dcn50 "
        )
        # Run the call as subprocess.
        subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

        # Check whether infernce result is present.
        assert os.path.exists(f"tests/ocdnet/out/result_ocdnet_inference_dcn50/img_1_result.jpg")
        assert os.path.exists(f"tests/ocdnet/out/result_ocdnet_inference_dcn50/img_1.txt")
        assert os.path.getsize(f"tests/ocdnet/out/result_ocdnet_inference_dcn50/img_1.txt") > 0

        # Delete temp output files
        shutil.rmtree(f"tests/ocdnet/out")
