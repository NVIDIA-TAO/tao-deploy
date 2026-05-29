# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCRNet TensorRT inference."""

import logging
import os
import json
import tensorrt as trt
from tqdm import tqdm

from nvidia_tao_deploy.config.ocrnet.default_config import ExperimentConfig

from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.ocrnet.dataloader import OCRNetLoader
from nvidia_tao_deploy.cv.ocrnet.inferencer import OCRNetInferencer
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.ocrnet.utils import decode_ctc, decode_attn


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="ocrnet", mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Convert encrypted uff or onnx model to TRT engine."""
    engine_file = cfg.evaluate.trt_engine
    batch_size = cfg.evaluate.batch_size
    img_dirs = cfg.evaluate.test_dataset_dir
    gt_list = cfg.evaluate.test_dataset_gt_file
    character_list_file = cfg.dataset.character_list_file
    img_width = cfg.evaluate.input_width
    img_height = cfg.evaluate.input_height
    img_channel = cfg.model.input_channel
    prediction_type = cfg.model.prediction
    shape = [img_channel, img_height, img_width]

    ocrnet_engine = OCRNetInferencer(engine_path=engine_file,
                                     batch_size=batch_size)

    if prediction_type == "CTC":
        character_list = ["CTCBlank"]
    elif prediction_type == "Attn":
        character_list = ["[GO]", "[s]"]
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    with open(character_list_file, "r", encoding="utf-8") as f:
        for ch in f.readlines():
            ch = ch.strip()
            character_list.append(ch)

    inf_dl = OCRNetLoader(shape=shape,
                          image_dirs=[img_dirs],
                          label_txts=[gt_list],
                          batch_size=batch_size,
                          dtype=trt.nptype(ocrnet_engine.input_tensors[0].tensor_dtype))

    total_cnt = 0
    acc_cnt = 0
    for imgs, labels in tqdm(inf_dl):
        y_preds = ocrnet_engine.infer(imgs)
        output_ids, output_probs = y_preds
        total_cnt += len(output_ids)
        for output_id, output_prob, label in zip(output_ids, output_probs, labels):
            if prediction_type == "CTC":
                text, _ = decode_ctc(output_id, output_prob, character_list=character_list)
            else:
                text, _ = decode_attn(output_id, output_prob, character_list=character_list)
            if text == label:
                acc_cnt += 1

    log_info = f"Accuracy: {acc_cnt}/{total_cnt} {float(acc_cnt) / float(total_cnt)}"
    acc = float(acc_cnt) / float(total_cnt)
    # logging.info("Accuracy: {}/{} {}".format(acc_cnt, total_cnt, float(acc_cnt)/float(total_cnt)))
    logging.info(log_info)
    logging.info("TensorRT engine evaluation finished successfully.")

    # Store evaluation results into JSON
    eval_results = {"Accuracy": acc}
    with open(os.path.join(cfg.results_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f)


if __name__ == '__main__':
    main()
