# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP TensorRT retrieval evaluation."""

import glob
import json
import logging
import os
import string
from typing import List

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nvidia_tao_deploy.config.multimodal.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_deploy.multimodal.clip.dataloader import CLIPRetrievalLoader
from nvidia_tao_deploy.multimodal.clip.evaluation import (
    RetrievalEvaluator,
    log_retrieval_metrics,
)
from nvidia_tao_deploy.multimodal.clip.inferencer import create_clip_inferencer
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.logging import status_logging
from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging as logger

logging.getLogger('PIL').setLevel(logging.WARNING)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Return canonicalized text (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/main/
    big_vision/evaluators/proj/image_text/prompt_engineering.py

    Args:
        text: String to be canonicalized.
        keep_punctuation_exact_string: If provided, this exact string is kept.
        trans_punctuation: Translation table for punctuation removal.

    Returns:
        Canonicalized text string.
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def find_export_artifacts(trt_engine_path: str) -> dict:
    """Find exported artifacts (config, tokenizer) associated with a TRT engine.

    During ONNX export, tao-pytorch saves:
    - *_config.yaml: Experiment configuration
    - *_tokenizer/: Tokenizer files

    Args:
        trt_engine_path: Path to the TRT engine file.

    Returns:
        Dictionary with paths:
            - config_path: Path to config file or None
            - tokenizer_path: Path to tokenizer directory or None
    """
    engine_dir = os.path.dirname(trt_engine_path)
    parent_dir = os.path.dirname(engine_dir)

    # Search patterns for config and tokenizer
    search_dirs = [
        engine_dir,
        os.path.join(parent_dir, "export"),
        parent_dir,
    ]

    result = {'config_path': None, 'tokenizer_path': None}

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        # Look for config
        if result['config_path'] is None:
            config_matches = glob.glob(os.path.join(search_dir, "*_config.yaml"))
            if config_matches:
                result['config_path'] = config_matches[0]

        # Look for tokenizer directory
        if result['tokenizer_path'] is None:
            tokenizer_matches = glob.glob(os.path.join(search_dir, "*_tokenizer"))
            for match in tokenizer_matches:
                if os.path.isdir(match):
                    result['tokenizer_path'] = match
                    break

        if result['config_path'] and result['tokenizer_path']:
            break

    return result


def load_model_config(trt_engine_path: str) -> dict:
    """Load model configuration from exported config file.

    Args:
        trt_engine_path: Path to the TRT engine file.

    Returns:
        Dictionary with model settings:
            - model_type: str
            - adaptor_name: str or None
            - canonicalize_text: bool (default False)
            - tokenizer_path: str (path to tokenizer directory)
    """
    artifacts = find_export_artifacts(trt_engine_path)

    if artifacts['config_path'] is None:
        raise FileNotFoundError(
            f"No export config file found for TRT engine: {trt_engine_path}\n"
            "The *_config.yaml file is required for correct preprocessing.\n"
            "Ensure the config file from ONNX export is in the same directory "
            "as the TRT engine or in the parent 'export' directory."
        )

    if artifacts['tokenizer_path'] is None:
        raise FileNotFoundError(
            f"No tokenizer directory found for TRT engine: {trt_engine_path}\n"
            "The *_tokenizer/ directory is required for text encoding.\n"
            "Ensure the tokenizer from ONNX export is in the same directory "
            "as the TRT engine or in the parent 'export' directory."
        )

    config = OmegaConf.load(artifacts['config_path'])
    model_cfg = config.get('model', {})

    model_type = getattr(model_cfg, 'type', None)
    adaptor_name = getattr(model_cfg, 'adaptor_name', None)
    canonicalize_text = getattr(model_cfg, 'canonicalize_text', False)

    if model_type is None:
        raise ValueError(
            f"Config file {artifacts['config_path']} is missing 'model.type'. "
            "This is required for correct preprocessing."
        )

    result = {
        'model_type': model_type,
        'adaptor_name': adaptor_name,
        'canonicalize_text': canonicalize_text,
        'tokenizer_path': artifacts['tokenizer_path'],
    }

    logger.info("Loaded model config from %s", artifacts['config_path'])
    logger.info("  model_type: %s", result['model_type'])
    logger.info("  adaptor_name: %s", result['adaptor_name'])
    logger.info("  canonicalize_text: %s", result['canonicalize_text'])
    logger.info("  tokenizer_path: %s", result['tokenizer_path'])

    return result


def extract_embeddings(
    trt_infer,
    dataloader: CLIPRetrievalLoader,
    tokenizer,
    context_length: int,
    batch_size: int,
    do_canonicalize: bool = False,
) -> tuple:
    """Extract image and text embeddings from dataset.

    Args:
        trt_infer: TensorRT inferencer for CLIP model.
        dataloader: Dataloader yielding (images, captions) batches.
        tokenizer: HuggingFace tokenizer for text encoding.
        context_length: Maximum sequence length for tokenization.
        batch_size: Batch size for text encoding.
        do_canonicalize: Whether to apply text canonicalization before
            tokenization. Should match the setting used during training.

    Returns:
        Tuple of (image_embeddings, text_embeddings) as numpy arrays.
    """
    all_image_embs = []
    all_text_embs = []
    all_captions: List[str] = []

    logger.info("Extracting image embeddings...")
    for images, captions in tqdm(dataloader, desc="Processing batches"):
        image_embs = trt_infer.get_image_embeddings(images)
        all_image_embs.append(image_embs)
        all_captions.extend(captions)

    all_image_embs = np.concatenate(all_image_embs, axis=0)
    logger.info("Extracted %d image embeddings", len(all_image_embs))

    logger.info("Extracting text embeddings...")
    if do_canonicalize:
        logger.info("Text canonicalization enabled")
    n_captions = len(all_captions)
    for i in tqdm(range(0, n_captions, batch_size), desc="Encoding text"):
        batch_captions = all_captions[i:i + batch_size]
        if do_canonicalize:
            batch_captions = [canonicalize_text(c) for c in batch_captions]
        tokens = tokenizer(
            batch_captions,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        text_embs = trt_infer.get_text_embeddings(input_ids)
        all_text_embs.append(text_embs)

    all_text_embs = np.concatenate(all_text_embs, axis=0)
    logger.info("Extracted %d text embeddings", len(all_text_embs))

    return all_image_embs, all_text_embs


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name='clip', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """CLIP TRT retrieval evaluation."""
    batch_size = cfg.evaluate.batch_size
    logger.info("TRT engine: %s", cfg.evaluate.trt_engine)
    logger.info("Batch size: %d", batch_size)

    # Load model config from exported config file
    model_config = load_model_config(cfg.evaluate.trt_engine)
    model_type = model_config['model_type']
    adaptor_name = model_config['adaptor_name']
    do_canonicalize = model_config['canonicalize_text']

    logger.info("Loading TensorRT engine...")
    trt_infer = create_clip_inferencer(
        cfg.evaluate.trt_engine,
        batch_size=batch_size,
        data_format="channel_first",
    )

    input_shape = trt_infer.image_input_shape
    img_dtype = trt_infer.image_input_dtype
    logger.info("Engine image input shape: %s", input_shape)

    context_length = trt_infer.context_length
    logger.info("Context length (from engine): %d", context_length)

    # Load tokenizer from exported directory
    tokenizer_path = model_config['tokenizer_path']
    logger.info("Loading tokenizer from: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    val_cfg = cfg.dataset.val
    if not val_cfg.datasets:
        raise ValueError(
            "No evaluation data configured. You must provide:\n"
            "  dataset.val.datasets:\n"
            "  - image_dir: /path/to/images\n"
            "    caption_dir: /path/to/captions"
        )

    logger.info("Retrieval evaluation: %d dataset(s)", len(val_cfg.datasets))

    all_image_embs = []
    all_text_embs = []

    for i, ds_cfg in enumerate(val_cfg.datasets):
        logger.info(
            "Dataset %d: images=%s, captions=%s",
            i + 1, ds_cfg.image_dir, ds_cfg.caption_dir
        )

        dataloader = CLIPRetrievalLoader(
            shape=input_shape,
            image_dir=ds_cfg.image_dir,
            model_type=model_type,
            adaptor_name=adaptor_name,
            caption_dir=ds_cfg.caption_dir,
            caption_file_suffix=ds_cfg.caption_file_suffix,
            image_list_file=ds_cfg.image_list_file,
            batch_size=batch_size,
            dtype=img_dtype,
        )
        logger.info(
            "Loaded %d samples in %d batches",
            dataloader.n_samples, len(dataloader)
        )

        image_embs, text_embs = extract_embeddings(
            trt_infer, dataloader, tokenizer, context_length, batch_size,
            do_canonicalize=do_canonicalize,
        )
        all_image_embs.append(image_embs)
        all_text_embs.append(text_embs)

    all_image_embs = np.concatenate(all_image_embs, axis=0)
    all_text_embs = np.concatenate(all_text_embs, axis=0)

    logger.info(
        "Total: %d images, %d texts",
        len(all_image_embs), len(all_text_embs)
    )

    logger.info("Computing retrieval metrics...")
    evaluator = RetrievalEvaluator(
        k_values=(1, 5, 10),
        compute_auc=True,
        batch_size=1024,
    )

    results = evaluator.evaluate_bidirectional(all_image_embs, all_text_embs)

    log_retrieval_metrics(results, prefix="TRT ")

    i2t = results.get('image_to_text')
    t2i = results.get('text_to_image')

    eval_results = {
        "num_images": int(len(all_image_embs)),
        "num_texts": int(len(all_text_embs)),
    }

    if i2t:
        eval_results["i2t_mAP"] = i2t.map_score
        eval_results["i2t_R@1"] = i2t.recall_at_k.get(1, 0)
        eval_results["i2t_R@5"] = i2t.recall_at_k.get(5, 0)
        eval_results["i2t_R@10"] = i2t.recall_at_k.get(10, 0)
        eval_results["i2t_MedR"] = i2t.median_rank
        eval_results["i2t_MeanR"] = i2t.mean_rank
        eval_results["i2t_AUC"] = i2t.auc

    if t2i:
        eval_results["t2i_mAP"] = t2i.map_score
        eval_results["t2i_R@1"] = t2i.recall_at_k.get(1, 0)
        eval_results["t2i_R@5"] = t2i.recall_at_k.get(5, 0)
        eval_results["t2i_R@10"] = t2i.recall_at_k.get(10, 0)
        eval_results["t2i_MedR"] = t2i.median_rank
        eval_results["t2i_MeanR"] = t2i.mean_rank
        eval_results["t2i_AUC"] = t2i.auc

    s_logger = status_logging.get_status_logger()
    s_logger.kpi = {
        "t2i_mAP": eval_results.get("t2i_mAP", 0),
        "i2t_mAP": eval_results.get("i2t_mAP", 0),
        "t2i_R@1": eval_results.get("t2i_R@1", 0),
        "i2t_R@1": eval_results.get("i2t_R@1", 0),
    }
    s_logger.write(
        message="Retrieval evaluation completed.",
        status_level=status_logging.Status.SUCCESS,
    )

    results_path = os.path.join(cfg.results_dir, "results.json")
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
