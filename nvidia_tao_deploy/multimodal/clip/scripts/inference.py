# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP TensorRT inference — extract image and/or text embeddings to HDF5."""

import logging
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nvidia_tao_deploy.config.multimodal.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)

from nvidia_tao_deploy.multimodal.clip.dataloader import (
    get_preprocessing_params,
    _center_crop,
    _resize_shortest_edge,
)
from nvidia_tao_deploy.multimodal.clip.inferencer import create_clip_inferencer
from nvidia_tao_deploy.multimodal.clip.scripts.evaluate import load_model_config
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging as logger

logging.getLogger('PIL').setLevel(logging.WARNING)

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_image_files(image_dir):
    """Recursively find supported image files."""
    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def _save_embeddings(items, embeddings, results_dir, embedding_type='image'):
    """Save embeddings to HDF5 file.

    Format matches tao-pytorch exactly:
      - 'embeddings': float32 (N, D)
      - 'image_paths' or 'texts': variable-length UTF-8 strings
      - attrs: num_images/num_texts, embedding_dim, embedding_type
    """
    if embedding_type == 'image':
        filename = "image_embeddings.h5"
        items_key = 'image_paths'
        count_key = 'num_images'
    else:
        filename = "text_embeddings.h5"
        items_key = 'texts'
        count_key = 'num_texts'

    embeddings_file = os.path.join(results_dir, filename)

    with h5py.File(embeddings_file, 'w') as f:
        f.create_dataset(
            'embeddings',
            data=embeddings.astype(np.float32),
            compression='gzip',
            compression_opts=4,
        )
        dt = h5py.special_dtype(vlen=str)
        items_ds = f.create_dataset(items_key, (len(items),), dtype=dt)
        for i, item in enumerate(items):
            items_ds[i] = item

        f.attrs[count_key] = len(items)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['embedding_type'] = embedding_type

    logger.info("%s embeddings saved to %s", embedding_type.capitalize(), embeddings_file)


def _run_image_inference(
    trt_infer, image_dir, batch_size, results_dir,
    model_type, adaptor_name
):
    """Extract image embeddings via TRT and save to HDF5.

    Args:
        trt_infer: TensorRT inferencer.
        image_dir: Directory containing images.
        batch_size: Batch size for inference.
        results_dir: Output directory for embeddings.
        model_type: Model type string for preprocessing params.
        adaptor_name: Adaptor name for RADIO models.
    """
    image_files = _get_image_files(image_dir)
    if not image_files:
        logger.warning("No images found in %s", image_dir)
        return

    logger.info("Found %d images in %s", len(image_files), image_dir)

    _, h, w = trt_infer.image_input_shape

    mean, std, resampling, center_crop = get_preprocessing_params(
        model_type, adaptor_name
    )
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    logger.info("Using preprocessing: mean=%s, std=%s, resampling=%s, "
                "center_crop=%s", mean, std, resampling, center_crop)

    all_embeddings = []
    all_paths = []
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(image_files), batch_size),
                  total=num_batches, desc="Image embeddings"):
        batch_files = image_files[i:i + batch_size]
        images = []
        valid_paths = []
        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert('RGB')
                if center_crop:
                    img = _resize_shortest_edge(img, h, resampling)
                    img = _center_crop(img, w, h)
                else:
                    img = img.resize((w, h), resampling)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = (arr - mean) / std
                arr = np.transpose(arr, (2, 0, 1))
                images.append(arr)
                valid_paths.append(img_path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", img_path, e)

        if not images:
            continue

        imgs = np.stack(images, axis=0)
        feats = trt_infer.get_image_embeddings(imgs)
        all_embeddings.append(feats)
        all_paths.extend(valid_paths)

    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        _save_embeddings(all_paths, embeddings_array, results_dir, 'image')
        logger.info("Extracted embeddings for %d images", len(all_paths))
    else:
        logger.warning("No image embeddings were extracted")


def _run_text_inference(trt_infer, text_file, tokenizer_path,
                        batch_size, results_dir):
    """Extract text embeddings via TRT and save to HDF5."""
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        logger.warning("No texts found in %s", text_file)
        return

    logger.info("Found %d text prompts in %s", len(texts), text_file)
    logger.info("Loading tokenizer from: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    context_length = trt_infer.context_length

    all_embeddings = []
    all_texts = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size),
                  total=num_batches, desc="Text embeddings"):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        feats = trt_infer.get_text_embeddings(input_ids)
        all_embeddings.append(feats)
        all_texts.extend(batch_texts)

    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        _save_embeddings(all_texts, embeddings_array, results_dir, 'text')
        logger.info("Extracted embeddings for %d texts", len(all_texts))
    else:
        logger.warning("No text embeddings were extracted")


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name='clip', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """CLIP TRT inference — extract image and/or text embeddings."""
    infer_cfg = cfg.inference
    batch_size = infer_cfg.batch_size
    datasets = infer_cfg.datasets
    text_file = infer_cfg.text_file

    if not datasets and not text_file:
        raise ValueError(
            "At least one of inference.datasets or inference.text_file "
            "must be specified"
        )

    logger.info("TRT engine: %s", infer_cfg.trt_engine)
    logger.info("Batch size: %d", batch_size)

    model_config = load_model_config(infer_cfg.trt_engine)
    model_type = model_config['model_type']
    adaptor_name = model_config['adaptor_name']

    trt_infer = create_clip_inferencer(
        infer_cfg.trt_engine,
        batch_size=batch_size,
        data_format="channel_first",
    )

    results_dir = cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)

    if datasets:
        for ds in datasets:
            if ds.image_dir:
                logger.info("Processing image dataset: %s", ds.image_dir)
                _run_image_inference(
                    trt_infer, ds.image_dir, batch_size, results_dir,
                    model_type, adaptor_name,
                )

    if text_file:
        tokenizer_path = model_config['tokenizer_path']
        _run_text_inference(
            trt_infer, text_file, tokenizer_path,
            batch_size, results_dir,
        )

    logger.info("Inference complete.")


if __name__ == '__main__':
    main()
