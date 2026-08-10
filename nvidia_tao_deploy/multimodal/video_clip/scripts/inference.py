# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP TensorRT inference — extract text and/or gallery-video embeddings.

Writes HDF5 files in the same schema as the tao-pytorch trainer:
``text_embeddings.h5`` (texts) and/or ``video_embeddings.h5`` (gallery chunk ids).
"""

import json
import logging
import os

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nvidia_tao_deploy.config.multimodal.video_clip.default_config import (
    VideoCLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_deploy.multimodal.video_clip.dataloader import (
    VideoCLIPGalleryLoader,
    build_chunk_index,
    canonicalize_text,
)
from nvidia_tao_deploy.multimodal.video_clip.inferencer import (
    create_video_clip_inferencer,
)
from nvidia_tao_deploy.multimodal.video_clip.scripts.evaluate import load_model_config
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging as logger

logging.getLogger('PIL').setLevel(logging.WARNING)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _save_embeddings(items, embeddings, path, items_key, count_key, embedding_type):
    """Save embeddings + item ids to HDF5 (tao-pytorch schema)."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings.astype(np.float32),
                         compression='gzip', compression_opts=4)
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset(items_key, (len(items),), dtype=dt)
        for i, item in enumerate(items):
            ds[i] = item
        f.attrs[count_key] = len(items)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['embedding_type'] = embedding_type
    logger.info("Saved %d %s embeddings to %s", len(items), embedding_type, path)


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name='video_clip', mode='inference')
def main(cfg: ExperimentConfig) -> None:
    """Extract text and/or gallery-video embeddings via TRT."""
    infer_cfg = cfg.inference
    val_cfg = cfg.dataset.val
    batch_size = infer_cfg.batch_size
    results_dir = infer_cfg.results_dir or cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)

    model_config = load_model_config(infer_cfg.trt_engine)
    trt_infer = create_video_clip_inferencer(
        infer_cfg.trt_engine, batch_size=batch_size, data_format="channel_first",
    )

    did_something = False

    # Text embeddings from a prompt file.
    if infer_cfg.text_file:
        with open(infer_cfg.text_file, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer_path'])
        ctx = trt_infer.context_length
        do_canonicalize = model_config['canonicalize_text']
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
            batch = texts[i:i + batch_size]
            if do_canonicalize:
                batch = [canonicalize_text(t) for t in batch]
            tokens = tokenizer(batch, padding="max_length",
                               truncation=True, max_length=ctx, return_tensors="np")
            embs.append(trt_infer.get_text_embeddings(
                tokens["input_ids"].astype(np.int64)))
        _save_embeddings(texts, np.concatenate(embs, 0),
                         os.path.join(results_dir, "text_embeddings.h5"),
                         "texts", "num_texts", "text")
        did_something = True

    # Gallery-video embeddings from the eval GT + chunk metadata.
    gt_queries = getattr(val_cfg, "gt_queries", None)
    metadata = getattr(val_cfg, "metadata", None)
    if gt_queries and metadata:
        with open(gt_queries, "r", encoding="utf-8") as f:
            gallery_ids = [g["chunk_id"] for g in json.load(f)["gallery"]]
        chunk_index = build_chunk_index(
            metadata, data_root=getattr(val_cfg, "video_root", None),
            path_prefix_mapping=OmegaConf.to_container(
                getattr(val_cfg, "path_prefix_mapping", {}) or {}),
        )
        loader = VideoCLIPGalleryLoader(
            gallery_ids, chunk_index, num_frames=trt_infer.num_frames,
            image_size=trt_infer.image_input_shape[2], batch_size=batch_size,
            dtype=trt_infer.image_input_dtype,
        )
        embs, ids = [], []
        for videos, chunk_ids in tqdm(loader, total=len(loader), desc="Gallery videos"):
            embs.append(trt_infer.get_image_embeddings(videos))
            ids.extend(chunk_ids)
        _save_embeddings(ids, np.concatenate(embs, 0),
                         os.path.join(results_dir, "video_embeddings.h5"),
                         "chunk_ids", "num_videos", "video")
        did_something = True

    if not did_something:
        raise ValueError(
            "Nothing to do: set inference.text_file and/or dataset.val."
            "{gt_queries,metadata}."
        )
    logger.info("Inference complete.")


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
