# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-CLIP TensorRT multi-relevance text->video retrieval evaluation.

Reproduces the tao-pytorch domain_test evaluation through the deploy TRT path:
embed the fixed gallery of video chunks and the text queries, then score each
query against the whole gallery by cosine similarity and compute mAP / Recall /
nDCG against its explicit ``relevant_clip_ids`` (overall + per slice).
"""

import glob
import json
import logging
import os

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
from nvidia_tao_deploy.multimodal.video_clip.evaluation import (
    RetrievalEvaluator,
    evaluate_by_slice,
    log_retrieval_metrics,
)
from nvidia_tao_deploy.multimodal.video_clip.inferencer import (
    create_video_clip_inferencer,
)
from nvidia_tao_deploy.cv.common.decorators import monitor_status
from nvidia_tao_deploy.cv.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_deploy.cv.common.logging import status_logging
from nvidia_tao_deploy.cv.common.logging.tlt_logging import logging as logger

logging.getLogger('PIL').setLevel(logging.WARNING)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_export_artifacts(trt_engine_path: str) -> dict:
    """Find the exported ``*_config.yaml`` and ``*_tokenizer/`` for an engine."""
    engine_dir = os.path.dirname(os.path.abspath(trt_engine_path))
    parent_dir = os.path.dirname(engine_dir)
    search_dirs = [engine_dir, os.path.join(parent_dir, "export"), parent_dir]

    result = {'config_path': None, 'tokenizer_path': None}
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        if result['config_path'] is None:
            matches = glob.glob(os.path.join(search_dir, "*_config.yaml"))
            if matches:
                result['config_path'] = matches[0]
        if result['tokenizer_path'] is None:
            for match in glob.glob(os.path.join(search_dir, "*_tokenizer")):
                if os.path.isdir(match):
                    result['tokenizer_path'] = match
                    break
        if result['config_path'] and result['tokenizer_path']:
            break
    return result


def load_model_config(trt_engine_path: str) -> dict:
    """Load model_type / canonicalize_text / tokenizer_path from export artifacts."""
    artifacts = find_export_artifacts(trt_engine_path)
    if artifacts['tokenizer_path'] is None:
        raise FileNotFoundError(
            f"No *_tokenizer/ directory found for engine {trt_engine_path}. "
            "Copy the tokenizer from ONNX export next to the engine."
        )
    model_type, canonicalize = "internvideo2-clip-l14", False
    if artifacts['config_path'] is not None:
        cfg = OmegaConf.load(artifacts['config_path'])
        model_cfg = cfg.get('model', {})
        model_type = getattr(model_cfg, 'type', model_type)
        canonicalize = bool(getattr(model_cfg, 'canonicalize_text', False))
    return {
        'model_type': model_type,
        'canonicalize_text': canonicalize,
        'tokenizer_path': artifacts['tokenizer_path'],
    }


def _embed_gallery(trt_infer, loader: VideoCLIPGalleryLoader):
    """Return (embeddings (M, D), ordered chunk_ids) for the gallery."""
    embs, ids = [], []
    for videos, chunk_ids in tqdm(loader, total=len(loader), desc="Gallery videos"):
        embs.append(trt_infer.get_image_embeddings(videos))
        ids.extend(chunk_ids)
    return np.concatenate(embs, axis=0), ids


def _embed_queries(trt_infer, texts, tokenizer, context_length, batch_size,
                   do_canonicalize=False):
    """Return (embeddings (N, D)) for the text queries."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Query text"):
        batch = texts[i:i + batch_size]
        if do_canonicalize:
            batch = [canonicalize_text(t) for t in batch]
        tokens = tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=context_length, return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        embs.append(trt_infer.get_text_embeddings(input_ids))
    return np.concatenate(embs, axis=0)


@hydra_runner(
    config_path=os.path.join(spec_root, "specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig,
)
@monitor_status(name='video_clip', mode='evaluate')
def main(cfg: ExperimentConfig) -> None:
    """Video-CLIP TRT retrieval evaluation."""
    eval_cfg = cfg.evaluate
    val_cfg = cfg.dataset.val
    batch_size = eval_cfg.batch_size
    logger.info("TRT engine: %s", eval_cfg.trt_engine)

    model_config = load_model_config(eval_cfg.trt_engine)
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer_path'])

    trt_infer = create_video_clip_inferencer(
        eval_cfg.trt_engine, batch_size=batch_size, data_format="channel_first",
    )
    num_frames = trt_infer.num_frames
    _, _, image_size, _ = trt_infer.image_input_shape
    context_length = trt_infer.context_length
    logger.info(
        "Engine: T=%d, image_size=%d, context_length=%d",
        num_frames, image_size, context_length,
    )

    # --- Ground truth: gallery (corpus) + queries (explicit relevance) --------
    with open(val_cfg.gt_queries, "r", encoding="utf-8") as f:
        gt = json.load(f)
    gallery = gt["gallery"]
    queries = gt["queries"]
    gallery_ids = [g["chunk_id"] for g in gallery]
    logger.info("Gallery: %d clips, Queries: %d", len(gallery_ids), len(queries))

    # --- Embed gallery videos --------------------------------------------------
    chunk_index = build_chunk_index(
        val_cfg.metadata,
        data_root=getattr(val_cfg, "video_root", None),
        path_prefix_mapping=OmegaConf.to_container(
            getattr(val_cfg, "path_prefix_mapping", {}) or {}
        ),
    )
    loader = VideoCLIPGalleryLoader(
        gallery_ids, chunk_index, num_frames=num_frames,
        image_size=image_size, batch_size=batch_size,
        dtype=trt_infer.image_input_dtype,
    )
    gallery_embs, ordered_ids = _embed_gallery(trt_infer, loader)
    gid_to_idx = {cid: i for i, cid in enumerate(ordered_ids)}

    # --- Embed query text ------------------------------------------------------
    query_texts = [q["query"] for q in queries]
    do_canonicalize = model_config['canonicalize_text']
    if do_canonicalize:
        logger.info("Applying text canonicalization before tokenization.")
    query_embs = _embed_queries(
        trt_infer, query_texts, tokenizer, context_length, batch_size,
        do_canonicalize=do_canonicalize,
    )

    # --- Map relevant_clip_ids -> gallery indices; keep slices -----------------
    ground_truth, slices, n_dropped = [], [], 0
    for q in queries:
        rel = [gid_to_idx[c] for c in q.get("relevant_clip_ids", []) if c in gid_to_idx]
        n_dropped += len(q.get("relevant_clip_ids", [])) - len(rel)
        ground_truth.append(rel)
        slices.append(q.get("slice") or "all")
    if n_dropped:
        logger.warning(
            "%d relevant_clip_ids were not present in the gallery and were "
            "dropped from scoring.", n_dropped,
        )

    # --- Metrics ---------------------------------------------------------------
    evaluator = RetrievalEvaluator(k_values=(1, 5, 10), compute_auc=True)
    results = evaluate_by_slice(
        evaluator, query_embs, gallery_embs, ground_truth, slices,
    )
    log_retrieval_metrics(results, prefix="TRT ")

    overall = results['overall']
    out = {name: m.to_dict() for name, m in results.items()}

    s_logger = status_logging.get_status_logger()
    s_logger.kpi = {
        "mAP": overall.map_score,
        "recall@1": overall.recall_at_k.get(1, 0),
        "recall@5": overall.recall_at_k.get(5, 0),
    }
    s_logger.write(
        message="Retrieval evaluation completed.",
        status_level=status_logging.Status.SUCCESS,
    )

    results_dir = eval_cfg.results_dir or cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("mAP (overall) = %.4f; results saved to %s",
                overall.map_score, results_path)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
