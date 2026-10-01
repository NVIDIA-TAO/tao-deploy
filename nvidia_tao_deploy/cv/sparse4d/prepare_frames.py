# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional bridge: run in the tao-pytorch environment, not the deploy image.

This deliberately reuses the training repository's validation transforms to
avoid reimplementing image normalization, crop/resize and camera calibration.
Only load trusted training specs/datasets: their annotation reader uses pickle.
The generated NPZ files require no pickle or tao-pytorch at deployment time.
"""

import argparse
from itertools import islice
import json
from pathlib import Path

import numpy as np


def prepare_frames(spec, output, max_frames):
    """Serialize preprocessed batch-one frames without model or engine execution."""
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
    from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import (  # pylint: disable=import-outside-toplevel
        Sparse4DDataModule,
    )
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(spec)
    cfg.dataset.num_workers = 0
    module = Sparse4DDataModule(cfg)
    module.setup("predict")
    frames = []
    for index, batch in enumerate(islice(module.predict_dataloader(), max_frames)):
        arrays = {name: np.ascontiguousarray(batch[name].cpu().numpy(), dtype=np.float32)
                  for name in ("img", "projection_mat", "image_wh")}
        if arrays["img"].shape[0] != 1:
            raise ValueError("Prepared frames must have batch size one")
        metas = batch.get("img_metas")
        if metas is not None and "T_global" in metas[0]:
            arrays["T_global"] = np.asarray(metas[0]["T_global"], dtype=np.float32)
        elif "T_global" in batch:
            arrays["T_global"] = np.asarray(batch["T_global"][0], dtype=np.float32)
        name = f"{index:06d}.npz"
        np.savez_compressed(root / name, **arrays)
        frames.append({"scene": str(batch["scene_name"][0]),
                       "timestamp": float(batch["timestamp"][0]), "tensors": name})
    if not frames:
        raise ValueError("Dataset produced no inference frames")
    (root / "frames.json").write_text(json.dumps({"version": 1, "frames": frames}, indent=2), encoding="utf-8")
    return len(frames)


def main():
    """Prepare frames with the current tao-pytorch Sparse4D prediction pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Trusted resolved tao-pytorch experiment YAML")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()
    print(f"Prepared {prepare_frames(args.spec, args.output, args.max_frames)} frames")


if __name__ == "__main__":
    main()
