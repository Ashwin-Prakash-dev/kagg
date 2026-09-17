# ---------------------------------------------------------------------
# GENERATED COPY -- do not edit here.
# Source of truth: <project>/src/evaluation.py
# Regenerate with: python scripts/create_colab_package.py
# ---------------------------------------------------------------------
"""
Evaluation for one resolution condition.

Everything here exists to make four differently-sized datasets comparable on
one scale. Predictions are made on the condition's own imagery, then mapped
back into ORIGINAL-resolution pixel coordinates before scoring:

    x_original = x_predicted / (out_dim / src_dim)

IoU is invariant under uniform scaling, so this changes no match decision. What
it buys is that every condition is scored in one coordinate frame against one
ground-truth table, with object-size bins anchored to the original resolution.
Without it, `AP_small` at r25 would be measured over whatever happened to be
small *at r25* -- a different set of ships in every row of the results table.

Latency is measured on the condition's real input size, before that mapping,
because the point of the cost comparison is what the model actually processes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import AREA_RANGES, evaluate, prf_at_threshold


def build_ground_truth(ships: pd.DataFrame, images: pd.DataFrame,
                       split: str = "Val") -> dict:
    """GT keyed by image stem, in original-resolution coordinates.

    Background images are included with zero boxes on purpose: dropping them
    would hide every false positive they provoke, which is exactly what they
    were retained to measure.
    """
    sel = images[images["Split"] == split]
    gt = {Path(r.basename).stem: {"boxes": np.zeros((0, 4)), "areas": np.zeros(0)}
          for r in sel.itertuples()}
    s = ships[ships["Split"] == split]
    for img_id, g in s.groupby("Img_ID"):
        stem = Path(g["basename"].iloc[0]).stem
        gt[stem] = {
            "boxes": g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(float),
            # The fixed original-resolution area that drives the size bin.
            "areas": g["barea"].to_numpy(float),
        }
    return gt


def predict_split(model, dataset_root: Path, images: pd.DataFrame,
                  split: str = "Val", conf: float = 0.001, iou: float = 0.7,
                  max_det: int = 1000, imgsz: int = 1024, device=None,
                  batch: int = 8, verbose: bool = False) -> tuple[dict, dict]:
    """Run inference over one split and return (detections, timing).

    Detections are returned in ORIGINAL-resolution coordinates.
    """
    split_dir = {"Train": "train", "Val": "val"}[split]
    img_dir = dataset_root / "images" / split_dir

    sel = images[images["Split"] == split].reset_index(drop=True)
    paths = [img_dir / f"{Path(b).stem}.jpg" for b in sel["basename"]]
    present = [p.exists() for p in paths]
    if not all(present):
        missing = [str(p) for p, ok in zip(paths, present) if not ok][:5]
        raise FileNotFoundError(f"{present.count(False)} images missing, e.g. {missing}")

    detections: dict[str, dict] = {}
    n_det = 0
    t_infer = 0.0

    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        rows = sel.iloc[start:start + batch]
        t0 = time.perf_counter()
        results = model.predict(
            [str(p) for p in chunk], imgsz=imgsz, conf=conf, iou=iou,
            max_det=max_det, device=device, verbose=verbose, stream=False)
        t_infer += time.perf_counter() - t0

        for res, (_, row) in zip(results, rows.iterrows()):
            stem = Path(row["basename"]).stem
            b = res.boxes
            if b is None or len(b) == 0:
                detections[stem] = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0)}
                continue
            xyxy = b.xyxy.cpu().numpy().astype(np.float64)
            scores = b.conf.cpu().numpy().astype(np.float64)

            # Map back to original-resolution coordinates using the realized
            # scale of THIS image, read from the rendered raster rather than
            # assumed from the nominal factor.
            out_h, out_w = res.orig_shape
            sx = out_w / float(row["ImageWidth"])
            sy = out_h / float(row["ImageHeight"])
            xyxy[:, [0, 2]] /= sx
            xyxy[:, [1, 3]] /= sy

            detections[stem] = {"boxes": xyxy, "scores": scores}
            n_det += len(scores)

    timing = {
        "n_images": len(paths),
        "total_inference_s": round(t_infer, 3),
        "latency_ms_per_image": round(t_infer / max(len(paths), 1) * 1000, 3),
        "images_per_s": round(len(paths) / max(t_infer, 1e-9), 2),
        "n_detections": n_det,
        "batch": batch,
        "imgsz": imgsz,
    }
    return detections, timing


def score(gt: dict, dt: dict, max_det: int = 1000,
          operating_conf: float = 0.25) -> dict:
    """Full metric block for one condition."""
    ap = evaluate(gt, dt, AREA_RANGES, max_det=max_det)
    prf = prf_at_threshold(gt, dt, conf=operating_conf, iou_thr=0.5,
                           max_det=max_det)
    out = {
        "mAP50": ap["all"].ap50,
        "mAP50_95": ap["all"].ap,
        "mAP75": ap["all"].ap75,
        "AR50_95": ap["all"].ar,
        "AP_small": ap["small"].ap,
        "AP_medium": ap["medium"].ap,
        "AP_large": ap["large"].ap,
        "AP50_small": ap["small"].ap50,
        "AP50_medium": ap["medium"].ap50,
        "AP50_large": ap["large"].ap50,
        "n_gt": {k: v.n_gt for k, v in ap.items()},
        "n_detections": ap["all"].n_dt,
        f"precision@{operating_conf}": prf["precision"],
        f"recall@{operating_conf}": prf["recall"],
        f"f1@{operating_conf}": prf["f1"],
        "operating_point": prf,
    }
    return out
