"""
Part R -- qualitative prediction comparisons across resolutions.

    python scripts/qualitative_comparison.py --results-root <dir>

For each selected scene, renders one row of panels: the same tile at original,
r75, r50 and r25, each with that condition's own model's predictions overlaid
against the shared ground truth.

The scenes are chosen once and reused for every condition. Picking whatever
looks good per condition would produce a figure that argues for a conclusion
rather than showing one; holding the scenes fixed means the panels differ only
in what the model found.

Scene categories (Part R): small ships, medium ships, large ships, dense
scenes, isolated ships, difficult backgrounds. Selection is deterministic
given a seed, and the chosen scene ids are written alongside the figures so
the same set can be regenerated or challenged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.paths import annotation_dir, find_datasets_root, find_results_root

CONDITIONS = [("original", "00_baseline", 1.00), ("r75", "01_r75", 0.75),
              ("r50", "02_r50", 0.50), ("r25", "03_r25", 0.25)]

GT_COLOUR = "#00e5ff"
TP_COLOUR = "#28d17c"
FP_COLOUR = "#ff3b30"


def select_scenes(ships: pd.DataFrame, images: pd.DataFrame, seed: int,
                  per_category: int = 2) -> pd.DataFrame:
    """Pick a fixed, reusable set of val scenes spanning the Part R cases."""
    rng = np.random.default_rng(seed)
    val = images[(images["Split"] == "Val") & (~images["is_background"])].copy()
    s = ships[ships["Split"] == "Val"]
    agg = s.groupby("Img_ID").agg(
        n=("barea", "size"), min_area=("barea", "min"),
        max_area=("barea", "max"), med_area=("barea", "median"),
        n_small=("size_bin", lambda x: (x == "small").sum()),
        n_large=("size_bin", lambda x: (x == "large").sum()))
    val = val.join(agg, on="Img_ID").dropna(subset=["n"])

    picked: list[tuple[str, int]] = []
    taken: set[int] = set()

    def take(mask, category: str, k: int = per_category) -> None:
        cand = val[mask & ~val["Img_ID"].isin(taken)]
        if cand.empty:
            return
        idx = rng.choice(cand.index.to_numpy(), size=min(k, len(cand)), replace=False)
        for i in idx:
            picked.append((category, int(val.loc[i, "Img_ID"])))
            taken.add(int(val.loc[i, "Img_ID"]))

    take((val["n_small"] >= 3) & (val["n"] <= 12), "small_ships")
    take((val["med_area"].between(1024, 9216)) & (val["n"].between(2, 10)),
         "medium_ships")
    take(val["n_large"] >= 1, "large_ships")
    take(val["n"] >= 25, "dense_scene")
    take(val["n"] == 1, "isolated_ship")
    # Harbours and moored clusters: many ships packed into a small footprint,
    # where the surroundings look much like the targets.
    take((val["n"] >= 8) & (val["med_area"] < 2000), "difficult_background")

    out = val[val["Img_ID"].isin(taken)].copy()
    out["scene_category"] = out["Img_ID"].map(dict((i, c) for c, i in picked))
    return out.sort_values(["scene_category", "Img_ID"])


def match(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Greedy IoU matching; returns a bool array of which predictions hit."""
    from src.metrics import iou_matrix
    if len(pred) == 0:
        return np.zeros(0, dtype=bool)
    if len(gt) == 0:
        return np.zeros(len(pred), dtype=bool)
    ious = iou_matrix(pred, gt)
    taken = np.zeros(len(gt), dtype=bool)
    hit = np.zeros(len(pred), dtype=bool)
    for i in range(len(pred)):
        j = int(np.argmax(np.where(taken, -1.0, ious[i])))
        if ious[i, j] >= thr and not taken[j]:
            taken[j] = hit[i] = True
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--datasets-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO

    results_root = find_results_root(args.results_root)
    datasets = find_datasets_root(args.datasets_root)
    out_dir = Path(args.out) if args.out else results_root / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = annotation_dir()
    ships = pd.read_parquet(ann / "ship_annotations.parquet")
    images = pd.read_csv(ann / "image_manifest.csv")

    models = {}
    for label, run_id, _ in CONDITIONS:
        w = results_root / run_id / "best.pt"
        if w.exists():
            models[label] = YOLO(str(w))
        else:
            print(f"  no checkpoint for {label} ({w}); it will be skipped")
    if not models:
        raise SystemExit(f"No trained checkpoints under {results_root}.")

    scenes = select_scenes(ships, images, args.seed, args.per_category)
    print(f"{len(scenes)} scene(s) across "
          f"{scenes['scene_category'].nunique()} categories, "
          f"{len(models)} condition(s) with checkpoints\n")

    index = []
    for r in scenes.itertuples():
        gt = ships[ships["Img_ID"] == r.Img_ID]
        gt_boxes = gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(float)
        stem = Path(r.basename).stem

        avail = [(l, i, s) for l, i, s in CONDITIONS if l in models]
        fig, axes = plt.subplots(1, len(avail), figsize=(4.6 * len(avail), 5.2))
        axes = np.atleast_1d(axes)
        row = {"Img_ID": int(r.Img_ID), "image": r.basename,
               "scene_category": r.scene_category, "n_gt": int(len(gt_boxes)),
               "per_condition": {}}

        for ax, (label, _, scale) in zip(axes, avail):
            p = datasets[label] / "images" / "val" / f"{stem}.jpg"
            img = np.asarray(Image.open(p).convert("RGB"))
            h, w = img.shape[:2]
            sx, sy = w / r.ImageWidth, h / r.ImageHeight

            res = models[label].predict(str(p), imgsz=1024, conf=args.conf,
                                        iou=0.7, max_det=1000,
                                        device=args.device, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                pred = res.boxes.xyxy.cpu().numpy().astype(float)
                scores = res.boxes.conf.cpu().numpy()
            else:
                pred, scores = np.zeros((0, 4)), np.zeros(0)

            # Match in original-resolution space so the same IoU criterion
            # applies at every condition.
            pred_orig = pred.copy()
            if len(pred_orig):
                pred_orig[:, [0, 2]] /= sx
                pred_orig[:, [1, 3]] /= sy
            hit = match(pred_orig, gt_boxes)

            ax.imshow(img, interpolation="nearest")
            for b in gt_boxes:
                ax.add_patch(mpatches.Rectangle(
                    (b[0] * sx, b[1] * sy), (b[2] - b[0]) * sx, (b[3] - b[1]) * sy,
                    fill=False, edgecolor=GT_COLOUR, linewidth=1.0, linestyle=":"))
            for b, ok in zip(pred, hit):
                ax.add_patch(mpatches.Rectangle(
                    (b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                    edgecolor=TP_COLOUR if ok else FP_COLOUR, linewidth=1.3))

            tp, fp = int(hit.sum()), int((~hit).sum())
            fn = len(gt_boxes) - tp
            ax.set_title(f"{label} ({w}x{h})\nTP {tp}  FP {fp}  FN {fn}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            row["per_condition"][label] = {"tp": tp, "fp": fp, "fn": fn,
                                           "n_pred": int(len(pred))}

        fig.suptitle(
            f"{r.scene_category.replace('_', ' ')} — {r.basename} — "
            f"{len(gt_boxes)} ships   "
            f"(dotted cyan = ground truth, green = matched, red = false positive; "
            f"conf ≥ {args.conf})", fontsize=11, y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        name = f"{r.scene_category}__{stem}.png"
        fig.savefig(out_dir / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        row["figure"] = name
        index.append(row)
        print(f"  {name}   " + "  ".join(
            f"{l}:{row['per_condition'][l]['tp']}/{row['n_gt']}"
            for l, _, _ in avail))

    (out_dir / "scene_index.json").write_text(
        json.dumps({"conf_threshold": args.conf, "seed": args.seed,
                    "conditions": [l for l, _, _ in CONDITIONS if l in models],
                    "note": "Scenes are fixed across conditions; TP/FP/FN are "
                            "matched at IoU 0.5 in original-resolution "
                            "coordinates.",
                    "scenes": index}, indent=2), encoding="utf-8")
    print(f"\nWrote {len(index)} figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
