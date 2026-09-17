"""
Reconstruct the ship-only annotation layer (image manifest + ground-truth
boxes) from the attached Kaggle Dataset, instead of shipping it precomputed.

    python scripts/build_annotation_layer.py

Why this exists: the local pipeline's `data/ship_only/{image_manifest.csv,
ship_annotations.parquet}` is ~1.5 MB. Embedding it in the setup notebook
(alongside the ~180 KB of code) pushed the notebook's own source past
Kaggle's 1 MB kernel-source limit ("kernel source must be less than 1
megabytes"), and the notebook could not be saved.

The fix is not to compress harder -- it's that this file doesn't need to
travel with the notebook at all. Its contents are fully recoverable from data
already inside the attached dataset: the YOLO label .txt files ARE the source
of truth for box geometry (normalized coordinates, invariant under resize,
already verified byte-identical across all four resolution conditions -- see
Step 4 of the notebook), and combined with each image's own pixel dimensions
(read from the 'original' condition, which alone carries native resolution),
every column the pipeline needs can be recomputed exactly:

    x_min = (cx - w/2) * ImageWidth   (and similarly x_max, y_min, y_max)

This reads image dimensions via PIL's lazy header parse (`Image.open(...).size`
does not decode pixel data), so it stays fast even for the largest source
tiles (~10,000x9,472 px).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.analysis import assign_size_bin
from src.paths import find_datasets_root


def build(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (images, ships) DataFrames from one condition's images+labels.

    Must be run against the 'original' condition: its per-image pixel
    dimensions are what "original-resolution ground truth" means throughout
    the rest of the pipeline (evaluation maps every condition's predictions
    back into this same coordinate frame).
    """
    image_rows: list[dict] = []
    ship_rows: list[dict] = []
    img_id = 0
    for split_dir, split_col in (("train", "Train"), ("val", "Val")):
        img_paths = sorted((dataset_dir / "images" / split_dir).glob("*.jpg"))
        for img_path in img_paths:
            with Image.open(img_path) as im:
                w, h = im.size  # header-only; no full decode
            lbl_path = dataset_dir / "labels" / split_dir / f"{img_path.stem}.txt"
            lines = ([l for l in lbl_path.read_text().splitlines() if l.strip()]
                     if lbl_path.exists() else [])
            for line in lines:
                _, cx, cy, bw, bh = (float(v) for v in line.split())
                x_min, x_max = (cx - bw / 2) * w, (cx + bw / 2) * w
                y_min, y_max = (cy - bh / 2) * h, (cy + bh / 2) * h
                ship_rows.append({
                    "Img_ID": img_id, "basename": img_path.name, "Split": split_col,
                    "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                    "bw": x_max - x_min, "bh": y_max - y_min,
                    "barea": (x_max - x_min) * (y_max - y_min),
                })
            image_rows.append({
                "Img_ID": img_id, "basename": img_path.name, "Split": split_col,
                "ImageWidth": w, "ImageHeight": h, "is_background": len(lines) == 0,
            })
            img_id += 1

    images = pd.DataFrame(image_rows)
    ships = pd.DataFrame(ship_rows)
    if len(ships):
        ships["size_bin"] = assign_size_bin(ships["barea"])
    else:
        ships["size_bin"] = pd.Series(dtype=object)
    return images, ships


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root", default=None)
    ap.add_argument("--out", default=None,
                    help="output dir (default: <pkg>/data/ship_only, i.e. "
                         "wherever src.paths.annotation_dir() looks)")
    args = ap.parse_args()

    datasets = find_datasets_root(args.datasets_root)
    out = Path(args.out) if args.out else PKG / "data" / "ship_only"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reconstructing the annotation layer from {datasets['original']} ...")
    images, ships = build(datasets["original"])
    images.to_csv(out / "image_manifest.csv", index=False)
    ships.to_parquet(out / "ship_annotations.parquet", index=False)

    n_bg = int(images["is_background"].sum())
    print(f"  {len(images):,} images ({len(images) - n_bg:,} ship-containing, "
          f"{n_bg:,} background), {len(ships):,} ship instances")
    print(f"  wrote {out / 'image_manifest.csv'}")
    print(f"  wrote {out / 'ship_annotations.parquet'}")


if __name__ == "__main__":
    main()
