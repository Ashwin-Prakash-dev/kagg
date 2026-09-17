# ---------------------------------------------------------------------
# GENERATED COPY -- do not edit here.
# Source of truth: <project>/src/analysis.py
# Regenerate with: python scripts/create_colab_package.py
# ---------------------------------------------------------------------
"""
Object-size analysis for the resolution study (Parts G and H).

Two ideas carry the whole resolution x object-size analysis and are worth
stating up front:

1. Size bins are a property of the OBJECT, fixed once at original resolution.
   They are *not* recomputed per resolution. If bins were re-derived at each
   resolution, every object would slide into "small" at r25 and the
   resolution x size table would compare different populations at each row,
   which would make it uninterpretable. Binning by original-resolution area
   means each row of that table describes the same ships, so a drop in
   `AP_small` is attributable to lost detail rather than to reclassification.

2. Downsampling by a linear factor s scales box area by s^2. A 50% resolution
   reduction therefore removes 75% of an object's pixel footprint, not 50%.
   `downsampled_size_table` makes that explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# COCO object-size convention, in pixels^2 of the ORIGINAL image.
#   small  : area <  32^2 =  1024
#   medium : 1024 <= area < 96^2 = 9216
#   large  : area >= 9216
COCO_SMALL_MAX = 32 ** 2
COCO_MEDIUM_MAX = 96 ** 2
SIZE_BINS = ("small", "medium", "large")

# Resolution conditions: label -> linear scale factor applied to width/height.
RESOLUTIONS: dict[str, float] = {
    "original": 1.00,
    "r75": 0.75,
    "r50": 0.50,
    "r25": 0.25,
}

# The "does this object still exist as a detectable footprint" ladder (Part G).
SIZE_THRESHOLDS_PX = (4, 8, 16, 32, 64)


def assign_size_bin(area: pd.Series,
                    small_max: float = COCO_SMALL_MAX,
                    medium_max: float = COCO_MEDIUM_MAX) -> pd.Series:
    """Map original-resolution box area (px^2) to small / medium / large."""
    return pd.cut(
        area,
        bins=[-np.inf, small_max, medium_max, np.inf],
        labels=list(SIZE_BINS),
        right=False,
    ).astype(object)


def bbox_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Per-object geometry used throughout Part G."""
    out = pd.DataFrame(index=df.index)
    out["width_px"] = df["bw"]
    out["height_px"] = df["bh"]
    out["area_px2"] = df["barea"]
    out["relative_area"] = df["barea"] / (df["ImageWidth"] * df["ImageHeight"])
    # Aspect ratio is undefined for zero-height boxes; those are already
    # excluded upstream, but guard anyway so this is safe on raw input.
    out["aspect_ratio"] = df["bw"] / df["bh"].replace(0, np.nan)
    return out


def describe(series: pd.Series, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    s = series.dropna()
    d = {"n": int(len(s)), "min": float(s.min()), "max": float(s.max()),
         "mean": float(s.mean()), "std": float(s.std())}
    for q in qs:
        d[f"p{int(q * 100)}"] = float(s.quantile(q))
    return d


def threshold_counts(width: pd.Series, height: pd.Series,
                     thresholds=SIZE_THRESHOLDS_PX) -> dict:
    """Count objects below each `t x t` px footprint.

    Two readings are reported because they answer different questions:
      by_area  -- area < t*t. Total footprint, the usual small-object measure.
      by_side  -- max(width, height) < t. Whether the object's *longest* side
                  has dropped below t px, i.e. whether it has effectively
                  vanished along every axis. Stricter, and the more honest
                  test of "is there anything left to detect".
    """
    n = len(width)
    area = width * height
    longest = np.maximum(width, height)
    res = {}
    for t in thresholds:
        by_area = int((area < t * t).sum())
        by_side = int((longest < t).sum())
        res[f"{t}x{t}"] = {
            "below_by_area": by_area,
            "below_by_area_pct": round(by_area / n * 100, 3) if n else 0.0,
            "below_by_longest_side": by_side,
            "below_by_longest_side_pct": round(by_side / n * 100, 3) if n else 0.0,
        }
    return res


def downsampled_size_table(df: pd.DataFrame,
                           resolutions: dict[str, float] = RESOLUTIONS) -> dict:
    """Part G: how the ship size distribution moves as resolution drops.

    Boxes are scaled analytically (w*s, h*s) rather than by re-measuring the
    resized imagery, because the preprocessing applies exactly this linear
    transform -- see src/preprocessing.py.
    """
    out = {}
    for label, s in resolutions.items():
        w = df["bw"] * s
        h = df["bh"] * s
        out[label] = {
            "scale": s,
            "width_px": describe(w),
            "height_px": describe(h),
            "area_px2": describe(w * h),
            "below_thresholds": threshold_counts(w, h),
        }
    return out


def size_bin_survival(df: pd.DataFrame,
                      resolutions: dict[str, float] = RESOLUTIONS,
                      floor_px: int = 8) -> pd.DataFrame:
    """For each fixed original-resolution size bin, the median footprint and
    the share of objects whose longest side falls under `floor_px` at each
    resolution. This is the quantitative form of the study's core hypothesis:
    that small objects degrade first and fastest."""
    rows = []
    for label, s in resolutions.items():
        for b in SIZE_BINS:
            g = df[df["size_bin"] == b]
            if not len(g):
                continue
            w, h = g["bw"] * s, g["bh"] * s
            longest = np.maximum(w, h)
            rows.append({
                "resolution": label,
                "scale": s,
                "size_bin": b,
                "n_objects": len(g),
                "median_w_px": round(float(w.median()), 1),
                "median_h_px": round(float(h.median()), 1),
                "median_area_px2": round(float((w * h).median()), 1),
                f"pct_longest_side_lt_{floor_px}px": round(
                    float((longest < floor_px).mean() * 100), 2),
                "pct_area_lt_16px2": round(float(((w * h) < 16).mean() * 100), 2),
            })
    return pd.DataFrame(rows)
