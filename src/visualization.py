# ---------------------------------------------------------------------
# GENERATED COPY -- do not edit here.
# Source of truth: <project>/src/visualization.py
# Regenerate with: python scripts/create_colab_package.py
# ---------------------------------------------------------------------
"""
Rendering helpers for preprocessing validation and qualitative results.

Boxes are always drawn from the *normalized* YOLO labels rather than from
pixel coordinates held in memory. That way the picture shows what a training
run would actually consume: if the written label file is wrong, the overlay is
wrong, and the error is visible instead of being masked by re-deriving the box
from a source that was never saved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BIN_COLOURS = {"small": "#ff2d55", "medium": "#ffb000", "large": "#00c2a8"}
DEFAULT_COLOUR = "#00e5ff"


def yolo_to_pixels(line: str, w: int, h: int) -> tuple[float, float, float, float]:
    """`class cx cy bw bh` (normalized) -> (x_min, y_min, box_w, box_h) in px."""
    _, cx, cy, bw, bh = line.split()
    cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
    return (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h


def draw_boxes(ax, lines, w: int, h: int, colours=None,
               linewidth: float = 1.0, min_visible_px: float = 3.0) -> None:
    """Overlay YOLO label lines onto an axis already showing the image.

    Boxes thinner than `min_visible_px` are padded outward for drawing only.
    At r25 a genuine box can be well under one pixel across; without this it
    would render as nothing at all and the figure would imply the annotation
    was lost when in fact it is present and correct.
    """
    import matplotlib.patches as mpatches

    for i, line in enumerate(lines):
        x, y, bw, bh = yolo_to_pixels(line, w, h)
        c = DEFAULT_COLOUR if colours is None else colours[i]
        dw = max(0.0, (min_visible_px - bw) / 2)
        dh = max(0.0, (min_visible_px - bh) / 2)
        ax.add_patch(mpatches.Rectangle(
            (x - dw, y - dh), bw + 2 * dw, bh + 2 * dh,
            fill=False, edgecolor=c, linewidth=linewidth))


def resolution_strip(out_path: Path, panels: list[dict], title: str,
                     colours_by_panel=None) -> None:
    """One row of panels showing the same scene at each resolution.

    Every panel is drawn at its true pixel size relative to the others, so the
    shrinking raster is visible rather than being normalised away by the
    figure layout -- that shrinkage is the independent variable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    base_h = 4.6
    widths = [p["image"].shape[1] / p["image"].shape[0] for p in panels]
    fig, axes = plt.subplots(
        1, n, figsize=(base_h * sum(widths) + 0.6 * n, base_h + 0.9),
        gridspec_kw={"width_ratios": widths})
    if n == 1:
        axes = [axes]

    for ax, p in zip(np.atleast_1d(axes), panels):
        img = p["image"]
        h, w = img.shape[:2]
        # interpolation="nearest" so a downsampled tile is not silently
        # smoothed by matplotlib into looking better than it is.
        ax.imshow(img, interpolation="nearest")
        draw_boxes(ax, p["lines"], w, h,
                   colours=p.get("colours"), linewidth=p.get("linewidth", 1.0))
        ax.set_title(f"{p['label']}\n{w}x{h} px  ({len(p['lines'])} boxes)",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Panel titles are two lines tall, so the suptitle needs explicit room;
    # tight_layout alone lets them collide on wide strips.
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def zoom_strip(out_path: Path, panels: list[dict], title: str,
               centre_xy_norm: tuple[float, float], span_norm: float) -> None:
    """Same scene, same *normalized* window, at each resolution.

    Cropping in normalized space means every panel frames the identical patch
    of ground, so the panels differ only in how much detail survives -- which
    is the comparison the study is making.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.6))
    if n == 1:
        axes = [axes]
    cxn, cyn = centre_xy_norm

    for ax, p in zip(np.atleast_1d(axes), panels):
        img = p["image"]
        h, w = img.shape[:2]
        half_w, half_h = span_norm * w / 2, span_norm * h / 2
        x0 = int(max(0, min(w - 1, cxn * w - half_w)))
        x1 = int(max(x0 + 1, min(w, cxn * w + half_w)))
        y0 = int(max(0, min(h - 1, cyn * h - half_h)))
        y1 = int(max(y0 + 1, min(h, cyn * h + half_h)))
        ax.imshow(img[y0:y1, x0:x1], interpolation="nearest")

        import matplotlib.patches as mpatches
        for line in p["lines"]:
            bx, by, bw, bh = yolo_to_pixels(line, w, h)
            ax.add_patch(mpatches.Rectangle(
                (bx - x0, by - y0), bw, bh, fill=False,
                edgecolor=DEFAULT_COLOUR, linewidth=1.4))
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
        ax.set_title(f"{p['label']}  ({x1 - x0}x{y1 - y0} px crop)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
