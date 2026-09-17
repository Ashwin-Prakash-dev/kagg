# ---------------------------------------------------------------------
# GENERATED COPY -- do not edit here.
# Source of truth: <project>/src/metrics.py
# Regenerate with: python scripts/create_colab_package.py
# ---------------------------------------------------------------------
"""
COCO-style detection metrics with fixed, caller-supplied object-size bins.

Why this exists rather than a call into pycocotools
---------------------------------------------------
Two reasons, one practical and one methodological.

Practical: pycocotools needs a C toolchain and does not install cleanly on the
Windows machine this study is prepared on, so the metric could not be exercised
or unit-tested locally before reaching Colab.

Methodological: the size-stratified numbers are the centrepiece of this study,
and COCO's area-range semantics -- specifically *which* detections get ignored
in a bin -- decide what `AP_small` actually means. That logic should be
readable in this repository rather than inherited from a binary.

The implementation follows COCOeval exactly: 101-point interpolated precision,
IoU thresholds 0.50:0.05:0.95, greedy score-ordered matching where each ground
truth is claimed at most once per threshold, and area-range filtering with
ignore semantics. `tests/test_metrics.py` checks it against hand-computable
cases, and `scripts/` cross-checks the overall numbers against the value
Ultralytics reports for the same predictions.

Evaluating across resolutions
-----------------------------
All geometry is expected in **original-resolution coordinates**. Predictions
made on an r25 image are divided by the realized scale before they get here.
IoU is invariant under uniform scaling, so matching is identical either way,
but working in one common frame means:

  * a detection's own area is comparable to the fixed size bins, so COCO's
    "ignore unmatched detections outside the area range" rule keeps meaning
    the same thing at every resolution;
  * `AP_small` at r25 is measured over exactly the ships that were small at
    original resolution, which is the Part H requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO defaults.
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)

# Area ranges, in px^2 of the ORIGINAL image. "all" must stay unbounded so the
# headline mAP is not silently restricted.
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 1024.0),          # < 32^2
    "medium": (1024.0, 9216.0),      # 32^2 .. 96^2
    "large": (9216.0, float("inf")), # >= 96^2
}


def iou_matrix(dt: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) boxes in xyxy."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)), dtype=np.float64)
    ix0 = np.maximum(dt[:, None, 0], gt[None, :, 0])
    iy0 = np.maximum(dt[:, None, 1], gt[None, :, 1])
    ix1 = np.minimum(dt[:, None, 2], gt[None, :, 2])
    iy1 = np.minimum(dt[:, None, 3], gt[None, :, 3])
    iw = np.clip(ix1 - ix0, 0, None)
    ih = np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    a_dt = np.clip(dt[:, 2] - dt[:, 0], 0, None) * np.clip(dt[:, 3] - dt[:, 1], 0, None)
    a_gt = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    union = a_dt[:, None] + a_gt[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


@dataclass
class ImageEval:
    """Per-image matching outcome for one area range, all IoU thresholds."""
    dt_scores: np.ndarray                       # (D,)
    dt_matched: np.ndarray                      # (T, D) bool
    dt_ignore: np.ndarray                       # (T, D) bool
    n_gt_considered: int                        # non-ignored GT in this range


def _evaluate_image(dt_boxes: np.ndarray, dt_scores: np.ndarray,
                    gt_boxes: np.ndarray, gt_areas: np.ndarray,
                    area_range: tuple[float, float],
                    max_det: int) -> ImageEval:
    """COCOeval.evaluateImg for a single image, single class."""
    lo, hi = area_range
    T = len(IOU_THRESHOLDS)

    # GT outside the area range is *ignored*, not deleted: a detection landing
    # on it must not be counted as a false positive for this bin.
    gt_ignore = (gt_areas < lo) | (gt_areas >= hi) if len(gt_boxes) else np.zeros(0, bool)
    # COCO sorts ignored GT last so the greedy matcher prefers a real match.
    order = np.argsort(gt_ignore, kind="stable")
    gt_boxes, gt_ignore = gt_boxes[order], gt_ignore[order]

    # Detections in descending score, truncated to max_det (as COCO does).
    dorder = np.argsort(-dt_scores, kind="stable")[:max_det]
    dt_boxes, dt_scores = dt_boxes[dorder], dt_scores[dorder]
    D, G = len(dt_boxes), len(gt_boxes)

    dt_matched = np.zeros((T, D), dtype=bool)
    dt_ignore = np.zeros((T, D), dtype=bool)
    if D == 0:
        return ImageEval(dt_scores, dt_matched, dt_ignore, int((~gt_ignore).sum()))

    ious = iou_matrix(dt_boxes, gt_boxes)
    dt_areas = ((dt_boxes[:, 2] - dt_boxes[:, 0])
                * (dt_boxes[:, 3] - dt_boxes[:, 1]))

    for t, thr in enumerate(IOU_THRESHOLDS):
        gt_taken = np.full(G, -1, dtype=int)
        for d in range(D):
            best_iou = min(thr, 1 - 1e-10)
            best_g = -1
            for g in range(G):
                if gt_taken[g] >= 0 and not gt_ignore[g]:
                    continue
                # GT is sorted ignored-last: once a real match exists, an
                # ignored GT can never be preferable, so stop.
                if best_g > -1 and not gt_ignore[best_g] and gt_ignore[g]:
                    break
                if ious[d, g] < best_iou:
                    continue
                best_iou = ious[d, g]
                best_g = g
            if best_g == -1:
                continue
            gt_taken[best_g] = d
            dt_matched[t, d] = True
            dt_ignore[t, d] = gt_ignore[best_g]

    # An unmatched detection whose own footprint is outside the bin is not this
    # bin's false positive.
    outside = (dt_areas < lo) | (dt_areas >= hi)
    dt_ignore |= (~dt_matched) & outside[None, :]
    return ImageEval(dt_scores, dt_matched, dt_ignore, int((~gt_ignore).sum()))


@dataclass
class APResult:
    ap: float                        # mAP@50:95
    ap50: float
    ap75: float
    ar: float                        # max recall at conf->0, averaged over IoU
    n_gt: int
    n_dt: int
    precision_curve: np.ndarray = field(repr=False, default=None)
    recall_curve: np.ndarray = field(repr=False, default=None)


def _accumulate(evals: list[ImageEval]) -> APResult:
    n_gt = sum(e.n_gt_considered for e in evals)
    scores = np.concatenate([e.dt_scores for e in evals]) if evals else np.zeros(0)
    # No ground truth in this bin: AP is undefined, not zero. COCO reports -1
    # here; nan is used so it cannot be silently averaged into a mean.
    if n_gt == 0:
        return APResult(float("nan"), float("nan"), float("nan"), float("nan"),
                        0, int(len(scores)))
    # Ground truth exists but nothing was detected: AP is a genuine zero. This
    # is the expected r25 failure mode for small ships, so it must not come
    # back as nan and vanish from the aggregate.
    if len(scores) == 0:
        return APResult(0.0, 0.0, 0.0, 0.0, n_gt, 0)

    matched = np.concatenate([e.dt_matched for e in evals], axis=1)
    ignored = np.concatenate([e.dt_ignore for e in evals], axis=1)

    order = np.argsort(-scores, kind="stable")
    matched, ignored = matched[:, order], ignored[:, order]

    T = len(IOU_THRESHOLDS)
    precisions = np.zeros((T, len(RECALL_THRESHOLDS)))
    recalls = np.zeros(T)
    pr_curves = []

    for t in range(T):
        keep = ~ignored[t]
        tp = np.cumsum(matched[t] & keep).astype(np.float64)
        fp = np.cumsum(~matched[t] & keep).astype(np.float64)
        rc = tp / n_gt
        pr = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        recalls[t] = rc[-1] if len(rc) else 0.0

        # Make precision monotonically non-increasing (COCO does this in place,
        # right to left) before interpolating at the 101 recall points.
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, RECALL_THRESHOLDS, side="left")
        q = np.zeros(len(RECALL_THRESHOLDS))
        valid = idx < len(pr)
        q[valid] = pr[idx[valid]]
        precisions[t] = q
        if t == 0:
            pr_curves = (q.copy(), RECALL_THRESHOLDS.copy())

    return APResult(
        ap=float(precisions.mean()),
        ap50=float(precisions[0].mean()),
        ap75=float(precisions[5].mean()),
        ar=float(recalls.mean()),
        n_gt=n_gt, n_dt=int(len(scores)),
        precision_curve=pr_curves[0] if len(pr_curves) else None,
        recall_curve=pr_curves[1] if len(pr_curves) else None,
    )


def evaluate(gt_by_image: dict, dt_by_image: dict,
             area_ranges: dict[str, tuple[float, float]] = None,
             max_det: int = 1000) -> dict[str, APResult]:
    """Compute AP for each named area range.

    Parameters
    ----------
    gt_by_image
        image key -> {"boxes": (M,4) xyxy in ORIGINAL-resolution pixels,
                      "areas": (M,) fixed original-resolution areas}.
        `areas` is passed separately rather than derived from `boxes` so the
        bin assignment stays anchored to the original resolution even if the
        caller ever supplies boxes in another frame.
    dt_by_image
        image key -> {"boxes": (N,4) xyxy, "scores": (N,)}. Images absent here
        are treated as having produced no detections.
    """
    area_ranges = area_ranges or AREA_RANGES
    out = {}
    keys = sorted(set(gt_by_image) | set(dt_by_image))
    for name, rng in area_ranges.items():
        evals = []
        for k in keys:
            g = gt_by_image.get(k, {})
            d = dt_by_image.get(k, {})
            gb = np.asarray(g.get("boxes", np.zeros((0, 4))), dtype=np.float64)
            ga = np.asarray(g.get("areas", np.zeros(len(gb))), dtype=np.float64)
            db = np.asarray(d.get("boxes", np.zeros((0, 4))), dtype=np.float64)
            ds = np.asarray(d.get("scores", np.zeros(0)), dtype=np.float64)
            if len(gb) == 0 and len(db) == 0:
                continue
            evals.append(_evaluate_image(db, ds, gb, ga, rng, max_det))
        out[name] = _accumulate(evals)
    return out


def prf_at_threshold(gt_by_image: dict, dt_by_image: dict, conf: float = 0.25,
                     iou_thr: float = 0.5, max_det: int = 1000) -> dict:
    """Precision / recall / F1 at a single operating point.

    AP integrates over every threshold, which is the right summary but is not
    what a deployed detector does. These are the numbers that describe the
    model at one usable confidence cut.
    """
    tp = fp = 0
    n_gt = 0
    for k in sorted(set(gt_by_image) | set(dt_by_image)):
        g = gt_by_image.get(k, {})
        d = dt_by_image.get(k, {})
        gb = np.asarray(g.get("boxes", np.zeros((0, 4))), dtype=np.float64)
        db = np.asarray(d.get("boxes", np.zeros((0, 4))), dtype=np.float64)
        ds = np.asarray(d.get("scores", np.zeros(0)), dtype=np.float64)
        keep = ds >= conf
        db, ds = db[keep], ds[keep]
        order = np.argsort(-ds, kind="stable")[:max_det]
        db = db[order]
        n_gt += len(gb)
        if len(db) == 0:
            continue
        if len(gb) == 0:
            fp += len(db)
            continue
        ious = iou_matrix(db, gb)
        taken = np.zeros(len(gb), dtype=bool)
        for i in range(len(db)):
            j = int(np.argmax(np.where(taken, -1.0, ious[i])))
            if ious[i, j] >= iou_thr and not taken[j]:
                taken[j] = True
                tp += 1
            else:
                fp += 1
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"conf": conf, "iou": iou_thr, "tp": tp, "fp": fp, "fn": fn,
            "n_gt": n_gt, "precision": precision, "recall": recall, "f1": f1}


def degradation(baseline: float, value: float) -> float:
    """Percentage drop relative to baseline (Part Q). Positive = worse."""
    if baseline in (0, None) or not np.isfinite(baseline) or not np.isfinite(value):
        return float("nan")
    return (baseline - value) / baseline * 100.0
