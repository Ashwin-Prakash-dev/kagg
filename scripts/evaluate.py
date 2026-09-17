"""
Score a trained checkpoint, with size-stratified AP on the fixed
original-resolution bins.

    # score a condition's own model on its own data
    python scripts/evaluate.py --config configs/r50.yaml --weights .../best.pt

    # score the SAME model on a different resolution (cross-resolution transfer)
    python scripts/evaluate.py --config configs/r50.yaml --weights .../best.pt \
        --eval-on original --tag r50_model_on_original

The `--eval-on` form is a separate question from the primary sweep: it asks
whether a model trained at one resolution still works at another, rather than
what resolution costs when you train and test consistently. Keep its output out
of the main results table -- `--tag` writes it to its own directory so it
cannot be picked up by collect_results.py by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.evaluation import build_ground_truth, predict_split, score
from src.paths import annotation_dir, find_datasets_root, find_results_root
from src.training import environment_info, load_master


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--eval-on", default=None,
                    help="resolution label to evaluate against "
                         "(default: the config's own condition)")
    ap.add_argument("--split", default="Val", choices=["Val", "Train"])
    ap.add_argument("--tag", default=None,
                    help="output subdirectory name; required when --eval-on "
                         "differs from the config's condition")
    ap.add_argument("--datasets-root", default=None)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    stub = yaml.safe_load(args.config.read_text())
    cfg = load_master(args.config.parent / stub.get("inherits", "master.yaml"))
    experiment = stub["experiment"]
    exp = cfg["experiments"][experiment]
    ev = cfg["evaluation"]

    eval_on = args.eval_on or exp["label"]
    cross = eval_on != exp["label"]
    if cross and not args.tag:
        raise SystemExit(
            f"--eval-on {eval_on} differs from this config's condition "
            f"({exp['label']}). Pass --tag so the result lands in its own "
            f"directory and cannot be mistaken for a primary-sweep number.")

    datasets = find_datasets_root(args.datasets_root)
    dataset_dir = datasets[eval_on]
    results_root = find_results_root(args.results_root)
    out_dir = results_root / (args.tag or exp["id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    ann = annotation_dir()
    ships = pd.read_parquet(ann / "ship_annotations.parquet")
    images = pd.read_csv(ann / "image_manifest.csv")

    print(f"Model      : {args.weights}")
    print(f"Trained on : {exp['label']}")
    print(f"Evaluating : {eval_on}  ({args.split} split)  dataset: {dataset_dir}")
    print(f"imgsz      : {cfg['train']['imgsz']}   max_det: {ev['max_det']}")

    t0 = time.time()
    dt, timing = predict_split(
        model, dataset_dir, images, split=args.split,
        conf=args.conf if args.conf is not None else ev["conf"],
        iou=ev["iou"], max_det=ev["max_det"], imgsz=cfg["train"]["imgsz"],
        device=args.device, batch=args.batch or cfg["train"]["batch"])
    gt = build_ground_truth(ships, images, split=args.split)
    m = score(gt, dt, max_det=ev["max_det"],
              operating_conf=ev["operating_point_conf"])
    m["timing"] = timing
    m["eval_wall_s"] = round(time.time() - t0, 1)

    payload = {
        "experiment": experiment,
        "trained_on": exp["label"],
        "evaluated_on": eval_on,
        "cross_resolution": cross,
        "split": args.split,
        "weights": str(args.weights),
        "metrics": m,
        "environment": environment_info(),
    }
    name = "metrics.json" if not cross else f"metrics_on_{eval_on}.json"
    (out_dir / name).write_text(json.dumps(payload, indent=2, default=str),
                                encoding="utf-8")

    if args.save_predictions:
        rows = []
        for stem, d in dt.items():
            for (x0, y0, x1, y1), s in zip(d["boxes"], d["scores"]):
                rows.append({"image": stem, "x_min": x0, "y_min": y0,
                             "x_max": x1, "y_max": y1, "score": s})
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        # Predictions are stored in ORIGINAL-resolution coordinates so they can
        # be overlaid on any condition's imagery without further conversion.
        pd.DataFrame(rows).to_csv(
            pred_dir / f"predictions_{eval_on}_{args.split}.csv", index=False)
        print(f"  saved {len(rows):,} predictions "
              f"(original-resolution coordinates)")

    print(f"\n  mAP50      {m['mAP50']:.4f}")
    print(f"  mAP50-95   {m['mAP50_95']:.4f}")
    print(f"  AP_small   {m['AP_small']:.4f}   (n_gt {m['n_gt']['small']:,})")
    print(f"  AP_medium  {m['AP_medium']:.4f}   (n_gt {m['n_gt']['medium']:,})")
    print(f"  AP_large   {m['AP_large']:.4f}   (n_gt {m['n_gt']['large']:,})")
    print(f"  P/R/F1 @ {ev['operating_point_conf']}: "
          f"{m['operating_point']['precision']:.4f} / "
          f"{m['operating_point']['recall']:.4f} / "
          f"{m['operating_point']['f1']:.4f}")
    print(f"\nWrote {out_dir / name}")


if __name__ == "__main__":
    main()
