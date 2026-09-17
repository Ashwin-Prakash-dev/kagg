"""
Train and evaluate ONE resolution condition end to end, on Kaggle.

    python scripts/run_experiment.py --config configs/baseline.yaml
    python scripts/run_experiment.py --config configs/r25.yaml --epochs 1 --sanity
    python scripts/run_experiment.py --config configs/baseline.yaml --time 1.5

Writes everything for the condition into its own directory under the results
root (default /kaggle/working/results), and refuses to touch a directory that
already holds a finished run -- a re-run after an unnoticed Kaggle session
timeout is the realistic way that happens. Use --force to override
deliberately.

Resumability: if the run's own `last.pt` exists but `metrics.json` does not,
training resumes from that checkpoint (same epoch, optimizer state, RNG)
instead of restarting at epoch 1 -- see src/training.py:train_one.

--sanity stages a small, ship-guaranteed 32+16-image subset instead of
applying --fraction to the full condition -- see src/paths.stage_sanity_subset
for why --fraction alone does not actually make a sanity run cheap on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.evaluation import build_ground_truth, predict_split, score
from src.paths import (annotation_dir, ensure_writable_dataset, find_datasets_root,
                       find_resume_checkpoint, find_results_root, is_experiment_complete,
                       resolve_data_yaml, stage_sanity_subset)
from src.training import environment_info, load_master, set_seeds, train_one


def load_experiment_config(config_path: Path) -> tuple[dict, str]:
    stub = yaml.safe_load(config_path.read_text())
    master_path = config_path.parent / stub.get("inherits", "master.yaml")
    cfg = load_master(master_path)
    return cfg, stub["experiment"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--datasets-root", default=None,
                    help="explicit common root holding original/r75/r50/r25 "
                         "subfolders; default: auto-discover under /kaggle/input")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="environment override only; keep it the same for all "
                         "four conditions or the comparison is invalid")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="sanity runs only; the study value lives in master.yaml")
    ap.add_argument("--fraction", type=float, default=None,
                    help="sanity runs only: train on a fraction of the data")
    ap.add_argument("--time", type=float, default=None,
                    help="wall-clock training cap in hours for THIS invocation "
                         "(Ultralytics' own time= arg). Stops cleanly after a "
                         "completed epoch; if the study's target epoch count "
                         "isn't reached yet, this exits after saving the "
                         "checkpoint WITHOUT evaluating or writing metrics.json, "
                         "so the next invocation resumes training rather than "
                         "treating the condition as finished. Sized to fit one "
                         "invocation inside a single Kaggle session.")
    ap.add_argument("--sanity", action="store_true",
                    help="mark outputs as a throwaway sanity run")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing finished run")
    ap.add_argument("--stage-to-working", action="store_true",
                    help="copy this condition's dataset to /kaggle/working "
                         "before training (fallback only; default reads "
                         "directly from the read-only /kaggle/input mount)")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    cfg, experiment = load_experiment_config(args.config)
    exp = cfg["experiments"][experiment]
    datasets = find_datasets_root(args.datasets_root)
    dataset_dir = datasets[exp["label"]]
    if args.stage_to_working:
        from src.paths import stage_condition
        dataset_dir = stage_condition(exp["label"], dataset_dir)
    results_root = find_results_root(args.results_root)

    if args.sanity:
        # A sanity run's whole point is proving the plumbing works in a
        # couple of minutes. Pointing Ultralytics at the full ~7,706-image
        # condition would not actually be cheap: it scans and verifies every
        # image/label pair to build its label cache BEFORE --fraction is
        # applied, and that cache can never be saved on the read-only
        # /kaggle/input mount -- so a "5%" run still pays the full scan cost,
        # on every launch, across all four conditions. Stage a small,
        # ship-guaranteed subset instead; both training and evaluation read
        # from it.
        eval_dir = stage_sanity_subset(exp["label"], dataset_dir)
        data_yaml = eval_dir / "data.yaml"
    else:
        # The data.yaml under dataset_dir bakes in the ABSOLUTE path of the
        # machine that prepared it (e.g. a Windows drive path) as its `path:`
        # field. That is stale here, and Ultralytics mis-resolves it silently
        # rather than raising a clear error -- so a corrected copy (path:
        # rewritten to the actually-discovered dataset_dir; train/val/nc/names
        # untouched) is written under /kaggle/working and used instead.
        eval_dir = dataset_dir
        data_yaml = resolve_data_yaml(exp["label"], dataset_dir)

    run_name = exp["id"] + ("_sanity" if args.sanity else "")
    out_dir = results_root / run_name
    if is_experiment_complete(out_dir) and not args.force:
        print(f"[skip] {out_dir / 'metrics.json'} already exists -- {run_name} "
              f"has already completed. Pass --force to overwrite.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: Ultralytics writes weights/last.pt every epoch. If it
    # exists without a finished metrics.json, this is a resume, not a fresh
    # start.
    train_project = out_dir / "train"
    resume_from = None if args.force else find_resume_checkpoint(out_dir, exp["id"])

    extra = {k: v for k, v in {
        "device": args.device, "batch": args.batch, "workers": args.workers,
        "epochs": args.epochs, "fraction": args.fraction, "time": args.time,
    }.items() if v is not None}

    print("=" * 70)
    print(f"{experiment}  {exp['label']}  (scale {exp['scale']})")
    print(f"  dataset  : {dataset_dir}")
    if args.sanity:
        n_tr = len(list((eval_dir / "images" / "train").glob("*.jpg")))
        n_va = len(list((eval_dir / "images" / "val").glob("*.jpg")))
        print(f"  data.yaml: {data_yaml}  (staged {n_tr}+{n_va}-image sanity "
              f"subset, not the full {exp['label']} condition)")
    else:
        print(f"  data.yaml: {data_yaml}  (path: corrected from source)")
    print(f"  results  : {out_dir}")
    print(f"  imgsz    : {cfg['train']['imgsz']}   epochs: "
          f"{extra.get('epochs', cfg['train']['epochs'])}   "
          f"batch: {extra.get('batch', cfg['train']['batch'])}")
    if resume_from:
        print(f"  RESUMING from {resume_from}")
    if extra:
        print(f"  env overrides: {extra}")
    print("=" * 70)

    set_seeds(cfg["reproducibility"]["seed"])

    # The resolved config is written BEFORE training, so an interrupted run
    # still leaves a record of exactly what it was going to do.
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump({"experiment": experiment, "resolved_from": str(args.config),
                        "sanity": args.sanity, "env_overrides": extra,
                        "dataset_dir": str(dataset_dir), "master": cfg},
                       sort_keys=False), encoding="utf-8")

    record = train_one(cfg, experiment, project=train_project,
                       data_yaml=data_yaml, extra=extra,
                       resume_from=resume_from)
    run_dir = Path(record["run_dir"])

    # Ultralytics' own per-epoch log is the training curve; keep it next to the
    # metrics rather than buried in the trainer directory.
    csv_src = run_dir / "results.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, out_dir / "training_log.csv")
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, out_dir / "best.pt")
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        shutil.copy2(last, out_dir / "last.pt")
    curves = out_dir / "curves"
    curves.mkdir(exist_ok=True)
    for png in run_dir.glob("*.png"):
        shutil.copy2(png, curves / png.name)

    if not record["fully_trained"]:
        # --time cut this invocation short before the study's target epoch
        # count. The checkpoint above is already saved (that's what makes
        # the next invocation's resume possible), but this run is NOT
        # finished: evaluating a partially-trained model and writing
        # metrics.json would make run_all.py / collect_results.py treat this
        # condition as done. Exit here instead, so the next invocation's
        # find_resume_checkpoint() picks it back up automatically.
        print(f"\nTime budget reached: {record['completed_epochs']}/"
              f"{record['target_epochs']} epochs done for {exp['label']}. "
              f"Saved {out_dir / 'last.pt'}; NOT evaluating yet.")
        print("Re-run (same command, or via run_all.py) to continue training "
              "this condition from here.")
        return

    metrics = {"experiment": experiment, "run": record}

    if not args.skip_eval:
        from ultralytics import YOLO
        print("\nEvaluating on the val split, in original-resolution coordinates ...")
        ann = annotation_dir()
        ships = pd.read_parquet(ann / "ship_annotations.parquet")
        images = pd.read_csv(ann / "image_manifest.csv")
        if args.sanity:
            # The annotation tables cover the full 2,776-image val split, but
            # eval_dir only holds the staged handful -- predict_split() hard
            # -fails on any listed image it can't find on disk, so both
            # tables are narrowed to exactly what was staged (ground truth
            # and predictions then cover the same images, not a subset vs.
            # the full split).
            stems = {p.stem for p in (eval_dir / "images" / "val").glob("*.jpg")}
            images = images[images["basename"].apply(lambda b: Path(b).stem in stems)]
            ships = ships[ships["basename"].apply(lambda b: Path(b).stem in stems)]

        model = YOLO(str(best if best.exists() else run_dir / "weights" / "last.pt"))
        ev = cfg["evaluation"]
        t0 = time.time()
        dt, timing = predict_split(
            model, eval_dir, images, split="Val",
            conf=ev["conf"], iou=ev["iou"], max_det=ev["max_det"],
            imgsz=cfg["train"]["imgsz"], device=args.device,
            batch=extra.get("batch", cfg["train"]["batch"]))
        gt = build_ground_truth(ships, images, split="Val")
        m = score(gt, dt, max_det=ev["max_det"],
                  operating_conf=ev["operating_point_conf"])
        m["timing"] = timing
        m["eval_wall_s"] = round(time.time() - t0, 1)
        metrics["metrics"] = m

        # Ultralytics' own val() as an independent cross-check of the headline
        # numbers. A large disagreement means one of the two is wrong and the
        # results should not be trusted until it is explained.
        try:
            v = model.val(data=str(data_yaml),
                          imgsz=cfg["train"]["imgsz"], conf=ev["conf"],
                          iou=ev["iou"], max_det=ev["max_det"],
                          device=args.device, verbose=False, plots=False)
            metrics["ultralytics_val"] = {
                "mAP50": float(v.box.map50), "mAP50_95": float(v.box.map),
                "precision": float(v.box.mp), "recall": float(v.box.mr),
            }
            d50 = abs(v.box.map50 - m["mAP50"])
            metrics["cross_check_mAP50_abs_diff"] = round(float(d50), 4)
            if d50 > 0.02:
                metrics["cross_check_warning"] = (
                    f"custom mAP50 {m['mAP50']:.4f} vs ultralytics "
                    f"{v.box.map50:.4f} differ by {d50:.4f}; investigate before "
                    f"reporting")
        except Exception as e:  # noqa: BLE001
            metrics["ultralytics_val_error"] = str(e)

        print(f"\n  mAP50      {m['mAP50']:.4f}")
        print(f"  mAP50-95   {m['mAP50_95']:.4f}")
        print(f"  AP_small   {m['AP_small']:.4f}   (n_gt {m['n_gt']['small']:,})")
        print(f"  AP_medium  {m['AP_medium']:.4f}   (n_gt {m['n_gt']['medium']:,})")
        print(f"  AP_large   {m['AP_large']:.4f}   (n_gt {m['n_gt']['large']:,})")
        print(f"  latency    {timing['latency_ms_per_image']:.1f} ms/img")

    metrics["environment"] = environment_info()
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
