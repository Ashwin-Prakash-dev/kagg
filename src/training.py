# ---------------------------------------------------------------------
# ADAPTED COPY for Kaggle -- based on <project>/src/training.py.
# Difference from the local/Colab version: `build_train_args`/`train_one`
# take an explicit `data_yaml` path rather than a `dataset_root` + label
# convention, because the Kaggle upload nests each condition under its own
# top-level folder (see src/paths.py) instead of one common parent directory.
# No hyperparameter, control or research-design logic differs.
# ---------------------------------------------------------------------
"""
Training driver for one resolution condition.

The whole point of this module is that a run's hyperparameters cannot come
from anywhere except configs/master.yaml. `build_train_args` assembles the
Ultralytics keyword set from the config and the experiment's declared
overrides, and refuses any override that is not part of the experiment's
declared difference. That is the mechanical enforcement of Part D: it is not
possible to accidentally give one condition a different learning rate.

Every run also records the environment it happened in -- GPU model, VRAM,
library versions, wall-clock -- because "training time" and "GPU memory" are
dependent variables in this study and are meaningless without the hardware
they were measured on.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Keys an experiment is allowed to override. Anything else would make the
# conditions differ in something other than the independent variable.
ALLOWED_OVERRIDES = {"data", "name", "project", "seed"}

# Environment-level knobs `extra` may set: things that change how long/where
# THIS invocation runs, never what the model is trained on. "time" (hours)
# is Ultralytics' own wall-clock training cap -- it stops cleanly after a
# completed epoch once elapsed time exceeds it, letting a single condition's
# 50-epoch run be sliced across several Kaggle sessions that each stay well
# under the platform's session time limit.
ENV_KEYS = {"device", "workers", "batch", "amp", "cache", "project",
            "exist_ok", "resume", "epochs", "fraction", "plots", "val", "time"}


def load_master(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_train_args(cfg: dict, experiment: str, project: str | Path,
                     data_yaml: str | Path | None = None,
                     extra: dict | None = None) -> dict:
    """Assemble the exact Ultralytics kwargs for one experiment.

    `extra` is intended for environment-level knobs (device, workers, batch on
    a smaller GPU) -- never for hyperparameters. Anything outside
    ALLOWED_OVERRIDES plus that environment set raises, rather than silently
    breaking the experimental control.
    """
    exp = cfg["experiments"][experiment]
    args: dict = {"model": cfg["model"]["weights"], "pretrained": cfg["model"]["pretrained"]}
    args.update(cfg["train"])
    args.update(cfg["augmentation"])
    args["seed"] = cfg["reproducibility"]["seed"]
    args["deterministic"] = cfg["reproducibility"]["deterministic"]
    # max_det lives under `evaluation:`, not `train:`, but Ultralytics also
    # uses it for the validation pass it runs after every epoch (which
    # decides which checkpoint becomes best.pt) -- without this, that
    # in-training validation silently reverts to Ultralytics' own default of
    # 300, capping recall on the densest scenes and letting a lower-recall
    # epoch win "best" purely from an uncontrolled max_det, not model quality.
    args["max_det"] = cfg["evaluation"]["max_det"]

    overrides = dict(exp["overrides"])
    if data_yaml is not None:
        # Kaggle nests each condition under its own top-level input folder, so
        # the data.yaml path is resolved by src.paths.find_datasets_root() and
        # passed in directly, rather than rebuilt from a common root + label.
        overrides["data"] = str(data_yaml)
    bad = set(overrides) - ALLOWED_OVERRIDES
    if bad:
        raise ValueError(
            f"experiment {experiment} tries to override {sorted(bad)}, which "
            f"would make it differ from the other conditions in something "
            f"other than resolution. Allowed: {sorted(ALLOWED_OVERRIDES)}")
    args.update(overrides)

    if extra:
        bad = set(extra) - ENV_KEYS
        if bad:
            raise ValueError(f"refusing environment overrides {sorted(bad)}: "
                             f"these are experimental parameters and belong in "
                             f"configs/master.yaml")
        args.update(extra)

    args["project"] = str(project)
    args.setdefault("name", exp["id"])
    return args


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["gpu_total_memory_gb"] = round(p.total_memory / 1e9, 2)
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=15).strip()
    except Exception:  # noqa: BLE001 - absent GPU is a normal state
        pass
    return info


def peak_gpu_memory_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    return None


def count_completed_epochs(run_dir: Path) -> int:
    """Epochs Ultralytics actually finished, from its own per-epoch log.

    Reading `results.csv` (one row per completed epoch) rather than a
    trainer attribute avoids depending on that attribute's exact off-by-one
    semantics, and works identically whether this was a fresh run or a
    resume.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus the header row


def train_one(cfg: dict, experiment: str, project: Path,
              data_yaml: str | Path | None = None,
              extra: dict | None = None,
              resume_from: str | Path | None = None) -> dict:
    """Train a single condition and return a record of the run.

    `resume_from` continues an interrupted Kaggle session from its own
    `last.pt` rather than restarting at epoch 1 -- Ultralytics reloads that
    run's original arguments (data, imgsz, hyperparameters) from the
    `args.yaml` it wrote next to the checkpoint, so nothing here needs to be
    re-specified and nothing can drift from what the run started with.

    `extra["time"]` (hours), if given, is a wall-clock cap Ultralytics
    enforces itself (stopping cleanly after a completed epoch), so one
    invocation can be sized to fit inside a single Kaggle session. The
    record's `fully_trained` flag tells the caller whether the run actually
    reached the study's target epoch count or was cut short by that cap --
    only a `fully_trained` run should be evaluated and marked complete;
    a cut-short one should just save its checkpoint and exit, to be resumed
    by a later invocation.
    """
    from ultralytics import YOLO

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    # The epoch TARGET for "did this run actually finish" purposes -- must
    # respect an --epochs override (e.g. sanity's epochs=1), not always the
    # master config's 50, or a fully-completed 1-epoch sanity run would be
    # wrongly flagged as cut short.
    target_epochs = (extra or {}).get("epochs", cfg["train"]["epochs"])

    if resume_from is not None:
        set_seeds(cfg["reproducibility"]["seed"])
        model = YOLO(str(resume_from))
        resume_kwargs = {"resume": True}
        if extra and "time" in extra:
            # Ultralytics reloads the checkpoint's original args as a base
            # and layers any args passed here on top -- the same mechanism
            # used to extend a finished run's epoch count -- so each resumed
            # invocation gets its own fresh time budget rather than reusing
            # (or ignoring) the one recorded from the very first invocation.
            resume_kwargs["time"] = extra["time"]
        t0 = time.time()
        model.train(**resume_kwargs)
        elapsed = time.time() - t0
        used_epochs = target_epochs
        used_batch = cfg["train"]["batch"]
        used_imgsz = cfg["train"]["imgsz"]
        train_args_record = {"resumed_from": str(resume_from), **resume_kwargs}
    else:
        args = build_train_args(cfg, experiment, project, data_yaml, extra)
        set_seeds(args["seed"])
        weights = args.pop("model")
        model = YOLO(weights)
        t0 = time.time()
        model.train(**args)
        elapsed = time.time() - t0
        used_epochs, used_batch, used_imgsz = args["epochs"], args["batch"], args["imgsz"]
        train_args_record = {k: v for k, v in args.items() if not callable(v)}

    run_dir = Path(model.trainer.save_dir)
    completed_epochs = count_completed_epochs(run_dir)
    record = {
        "experiment": experiment,
        "experiment_id": cfg["experiments"][experiment]["id"],
        "resolution_label": cfg["experiments"][experiment]["label"],
        "scale": cfg["experiments"][experiment]["scale"],
        "run_dir": str(run_dir),
        "weights_best": str(run_dir / "weights" / "best.pt"),
        "weights_last": str(run_dir / "weights" / "last.pt"),
        "training_time_s": round(elapsed, 1),
        "training_time_h": round(elapsed / 3600, 3),
        "epochs": used_epochs,
        "batch": used_batch,
        "imgsz": used_imgsz,
        "seed": cfg["reproducibility"]["seed"],
        "resumed": resume_from is not None,
        "completed_epochs": completed_epochs,
        "target_epochs": target_epochs,
        "fully_trained": completed_epochs >= target_epochs,
        "peak_gpu_memory_gb": peak_gpu_memory_gb(),
        "environment": environment_info(),
        "train_args": train_args_record,
    }
    (run_dir / "run_record.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record
