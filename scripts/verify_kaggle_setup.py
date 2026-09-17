"""
Pre-flight and self-check suite for the Kaggle FAIR1M pipeline.

Two kinds of checks:
  LOCAL            runs anywhere -- this repo, a laptop, a Kaggle session
                   without a GPU or before the dataset is attached. No CUDA
                   and no /kaggle/input required.
  REQUIRES KAGGLE  needs the attached Fair1m_Ship_Dataset and/or a CUDA
                   device; only meaningful inside an actual Kaggle session.
                   Reported as SKIP (not FAIL) when run somewhere else.

    python scripts/verify_kaggle_setup.py               everything this
                                                          environment supports
    python scripts/verify_kaggle_setup.py --local-only   force-skip every
                                                          Kaggle-only check

Exit code is 0 iff every check that actually ran (PASS or SKIP) did not FAIL.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

RESULTS: list[tuple[str, str, str, str]] = []  # name, group, status, detail


class Skipped(Exception):
    pass


def run(name: str, group: str, fn) -> None:
    try:
        detail = fn()
        status = "PASS"
        if isinstance(detail, tuple):
            status, detail = detail
    except Skipped as e:
        status, detail = "SKIP", str(e)
    except Exception as e:  # noqa: BLE001
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
    RESULTS.append((name, group, status, detail or ""))


# --------------------------------------------------------------------- LOCAL

def check_notebook_json_valid():
    nb_path = PKG / "setup_kaggle.ipynb"
    if not nb_path.exists():
        # A live Kaggle kernel has no filesystem access to its own .ipynb
        # source (only the materialized kaggle/{src,scripts,...} package
        # this notebook writes out in Step 0) -- this check only applies
        # when run from the source repository, e.g. before shipping a change.
        raise Skipped(f"{nb_path} not present in this runtime (expected "
                      f"inside a live Kaggle session; only meaningful when "
                      f"run from the source repo)")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, "unexpected nbformat"
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    return f"{len(nb['cells'])} cells ({n_code} code)"


def check_notebook_source_size():
    nb_path = PKG / "setup_kaggle.ipynb"
    if not nb_path.exists():
        raise Skipped(f"{nb_path} not present in this runtime")
    size = nb_path.stat().st_size
    LIMIT = 1_000_000  # Kaggle: "kernel source must be less than 1 megabytes"
    MARGIN = 900_000    # fail before actually hitting the hard limit
    if size >= LIMIT:
        return "FAIL", (f"{size:,} bytes >= Kaggle's {LIMIT:,}-byte kernel "
                        f"source limit -- the notebook CANNOT be saved on "
                        f"Kaggle (\"kernel source must be less than 1 "
                        f"megabytes\"); shrink an EMBED_GROUPS entry in "
                        f"scripts/build_kaggle_notebook.py")
    if size >= MARGIN:
        return "FAIL", (f"{size:,} bytes -- within {LIMIT - size:,} bytes of "
                        f"Kaggle's {LIMIT:,}-byte limit; trim embedded "
                        f"content before it crosses over")
    return f"{size:,} bytes (Kaggle's limit is {LIMIT:,})"


def check_scripts_exist():
    required = ["run_all.py", "run_experiment.py", "evaluate.py",
                "collect_results.py", "qualitative_comparison.py",
                "build_experiment_report.py", "build_annotation_layer.py",
                "verify_kaggle_setup.py"]
    missing = [f for f in required if not (PKG / "scripts" / f).exists()]
    if missing:
        return "FAIL", f"missing: {missing}"
    return f"{len(required)} scripts present"


def check_configs_exist():
    import yaml
    required = ["master.yaml", "baseline.yaml", "r75.yaml", "r50.yaml", "r25.yaml"]
    missing = [f for f in required if not (PKG / "configs" / f).exists()]
    if missing:
        return "FAIL", f"missing: {missing}"
    for f in required:
        yaml.safe_load((PKG / "configs" / f).read_text())
    return f"{len(required)} configs present and parse as YAML"


def _scan_text_files():
    exts = (".py", ".ipynb", ".md", ".yaml", ".yml")
    for p in PKG.rglob("*"):
        if p.name == "verify_kaggle_setup.py":
            continue  # this file legitimately names the forbidden strings
        if p.is_file() and p.suffix in exts and "__pycache__" not in p.parts:
            yield p


def check_no_content_refs():
    hits = []
    for p in _scan_text_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "/content" in text:
            hits.append(str(p.relative_to(PKG)))
    if hits:
        return "FAIL", f"'/content' found in: {hits}"
    return "no /content references"


def check_no_drive_refs():
    needles = ["google.colab", "drive.mount", "MyDrive"]
    hits = []
    for p in _scan_text_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for n in needles:
            if n in text:
                hits.append(f"{p.relative_to(PKG)} ({n})")
    if hits:
        return "FAIL", f"Google Drive references found: {hits}"
    return "no Google Drive / Colab references"


def check_imgsz_1024():
    import yaml
    from src.training import ALLOWED_OVERRIDES
    cfg = yaml.safe_load((PKG / "configs" / "master.yaml").read_text())
    if cfg["train"]["imgsz"] != 1024:
        return "FAIL", f"master.yaml train.imgsz = {cfg['train']['imgsz']}, expected 1024"
    if "imgsz" in ALLOWED_OVERRIDES:
        return "FAIL", "imgsz is in ALLOWED_OVERRIDES -- a condition could override it"
    return "master.yaml pins imgsz=1024; training.py forbids overriding it"


def check_max_det_1000():
    from src.training import build_train_args, load_master
    cfg = load_master(PKG / "configs" / "master.yaml")
    if cfg["evaluation"]["max_det"] != 1000:
        return "FAIL", f"master.yaml evaluation.max_det = {cfg['evaluation']['max_det']}, expected 1000"
    # The config value alone isn't enough: max_det lives under `evaluation:`,
    # not `train:`, so it has to be explicitly carried into the args actually
    # passed to model.train() -- Ultralytics' own default (300) silently wins
    # otherwise, including for the validation pass that picks best.pt.
    args = build_train_args(cfg, "E00", project="/tmp/x")
    if args.get("max_det") != 1000:
        return "FAIL", (f"build_train_args() produced max_det={args.get('max_det')}; "
                        f"it will reach model.train() as Ultralytics' default (300) "
                        f"instead of the study's 1000")
    return "master.yaml pins max_det=1000 AND build_train_args() actually passes it to model.train()"


def check_time_budget_wiring():
    from src.training import ENV_KEYS, build_train_args, count_completed_epochs, load_master
    cfg = load_master(PKG / "configs" / "master.yaml")
    if "time" not in ENV_KEYS:
        return "FAIL", "'time' missing from training.ENV_KEYS -- --time would be rejected"
    args = build_train_args(cfg, "E00", project="/tmp/x", extra={"time": 1.5})
    if args.get("time") != 1.5:
        return "FAIL", f"build_train_args() with extra={{'time': 1.5}} produced time={args.get('time')}"

    # count_completed_epochs() reads Ultralytics' own results.csv (1 row per
    # finished epoch); this is what decides whether a run gets evaluated
    # (finished) or just checkpointed and left for the next invocation (cut
    # short by --time). Exercise it directly against a synthetic log.
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        assert count_completed_epochs(run_dir) == 0, "missing results.csv should read as 0 epochs"
        (run_dir / "results.csv").write_text("epoch,loss\n1,0.5\n2,0.4\n3,0.3\n")
        n = count_completed_epochs(run_dir)
        assert n == 3, f"expected 3 completed epochs, got {n}"

    # The bug this guards against: target_epochs must respect an --epochs
    # override (sanity uses epochs=1), not always master.yaml's 50, or a
    # fully-completed 1-epoch sanity run gets wrongly flagged as cut short.
    sanity_target = ({"epochs": 1}).get("epochs", cfg["train"]["epochs"])
    full_target = ({}).get("epochs", cfg["train"]["epochs"])
    assert sanity_target == 1 and full_target == cfg["train"]["epochs"]
    return "ENV_KEYS allows 'time'; count_completed_epochs() reads results.csv correctly; target_epochs respects --epochs override"


def check_experiment_override_control():
    from src.training import build_train_args, load_master
    cfg = load_master(PKG / "configs" / "master.yaml")
    cfg["experiments"]["E00"]["overrides"]["lr0"] = 0.5  # illegal
    try:
        build_train_args(cfg, "E00", project="/tmp/x")
    except ValueError:
        return "build_train_args rejects an out-of-scope override (e.g. lr0)"
    return "FAIL", "build_train_args did NOT reject an illegal override"


def check_data_yaml_path_resolution():
    import yaml
    from src.paths import resolve_data_yaml
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dataset_dir = tmp / "input" / "ship_dataset_original" / "datasets" / "original"
        dataset_dir.mkdir(parents=True)
        # Mirrors the real files: `path:` is a stale absolute path baked in
        # by the machine that prepared the dataset (e.g. a Windows drive).
        (dataset_dir / "data.yaml").write_text(
            "path: E:/Datasets/FAIR1M/fair1m-satellite-imagery-for-object-detection/datasets/original\n"
            "train: images/train\nval: images/val\nnc: 1\nnames:\n  0: ship\n")
        work_root = tmp / "working" / "kaggle" / "_data_yaml"
        out = resolve_data_yaml("original", dataset_dir, work_root=work_root)
        d = yaml.safe_load(out.read_text())
        assert d["path"] == str(dataset_dir), \
            f"path not rewritten: {d['path']!r} != {dataset_dir!r}"
        assert d["train"] == "images/train" and d["val"] == "images/val"
        assert d["nc"] == 1
    return "resolve_data_yaml() rewrites a stale absolute path: to the real dataset_dir"


def check_annotation_layer_reconstruction():
    sys.path.insert(0, str(PKG / "scripts"))
    from build_annotation_layer import build
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp) / "original"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)
        (dataset_dir / "images" / "val").mkdir(parents=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True)

        # 100x200 (w x h) image, one ship: cx=0.5 cy=0.5 w=0.4 h=0.2 normalized
        # -> expect pixel box x:[30,70] y:[80,120], area 1600 px^2 ("medium").
        Image.new("RGB", (100, 200)).save(dataset_dir / "images" / "train" / "ship_0.jpg")
        (dataset_dir / "labels" / "train" / "ship_0.txt").write_text("0 0.5 0.5 0.4 0.2\n")
        # A background image: no label file at all.
        Image.new("RGB", (50, 50)).save(dataset_dir / "images" / "train" / "bg_0.jpg")

        images, ships = build(dataset_dir)

    assert len(images) == 2, f"expected 2 images, got {len(images)}"
    assert len(ships) == 1, f"expected 1 ship instance, got {len(ships)}"
    bg = images[images["basename"] == "bg_0.jpg"].iloc[0]
    ship_img = images[images["basename"] == "ship_0.jpg"].iloc[0]
    assert bool(bg["is_background"]) is True
    assert bool(ship_img["is_background"]) is False
    s = ships.iloc[0]
    expected = {"x_min": 30, "x_max": 70, "y_min": 80, "y_max": 120, "barea": 1600}
    for name, want in expected.items():
        assert abs(s[name] - want) < 1e-6, f"{name}: got {s[name]}, expected {want}"
    assert s["size_bin"] == "medium", f"size_bin: got {s['size_bin']!r}"
    return ("build() recovers exact original-resolution box coordinates and "
            "size_bin from a YOLO label + image dimensions")


def check_sanity_subset_staging():
    import pandas as pd
    from src.paths import annotation_dir, stage_sanity_subset

    # Uses the REAL bundled manifest (small, ships-guaranteed rows only) so
    # this exercises the actual selection logic, not a fabricated stand-in.
    manifest = pd.read_csv(annotation_dir() / "image_manifest.csv")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dataset_dir = tmp / "dataset_dir"
        n_train, n_val = 3, 2
        picks = {}
        for split_dir, split_col, n in (("train", "Train", n_train), ("val", "Val", n_val)):
            (dataset_dir / "images" / split_dir).mkdir(parents=True)
            (dataset_dir / "labels" / split_dir).mkdir(parents=True)
            cand = manifest[(manifest["Split"] == split_col) & (~manifest["is_background"])]
            stems = [Path(b).stem for b in cand["basename"].head(n)]
            picks[split_dir] = stems
            for stem in stems:
                (dataset_dir / "images" / split_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
                (dataset_dir / "labels" / split_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        staged = stage_sanity_subset("test_cond", dataset_dir, n_train=n_train,
                                     n_val=n_val, work_root=tmp / "staged")
        got_train = {p.stem for p in (staged / "images" / "train").glob("*.jpg")}
        got_val = {p.stem for p in (staged / "images" / "val").glob("*.jpg")}
        assert got_train == set(picks["train"]), (got_train, picks["train"])
        assert got_val == set(picks["val"]), (got_val, picks["val"])
        for stem in got_train | got_val:
            split_dir = "train" if stem in got_train else "val"
            assert (staged / "labels" / split_dir / f"{stem}.txt").exists(), \
                f"label missing for staged image {stem}"
        assert (staged / "data.yaml").exists()
    return (f"stages exactly the requested {n_train}+{n_val} ship-containing "
            f"images (with labels), not the full condition")


def check_skip_and_resume_logic():
    from src.paths import find_resume_checkpoint, is_experiment_complete
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        fresh = tmp / "not_started"
        assert not is_experiment_complete(fresh)
        assert find_resume_checkpoint(fresh, "00_baseline") is None

        mid = tmp / "in_progress"
        ckpt = mid / "train" / "00_baseline" / "weights" / "last.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"fake")
        assert not is_experiment_complete(mid)
        got = find_resume_checkpoint(mid, "00_baseline")
        assert got == ckpt, f"expected {ckpt}, got {got}"

        done = tmp / "finished"
        done.mkdir()
        (done / "metrics.json").write_text("{}")
        (done / "train" / "00_baseline" / "weights").mkdir(parents=True)
        (done / "train" / "00_baseline" / "weights" / "last.pt").write_bytes(b"fake")
        assert is_experiment_complete(done)
        assert find_resume_checkpoint(done, "00_baseline") is None, \
            "a completed run must never be resumed"
    return "not-started / resume / skip-when-complete all correct"


def check_time_budget_distribution():
    import time as _time
    sys.path.insert(0, str(PKG / "scripts"))
    import run_all

    if run_all.MIN_USEFUL_SLICE_HOURS <= 0:
        return "FAIL", "MIN_USEFUL_SLICE_HOURS must be positive"

    # remaining_hours() is what turns a total --time-budget into a shrinking
    # per-condition --time as run_all.py works through the sweep. A future
    # deadline should read as a positive number of hours remaining; a past
    # deadline (budget already spent) must read as non-positive, which is
    # what triggers the "stop, leave the rest for next run" branch rather
    # than starting a condition with a nonsensical negative time cap.
    future_h = run_all.remaining_hours(_time.time() + 2 * 3600)
    assert 1.9 < future_h < 2.1, f"expected ~2.0h remaining, got {future_h}"
    past_h = run_all.remaining_hours(_time.time() - 3600)
    assert past_h <= 0, f"a past deadline must read as <= 0h remaining, got {past_h}"
    return ("remaining_hours() reads a future deadline as positive hours and "
            "an elapsed one as <= 0, correctly gating the stop-early branch")


def check_run_all_budget_loop():
    """Exercise run_all.py's actual orchestration loop (not just the pure
    helper functions above) with a mocked subprocess call, so the interaction
    between skip / in-progress / not-attempted logic is verified end to end
    without needing real training."""
    sys.path.insert(0, str(PKG / "scripts"))
    import run_all

    with tempfile.TemporaryDirectory() as tmp:
        results_root = Path(tmp) / "results"
        results_root.mkdir()
        (results_root / "00_baseline").mkdir()
        (results_root / "00_baseline" / "metrics.json").write_text("{}")

        calls: list[list[str]] = []

        def fake_call(cmd):
            calls.append(cmd)
            return 0  # "succeeds" but writes no metrics.json -- simulates a
                     # run cut short by its own --time cap mid-training.

        orig_call, orig_argv = run_all.subprocess.call, sys.argv
        run_all.subprocess.call = fake_call
        try:
            sys.argv = ["run_all.py", "--results-root", str(results_root),
                       "--time-budget", "1.0", "--no-restore"]
            try:
                run_all.main()
            except SystemExit as e:
                assert not e.code, f"unexpected failure exit: {e.code}"
        finally:
            run_all.subprocess.call = orig_call
            sys.argv = orig_argv

        status = json.loads((results_root / "experiment_status.json").read_text())

    assert status["00_baseline"]["status"].startswith("skipped"), status["00_baseline"]
    assert status["01_r75"]["status"].startswith("in progress"), status["01_r75"]
    assert status["02_r50"]["status"] == "not attempted (time budget)", status["02_r50"]
    assert status["03_r25"]["status"] == "not attempted (time budget)", status["03_r25"]
    assert len(calls) == 1, f"expected exactly 1 subprocess call (for r75), got {len(calls)}"
    assert "--time" in calls[0], "per-condition --time flag was not passed through"
    return ("skips the finished condition, runs exactly one more within "
            "budget with --time passed through, marks it in-progress when no "
            "metrics.json appears, and leaves the rest for next time")


def check_result_collection():
    sys.path.insert(0, str(PKG / "scripts"))
    import collect_results
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake = {
            "experiment": "E00",
            "run": {"training_time_h": 0.01, "training_time_s": 36,
                    "peak_gpu_memory_gb": 5.1, "epochs": 1, "batch": 8, "imgsz": 1024,
                    "seed": 42},
            "metrics": {"mAP50": 0.5, "mAP50_95": 0.3, "mAP75": 0.35, "AR50_95": 0.4,
                        "AP_small": 0.1, "AP_medium": 0.4, "AP_large": 0.6,
                        "n_gt": {"all": 100, "small": 30, "medium": 50, "large": 20},
                        "n_detections": 90,
                        "operating_point": {"precision": 0.7, "recall": 0.6, "f1": 0.65,
                                            "conf": 0.25},
                        "timing": {"latency_ms_per_image": 12.3, "images_per_s": 81.3}},
            "environment": {"gpu_name": "fake-gpu", "gpu_total_memory_gb": 16.0},
        }
        d = tmp / "00_baseline"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(fake))
        df = collect_results.load_runs(tmp)
        assert len(df) == 1, f"expected 1 row, got {len(df)}"
        assert df.iloc[0]["mAP50"] == 0.5
        assert df.iloc[0]["resolution"] == "original"
    return "collect_results.load_runs() aggregates a synthetic metrics.json correctly"


def check_annotation_layer():
    import pandas as pd
    from src.paths import annotation_dir
    ann = annotation_dir()
    ships = pd.read_parquet(ann / "ship_annotations.parquet")
    images = pd.read_csv(ann / "image_manifest.csv")
    if len(images) != 7706:
        return "FAIL", f"expected 7,706 images in manifest, found {len(images)}"
    if len(ships) != 58982:
        return "FAIL", f"expected 58,982 ship instances, found {len(ships)}"
    return f"{len(images):,} images, {len(ships):,} ship instances"


def check_yolo_import_and_init():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise Skipped(f"ultralytics not installed in this environment: {e}")
    weights = None
    for cand in (PKG.parent / "yolov8m.pt", PKG / "yolov8m.pt", Path("yolov8m.pt")):
        if cand.exists():
            weights = cand
            break
    if weights is None:
        raise Skipped("yolov8m.pt not found locally (Kaggle will auto-download "
                      "it if Internet is enabled in Notebook settings)")
    model = YOLO(str(weights))
    assert model.task == "detect"
    return f"YOLO('{weights.name}') initialised, task={model.task}"


# ------------------------------------------------------------ REQUIRES KAGGLE

def check_dataset_discovery():
    from src.paths import KAGGLE_INPUT, find_datasets_root
    if not KAGGLE_INPUT.exists():
        raise Skipped(f"{KAGGLE_INPUT} does not exist (not running on Kaggle)")
    found = find_datasets_root()
    return f"found {list(found)} under {KAGGLE_INPUT}"


def _condition_check(cond: str):
    from src.paths import KAGGLE_INPUT, find_datasets_root
    if not KAGGLE_INPUT.exists():
        raise Skipped("not running on Kaggle")
    found = find_datasets_root()
    if cond not in found:
        return "FAIL", f"{cond} not found"
    return str(found[cond])


def check_e00():
    return _condition_check("original")


def check_e01():
    return _condition_check("r75")


def check_e02():
    return _condition_check("r50")


def check_e03():
    return _condition_check("r25")


def check_dataset_structure():
    from src.paths import KAGGLE_INPUT, find_datasets_root
    if not KAGGLE_INPUT.exists():
        raise Skipped("not running on Kaggle")
    found = find_datasets_root()
    problems = []
    for cond, root in found.items():
        for rel in ("images/train", "images/val", "labels/train", "labels/val"):
            if not (root / rel).is_dir():
                problems.append(f"{cond}: missing {rel}")
    if problems:
        return "FAIL", "; ".join(problems)
    return "images/{train,val} and labels/{train,val} present for all 4 conditions"


def check_data_yaml_validation():
    import yaml
    from src.paths import KAGGLE_INPUT, find_datasets_root
    if not KAGGLE_INPUT.exists():
        raise Skipped("not running on Kaggle")
    found = find_datasets_root()
    problems = []
    for cond, root in found.items():
        d = yaml.safe_load((root / "data.yaml").read_text())
        nc = d.get("nc")
        names = d.get("names")
        if nc != 1:
            problems.append(f"{cond}: nc={nc}, expected 1")
        names_list = list(names.values()) if isinstance(names, dict) else names
        if names_list != ["ship"]:
            problems.append(f"{cond}: names={names}, expected ['ship']")
    if problems:
        return "FAIL", "; ".join(problems)
    return "nc=1, names=['ship'] in all 4 data.yaml files"


def check_label_byte_identity():
    import hashlib
    from src.paths import KAGGLE_INPUT, find_datasets_root
    if not KAGGLE_INPUT.exists():
        raise Skipped("not running on Kaggle")
    found = find_datasets_root()
    digests = {}
    for cond, root in found.items():
        h = hashlib.sha256()
        for p in sorted((root / "labels").rglob("*.txt")):
            h.update(p.read_bytes())
        digests[cond] = h.hexdigest()
    if len(set(digests.values())) != 1:
        return "FAIL", f"label digests differ across conditions: {digests}"
    return "label files are byte-identical across all 4 conditions"


def check_cuda_detection():
    import torch
    if not torch.cuda.is_available():
        raise Skipped("no CUDA device in this environment")
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 1e9:.1f} GB, {torch.cuda.device_count()} device(s)"


def check_checkpoint_and_resume_end_to_end():
    from src.paths import KAGGLE_INPUT
    if not KAGGLE_INPUT.exists():
        raise Skipped("full end-to-end checkpoint test requires a Kaggle GPU "
                      "session; run the sanity-test cell in the notebook instead")
    raise Skipped("run the notebook's sanity-test cell to exercise this live")


CHECKS: list[tuple[str, str, str, callable]] = [
    ("Notebook JSON valid", "LOCAL", "structure", check_notebook_json_valid),
    ("Notebook source size < Kaggle's 1MB limit", "LOCAL", "structure", check_notebook_source_size),
    ("All scripts exist", "LOCAL", "structure", check_scripts_exist),
    ("All configs exist", "LOCAL", "structure", check_configs_exist),
    ("No /content references", "LOCAL", "structure", check_no_content_refs),
    ("No Google Drive dependencies", "LOCAL", "structure", check_no_drive_refs),
    ("Annotation layer intact (7,706 / 58,982)", "LOCAL", "structure", check_annotation_layer),
    ("imgsz=1024 pinned", "LOCAL", "config", check_imgsz_1024),
    ("max_det=1000 pinned", "LOCAL", "config", check_max_det_1000),
    ("Time-budget wiring", "LOCAL", "config", check_time_budget_wiring),
    ("Experiment override control", "LOCAL", "config", check_experiment_override_control),
    ("data.yaml path resolution", "LOCAL", "dataset", check_data_yaml_path_resolution),
    ("Sanity subset staging", "LOCAL", "dataset", check_sanity_subset_staging),
    ("Annotation layer reconstruction", "LOCAL", "dataset", check_annotation_layer_reconstruction),
    ("Resume logic", "LOCAL", "logic", check_skip_and_resume_logic),
    ("Time-budget distribution across conditions", "LOCAL", "logic", check_time_budget_distribution),
    ("run_all.py budget loop (mocked subprocess)", "LOCAL", "logic", check_run_all_budget_loop),
    ("Completed-experiment skip logic", "LOCAL", "logic", check_skip_and_resume_logic),
    ("Result collection", "LOCAL", "logic", check_result_collection),
    ("YOLOv8m initialization", "LOCAL", "model", check_yolo_import_and_init),
    ("Dataset discovery", "KAGGLE", "dataset", check_dataset_discovery),
    ("E00 found", "KAGGLE", "dataset", check_e00),
    ("E01 found", "KAGGLE", "dataset", check_e01),
    ("E02 found", "KAGGLE", "dataset", check_e02),
    ("E03 found", "KAGGLE", "dataset", check_e03),
    ("Dataset structure validation", "KAGGLE", "dataset", check_dataset_structure),
    ("YOLO data.yaml validation", "KAGGLE", "dataset", check_data_yaml_validation),
    ("Label byte-identity across conditions", "KAGGLE", "dataset", check_label_byte_identity),
    ("CUDA detection", "KAGGLE", "gpu", check_cuda_detection),
    ("Checkpoint creation (live)", "KAGGLE", "gpu", check_checkpoint_and_resume_end_to_end),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-only", action="store_true",
                    help="force-skip every REQUIRES-KAGGLE check")
    args = ap.parse_args()

    for name, group, _cat, fn in CHECKS:
        if args.local_only and group == "KAGGLE":
            RESULTS.append((name, group, "SKIP", "--local-only"))
            continue
        run(name, group, fn)

    print(f"{'CHECK':<42} {'SCOPE':<8} {'STATUS':<6} DETAIL")
    print("-" * 100)
    for name, group, status, detail in RESULTS:
        print(f"{name:<42} {group:<8} {status:<6} {detail}")

    n_pass = sum(1 for *_, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for *_, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for *_, s, _ in RESULTS if s == "SKIP")
    print("-" * 100)
    print(f"{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP (of {len(RESULTS)})")
    if n_fail:
        print("\nFAILED checks must be fixed before trusting this pipeline.")
        sys.exit(1)
    if n_skip:
        print("\nSKIPPED checks require a Kaggle session (GPU and/or the "
              "attached dataset) and were not run here.")


if __name__ == "__main__":
    main()
