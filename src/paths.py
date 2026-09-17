"""
Locate the prepared dataset and annotation layer inside a Kaggle session.

The Kaggle Dataset (`Fair1m_Ship_Dataset`) is already uploaded and attached by
the user. Kaggle mounts it read-only under /kaggle/input/<auto-slug>/..., and
the exact slug and the nesting of the four resolution folders depend on how
the dataset was uploaded (observed layout:
`ship_dataset_<cond>/datasets/<cond>/...`, but this is not assumed). Path
resolution is therefore done by *searching* for `data.yaml` files under
/kaggle/input and classifying each by its path, rather than by hard-coding
any username, slug or directory depth.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

KAGGLE_INPUT = Path(os.environ.get("FAIR1M_KAGGLE_INPUT", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("FAIR1M_KAGGLE_WORKING", "/kaggle/working"))

CONDITIONS = ("original", "r75", "r50", "r25")

# Directory names (case-insensitive, matched as a whole path component) that
# identify each condition. Covers the observed upload layout
# (`ship_dataset_r25/...`), the plain condition name, and the run-id form,
# so a re-upload with a slightly different layout still resolves.
CONDITION_DIR_NAMES: dict[str, set[str]] = {
    "original": {"original", "ship_dataset_original", "00_baseline", "baseline"},
    "r75": {"r75", "ship_dataset_r75", "01_r75"},
    "r50": {"r50", "ship_dataset_r50", "02_r50"},
    "r25": {"r25", "ship_dataset_r25", "03_r25"},
}

# Markers that identify a directory as a *results* bundle (this study's own
# output), used to find a previous run's Kaggle Notebook Output attached as a
# fresh input source so a new session can resume instead of restarting.
RESULT_MARKER_FILES = ("experiment_status.json", "master_results.json")
RESULT_MARKER_DIRS = ("00_baseline", "01_r75", "02_r50", "03_r25")


def _classify(data_yaml: Path) -> str | None:
    parts = {p.lower() for p in data_yaml.parts}
    for cond, names in CONDITION_DIR_NAMES.items():
        if any(n.lower() in parts for n in names):
            return cond
    return None


def find_datasets_root(explicit: str | Path | None = None,
                       search_root: str | Path | None = None) -> dict[str, Path]:
    """Return {condition: dataset_dir}, discovered by scanning for data.yaml.

    `dataset_dir` is the directory that directly holds `data.yaml`, `images/`
    and `labels/` for that condition. Unlike the local/Colab layout, this is
    NOT necessarily a common parent across conditions -- the Kaggle upload can
    nest each condition under its own top-level folder.
    """
    if explicit:
        p = Path(explicit)
        found = {c: p / c for c in CONDITIONS if (p / c / "data.yaml").exists()}
        if len(found) == len(CONDITIONS):
            return found

    root = Path(search_root) if search_root else KAGGLE_INPUT
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Attach the 'Fair1m_Ship_Dataset' Kaggle "
            f"Dataset to this notebook (Notebook -> Add Input) and rerun.")

    found: dict[str, Path] = {}
    all_yaml: list[Path] = []
    for yaml_path in sorted(root.rglob("data.yaml")):
        all_yaml.append(yaml_path)
        cond = _classify(yaml_path)
        if cond and cond not in found:
            found[cond] = yaml_path.parent

    missing = [c for c in CONDITIONS if c not in found]
    if missing:
        seen = "\n  ".join(str(p) for p in all_yaml) or "(none found)"
        raise FileNotFoundError(
            f"Could not find data.yaml for condition(s) {missing} under {root}.\n"
            f"data.yaml files found:\n  {seen}\n\n"
            f"Confirm 'Fair1m_Ship_Dataset' is attached in the notebook's Data "
            f"pane and contains all four resolution conditions {CONDITIONS}.")
    return found


def find_results_root(explicit: str | Path | None = None) -> Path:
    """Where experiment outputs go: always under /kaggle/working.

    /kaggle/working is NOT guaranteed to survive a fresh session, so this does
    not try to be clever about persistence -- that is handled separately by
    `find_previous_results` (restoring a prior Notebook Output) and by the
    notebook instructing the user to Save Version after each condition.
    """
    p = Path(explicit) if explicit else KAGGLE_WORKING / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_previous_results(search_root: str | Path | None = None) -> Path | None:
    """Find a previously-attached Kaggle Notebook Output of THIS study's
    results, if the user re-attached one after a session was interrupted.

    Searches /kaggle/input UNBOUNDED depth for this study's marker files --
    same as find_datasets_root()'s rglob("data.yaml"), and for the same
    reason: an attached input's actual nesting is not guaranteed to be
    shallow. (The attached ship dataset itself turned out to sit 4 levels
    down, at /kaggle/input/datasets/<owner>/<slug>/..., not directly under
    /kaggle/input/<slug>/ as first assumed.) An earlier version of this
    function only checked 2 levels deep and silently found nothing on a
    deeper-nested Notebook Output, which made every "resumed" session
    silently restart every condition from epoch 1 instead.
    """
    root = Path(search_root) if search_root else KAGGLE_INPUT
    if not root.exists():
        return None
    for marker_file in RESULT_MARKER_FILES:
        for hit in root.rglob(marker_file):
            if hit.is_file():
                return hit.parent
    for marker_dir in RESULT_MARKER_DIRS:
        for hit in root.rglob(marker_dir):
            if hit.is_dir() and (hit / "metrics.json").exists():
                return hit.parent
    return None


def restore_previous_results(results_root: Path,
                             search_root: str | Path | None = None) -> Path | None:
    """Copy a previously-found results bundle into the (fresh) working results
    root, so run_all.py's skip-if-metrics.json-exists logic resumes cleanly.

    Never overwrites a run that is already present locally (a mid-session
    rerun of this cell must not clobber progress made since the restore).
    """
    prev = find_previous_results(search_root)
    if prev is None:
        return None
    results_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in sorted(prev.iterdir()):
        dst = results_root / item.name
        if dst.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
        copied.append(item.name)
    return prev if copied or any(results_root.iterdir()) else None


def annotation_dir() -> Path:
    """The ship-only annotation layer shipped inside this package."""
    for p in (PACKAGE_ROOT / "data" / "ship_only",):
        if (p / "ship_annotations.parquet").exists():
            return p
    raise FileNotFoundError(
        f"ship_annotations.parquet not found under {PACKAGE_ROOT}/data/ship_only")


def ensure_writable_dataset(cond: str, src_dir: Path,
                            work_root: Path | None = None) -> Path:
    """Return a directory Ultralytics can train from for this condition.

    /kaggle/input is always read-only, but Ultralytics only ever *wants* to
    write a redundant `.cache` file next to `labels/`; it catches the
    resulting PermissionError and degrades gracefully (training still works)
    -- so by default this returns `src_dir` unchanged and reads directly from
    the attached dataset. The cost of that degradation is real, though: with
    nowhere to persist the cache, Ultralytics re-scans and verifies every
    image/label pair in `train:`/`val:` on every single launch (see
    `stage_sanity_subset`, which is why the sanity test stages a tiny subset
    instead of pointing at the full condition). This function copies -- ONLY
    this one condition, never all four -- solely as an explicit, opt-in
    fallback if a real training failure shows the read-only mount is not
    tolerated at all.
    """
    work_root = work_root or (KAGGLE_WORKING / "datasets")
    dst = work_root / cond
    if (dst / "data.yaml").exists():
        return dst  # already staged by a previous call/cell
    return src_dir


def stage_sanity_subset(label: str, dataset_dir: Path, n_train: int = 32,
                        n_val: int = 16, work_root: Path | None = None) -> Path:
    """Stage a tiny, ship-guaranteed subset for the GPU sanity test and
    return its root directory (holding data.yaml, images/, labels/).

    The sanity test's whole point is proving the plumbing works in a couple
    of minutes, not covering the data -- but `--fraction` does NOT make that
    cheap: Ultralytics scans and verifies EVERY image/label pair listed in
    `train:`/`val:` to build its label cache *before* `fraction` is ever
    applied, and (per `ensure_writable_dataset`) that cache can never be
    saved and reused on the read-only /kaggle/input mount. Left pointed at
    the full ~7,706-image dataset, a "5%" sanity run still pays the FULL
    scan cost -- across all four conditions, on every run -- which is the
    actual source of Step 6 being slow, not the one epoch of training.
    Copying a small, deterministic, ship-containing subset (not the whole
    dataset, and only for the one condition currently being staged) makes
    that scan trivial regardless of mount latency or image size.

    Images are picked from `data/ship_only/image_manifest.csv` (already
    bundled in this package) filtered to non-background images, so the
    subset is guaranteed to contain real ship annotations -- an empty-GT
    subset would make Step 6's own `n_gt > 0` pass/fail check meaningless.
    """
    import pandas as pd

    work_root = work_root or (KAGGLE_WORKING / "kaggle" / "_sanity")
    dst = work_root / label
    if (dst / "data.yaml").exists():
        return dst  # already staged by a previous cell/run

    images = pd.read_csv(annotation_dir() / "image_manifest.csv")
    for split_dir, split_col, n in (("train", "Train", n_train), ("val", "Val", n_val)):
        (dst / "images" / split_dir).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split_dir).mkdir(parents=True, exist_ok=True)
        cand = images[(images["Split"] == split_col) & (~images["is_background"])]
        for basename in cand["basename"].head(n):
            stem = Path(basename).stem
            img_src = dataset_dir / "images" / split_dir / f"{stem}.jpg"
            lbl_src = dataset_dir / "labels" / split_dir / f"{stem}.txt"
            if img_src.exists():
                shutil.copy2(img_src, dst / "images" / split_dir / img_src.name)
            if lbl_src.exists():
                shutil.copy2(lbl_src, dst / "labels" / split_dir / lbl_src.name)

    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: ship\n",
        encoding="utf-8")
    return dst


def resolve_data_yaml(label: str, dataset_dir: Path,
                      work_root: Path | None = None) -> Path:
    """Return a data.yaml Ultralytics can actually use for this condition.

    The `data.yaml` files inside the uploaded dataset were written by
    `scripts/create_resolution_datasets.py` on the machine that prepared
    them, and their `path:` field is that machine's own absolute path (e.g.
    a Windows drive path). That is correct locally, but on Kaggle it is a
    dead path -- and Ultralytics does not raise a clear error for it, it
    silently re-resolves the bogus `path:` against its own `datasets_dir`
    setting and looks for images somewhere like
    `/kaggle/working/datasets/<stale-path>/images/val`, which does not
    exist. `train`/`val`/`nc`/`names` are untouched and correct (they are
    relative to `path:`); only `path:` itself is stale.

    This rewrites just that field to the directory `find_datasets_root()`
    actually discovered, and writes the corrected file under
    `/kaggle/working` -- the original under `/kaggle/input` is read-only and
    is never modified.
    """
    import yaml

    work_root = work_root or (KAGGLE_WORKING / "kaggle" / "_data_yaml")
    work_root.mkdir(parents=True, exist_ok=True)
    d = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    d["path"] = str(dataset_dir)
    out = work_root / f"{label}.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return out


def is_experiment_complete(out_dir: Path) -> bool:
    """A finished, scored run: metrics.json has been written."""
    return (out_dir / "metrics.json").exists()


def find_resume_checkpoint(out_dir: Path, run_id: str) -> Path | None:
    """The checkpoint to resume from, if this run was interrupted mid-training.

    Ultralytics writes `weights/last.pt` after every epoch under
    `<out_dir>/train/<run_id>/`. If that exists but the run never finished
    (no metrics.json), training should continue from it rather than restart
    at epoch 1. Shared by run_experiment.py, run_all.py, and the setup
    notebook's own resume-status display, so all three agree on one
    definition.
    """
    if is_experiment_complete(out_dir):
        return None
    ckpt = out_dir / "train" / run_id / "weights" / "last.pt"
    return ckpt if ckpt.exists() else None


def stage_condition(cond: str, src_dir: Path, work_root: Path | None = None) -> Path:
    """Explicitly copy ONE condition's dataset to /kaggle/working. Fallback
    path only -- see `ensure_writable_dataset`. Never copies all four."""
    work_root = work_root or (KAGGLE_WORKING / "datasets")
    dst = work_root / cond
    if (dst / "data.yaml").exists():
        return dst
    print(f"  staging {cond}: {src_dir} -> {dst} ...")
    shutil.copytree(src_dir, dst)
    return dst
