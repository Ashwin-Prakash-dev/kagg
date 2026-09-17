# FAIR1M ship detection — resolution sweep (Kaggle side)

Everything needed to run the GPU half of the study on Kaggle, against the
**already-uploaded** `Fair1m_Ship_Dataset` Kaggle Dataset. Local data
preparation, annotation validation and preprocessing are already done; this
package holds the code, the frozen configuration, and the small ship-only
annotation layer (~1.5 MB) that scores predictions. The four prepared
imagery datasets (~6.22 GB) are **not** duplicated here — they live in the
attached Kaggle Dataset and are read directly from `/kaggle/input`.

## The experiment in one paragraph

Four YOLOv8m detectors are trained on the *same* 7,706 FAIR1M images with the
*same* 58,982 ship annotations, differing only in the spatial resolution of
the imagery: 100%, 75%, 50% and 25% of native. Network input size is held
constant at `imgsz=1024` in every condition, so what varies is how much real
detail the imagery carries — not how large a tensor the network sees.
Performance is then broken down by object size, using bins fixed at original
resolution, to test whether small ships degrade faster than large ones.

## Quick start

Open `setup_kaggle.ipynb` on Kaggle, attach `Fair1m_Ship_Dataset`, enable a
GPU accelerator, turn Internet on, and **Run All**. No cell needs editing —
see the notebook's own top cell for the exact pre-flight checklist.

The notebook is intentionally self-contained: Step 0 embeds this whole
`kaggle/` code + annotation package (not the imagery) and writes it out to
`/kaggle/working/kaggle` at the start of every session, so opening the single
`.ipynb` is enough — nothing else needs to be uploaded separately.

## Layout

```text
kaggle/
  setup_kaggle.ipynb   the turnkey notebook (generated -- see below)
  configs/
    master.yaml        every hyperparameter in the study, in one file
    baseline.yaml       E00 - original    (scale 1.00)
    r75.yaml            E01 - 75%         (scale 0.75)
    r50.yaml             E02 - 50%         (scale 0.50)
    r25.yaml             E03 - 25%         (scale 0.25)
  scripts/
    run_all.py          the full sweep, one subprocess per condition, resumable
    run_experiment.py   train + evaluate one condition, resumable
    evaluate.py         score a checkpoint (incl. cross-resolution transfer)
    collect_results.py  aggregate into results/ tables and plots
    qualitative_comparison.py   matched scene comparisons across resolutions
    build_experiment_report.py  assemble results/EXPERIMENT_REPORT.md
    verify_kaggle_setup.py      pre-flight + self-check suite (LOCAL vs KAGGLE)
  src/
    paths.py            locates the attached dataset dynamically under
                         /kaggle/input, resolves the results root under
                         /kaggle/working, and restores a previous session's
                         Notebook Output if one is attached
    training.py          config-driven trainer; enforces the experimental
                         control and supports mid-run resume
    metrics.py            COCO AP with fixed original-resolution size bins
    evaluation.py          inference + mapping predictions back to original coords
    analysis.py             object-size definitions shared with the local pipeline
    visualization.py        figure helpers
  data/ship_only/       the annotation layer (parquet + manifest, ~1.5 MB)
  docs/                 size-bin justification, annotation cleaning report
```

`src/metrics.py`, `evaluation.py`, `analysis.py`, `visualization.py` and the
annotation layer are the same generated copies used by the Colab package
(`scripts/create_colab_package.py`'s source of truth); `paths.py` and
`training.py` are genuinely different from the Colab versions because
Kaggle's read-only, per-folder-nested dataset mount and ephemeral
`/kaggle/working` require different discovery and resume logic — the
research design, hyperparameters and controls are identical.

## Regenerating the notebook

`setup_kaggle.ipynb` is generated, not hand-edited, so it can never drift
from the scripts it embeds:

```bash
python scripts/build_kaggle_notebook.py   # from the project root
```

Edit files under `kaggle/{src,scripts,configs,docs,data/ship_only}`, then
regenerate. Hand-editing the notebook's bootstrap cells directly will be
overwritten.

## Kaggle-specific design notes

**Dataset discovery is dynamic, not hard-coded.** Kaggle assigns the mount
path under `/kaggle/input/<owner>-<slug>/...` at upload time, and the four
resolution folders may be nested under their own top-level directory (e.g.
`ship_dataset_r25/datasets/r25/...`). `src/paths.find_datasets_root()`
searches for `data.yaml` files and classifies each by its path components,
so it works regardless of the exact nesting depth or slug.

**The dataset is read directly, not copied.** `/kaggle/input` is read-only,
but Ultralytics only ever *wants* to write a redundant labels `.cache` file
next to `labels/`; it catches the resulting `PermissionError` and degrades
gracefully. Training therefore reads straight from `/kaggle/input` by
default. `src/paths.stage_condition()` is available as an explicit, opt-in
fallback that copies **one** condition — never all four — to
`/kaggle/working` if a real failure ever shows the read-only mount is not
tolerated.

**Resumability does not rely on `/kaggle/working` surviving a session.**
`run_all.py` skips any condition with a finished `metrics.json`, and resumes
any condition that has a `last.pt` but no `metrics.json` from that exact
checkpoint (`model.train(resume=True)`) rather than restarting at epoch 1.
Because `/kaggle/working` itself is not guaranteed to survive a fresh
session, the notebook also looks for a **previously-attached Kaggle
Notebook Output** of this study under `/kaggle/input`
(`src/paths.restore_previous_results`) and restores it before continuing —
so committing a version (Save & Run All) after each condition, then
reattaching that output in a new session, is enough to survive an
interruption without creating a new Kaggle Dataset.

## The controls, and why they matter

Identical to the Colab package: `imgsz=1024` in every condition (never scaled
with source resolution), `max_det=1000` (up from the Ultralytics default of
300 — the densest FAIR1M validation tile holds 581 ships), and
`src/training.py` refuses any experiment-level override outside
`{data, name, project, seed}`.

## Object-size bins

`small < 1024 px²`, `medium 1024–9216 px²`, `large ≥ 9216 px²` — COCO's
thresholds, measured on the **original-resolution** box and then **fixed**.
Full reasoning in `docs/object_size_definition.md`.

## Rules this package enforces

- An experiment directory holding a finished `metrics.json` is never
  overwritten without `--force`.
- Conditions that did not run are absent from the result tables, never
  interpolated.
- Cross-resolution evaluations must be given their own `--tag` so they
  cannot be mistaken for primary-sweep numbers.
- The study is single-seed unless the optional repeated-seed runs are
  actually executed. Do not claim significance from one run per condition.
