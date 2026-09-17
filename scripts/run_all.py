"""
Run the E00-E03 sweep on Kaggle, one condition after another.

    python scripts/run_all.py
    python scripts/run_all.py --order cheapest-first
    python scripts/run_all.py --sanity --epochs 1
    python scripts/run_all.py --time-budget 1.5

Each condition is an independent subprocess. That is deliberate: a crashed or
OOM-killed run cannot take the rest of the sweep with it, and CUDA memory is
returned cleanly between conditions instead of accumulating.

Conditions that already have a metrics.json are skipped, so re-running after a
Kaggle session interruption resumes the sweep rather than restarting it. A
condition that has a `last.pt` but no `metrics.json` resumes training from
that checkpoint (see run_experiment.py / src/training.py) instead of starting
at epoch 1.

--time-budget HOURS caps the TOTAL wall-clock time of this one invocation,
across however many conditions fit -- the intended way to run this on Kaggle,
where a full sweep (~12-20 GPU-hours) does not fit in one session. Before each
condition, the remaining budget is passed through as that condition's own
--time; once remaining time drops below a useful minimum, this stops (leaving
whatever's left for the next invocation) rather than starting a condition it
can't make meaningful progress on. Re-running the SAME command later resumes
exactly where this stopped: already-finished conditions are skipped, the
in-progress one resumes from its own checkpoint. This is meant to be commited
(Save Version -> Save & Run All) repeatedly, each commit doing one budget's
worth of work, until experiment_status.json shows everything done.

Before starting, this also looks for a previously-attached Kaggle Notebook
Output of this study's results under /kaggle/input (see
src/paths.restore_previous_results) and restores it into the results root, so
a fresh Kaggle session that reattaches its own prior output resumes instead of
starting over.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.paths import (find_resume_checkpoint, find_results_root,
                       is_experiment_complete, restore_previous_results)

# Declaration order = the order results are reported in.
CONDITIONS = [
    ("baseline", "00_baseline"),
    ("r75", "01_r75"),
    ("r50", "02_r50"),
    ("r25", "03_r25"),
]

# Below this much remaining budget, don't start (or resume) a condition --
# there isn't enough time left to make meaningful progress, so it's better to
# stop cleanly and leave it whole for the next invocation.
MIN_USEFUL_SLICE_HOURS = 0.1  # 6 minutes


def remaining_hours(deadline: float) -> float:
    return (deadline - time.time()) / 3600


def write_status(results_root: Path, summary: list[tuple[str, str, float]]) -> None:
    status = {name: {"status": s, "minutes": round(m, 1)} for name, s, m in summary}
    (results_root / "experiment_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order", choices=["declared", "cheapest-first"],
                    default="declared",
                    help="cheapest-first runs r25 -> original, so a truncated "
                         "session still yields a partial trend line")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--datasets-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage-to-working", action="store_true")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="total wall-clock hours for this whole invocation; "
                         "see module docstring. Ignored for --sanity (already "
                         "fast/fixed-epoch).")
    ap.add_argument("--no-restore", action="store_true",
                    help="skip searching /kaggle/input for a previous "
                         "Notebook Output to resume from")
    args = ap.parse_args()

    order = CONDITIONS if args.order == "declared" else list(reversed(CONDITIONS))
    results_root = find_results_root(args.results_root)

    if not args.no_restore and not args.sanity:
        prev = restore_previous_results(results_root)
        if prev:
            print(f"Restored previous results from {prev} -> {results_root}\n")

    passthrough: list[str] = []
    for flag, val in [("--datasets-root", args.datasets_root),
                      ("--results-root", str(results_root)),
                      ("--device", args.device), ("--batch", args.batch),
                      ("--workers", args.workers), ("--epochs", args.epochs),
                      ("--fraction", args.fraction)]:
        if val is not None:
            passthrough += [flag, str(val)]
    if args.sanity:
        passthrough.append("--sanity")
    if args.force:
        passthrough.append("--force")
    if args.stage_to_working:
        passthrough.append("--stage-to-working")

    deadline = (time.time() + args.time_budget * 3600
               if args.time_budget and not args.sanity else None)

    print(f"Sweep order: {' -> '.join(c[0] for c in order)}")
    print(f"Results root: {results_root}")
    if deadline:
        print(f"Time budget: {args.time_budget:.2f}h for this invocation")
    print()

    summary = []
    for cfg_name, run_id in order:
        name = run_id + ("_sanity" if args.sanity else "")
        out_dir = results_root / name
        if is_experiment_complete(out_dir) and not args.force:
            print(f"[skip] {name} already has metrics.json")
            summary.append((name, "skipped (already complete)", 0.0))
            write_status(results_root, summary)
            continue

        this_passthrough = list(passthrough)
        if deadline is not None:
            remaining = remaining_hours(deadline)
            if remaining <= MIN_USEFUL_SLICE_HOURS:
                print(f"[stop] time budget exhausted ({max(remaining, 0) * 60:.0f} "
                      f"min left, below the {MIN_USEFUL_SLICE_HOURS * 60:.0f}-min "
                      f"minimum) -- leaving {name} onward for the next run")
                for rest_cfg, rest_id in order[order.index((cfg_name, run_id)):]:
                    rest_name = rest_id + ("_sanity" if args.sanity else "")
                    summary.append((rest_name, "not attempted (time budget)", 0.0))
                write_status(results_root, summary)
                break
            print(f"[budget] {remaining:.2f}h remaining -> giving {name} up to that")
            this_passthrough += ["--time", f"{remaining:.3f}"]

        resume_ckpt = None if args.force else find_resume_checkpoint(out_dir, run_id)
        if resume_ckpt:
            print(f"[resume] {name} has a checkpoint at {resume_ckpt}; continuing "
                  f"from there rather than epoch 1.")

        cmd = [sys.executable, str(PKG / "scripts" / "run_experiment.py"),
               "--config", str(PKG / "configs" / f"{cfg_name}.yaml")] + this_passthrough
        print(f"\n{'#' * 70}\n# {name}\n# {' '.join(cmd)}\n{'#' * 70}", flush=True)

        t0 = time.time()
        rc = subprocess.call(cmd)
        el = (time.time() - t0) / 60
        if rc != 0:
            status = f"FAILED (rc={rc})"
        elif is_experiment_complete(out_dir):
            status = "ok"
        else:
            status = "in progress (time budget; resume next run)"
        summary.append((name, status, el))
        write_status(results_root, summary)
        if rc != 0:
            # Keep going: three good conditions beat none, and a partial sweep
            # is still reportable as long as the gap is stated.
            print(f"\n!! {name} exited with {rc}; continuing with the rest.\n")
        elif status.startswith("in progress"):
            # This invocation's budget is now certainly spent (the condition
            # itself used up its allotted --time); no point trying the next
            # condition with whatever sliver is nominally left.
            for rest_cfg, rest_id in order[order.index((cfg_name, run_id)) + 1:]:
                rest_name = rest_id + ("_sanity" if args.sanity else "")
                summary.append((rest_name, "not attempted (time budget)", 0.0))
            write_status(results_root, summary)
            break

    print(f"\n{'=' * 70}\nSWEEP SUMMARY\n{'=' * 70}")
    for name, status, el in summary:
        print(f"  {name:<20} {status:<32} {el:6.1f} min")
    failed = [s for s in summary if s[1].startswith("FAILED")]
    done = [s for s in summary if s[1] == "ok" or s[1].startswith("skipped")]
    print(f"\n{len(done)}/{len(CONDITIONS)} condition(s) complete.")
    remaining_work = [s for s in summary if not (s[1] == "ok" or s[1].startswith("skipped"))]
    if remaining_work and not failed:
        print("Not finished yet -- re-run this same command (or commit this "
              "notebook again) to continue from here.")
    elif len(done) == len(CONDITIONS):
        print(f"\nNext: python scripts/collect_results.py")
    print("\nRemember to Save Version (Save & Run All) so this progress "
          "persists as a Kaggle Notebook Output -- /kaggle/working is not "
          "guaranteed to survive a fresh session.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
