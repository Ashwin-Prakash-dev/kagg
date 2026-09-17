"""
Aggregate the finished experiments into the study's result tables and plots
(Parts Q and U).

    python scripts/collect_results.py
    python scripts/collect_results.py --results-root /kaggle/working/results

Reads every `<results_root>/<run>/metrics.json` and writes:

    results/master_results.csv / .json
    results/resolution_comparison.csv
    results/object_size_comparison.csv
    results/training_cost_comparison.csv
    results/plots/*.png

Conditions that have not run are simply absent from the tables. They are never
filled in, interpolated or estimated -- Rule 6. The generated tables state
which conditions were present so a partial sweep cannot be mistaken for a
complete one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from src.paths import find_results_root

# Declaration order is also plot order.
RUN_ORDER = [("00_baseline", "original", 1.00),
             ("01_r75", "r75", 0.75),
             ("02_r50", "r50", 0.50),
             ("03_r25", "r25", 0.25)]

METRIC_COLS = ["mAP50", "mAP50_95", "precision", "recall", "f1",
               "AP_small", "AP_medium", "AP_large"]

BIN_COLOURS = {"small": "#d1495b", "medium": "#edae49", "large": "#00798c"}


def load_runs(results_root: Path, include_sanity: bool = False) -> pd.DataFrame:
    rows = []
    for run_id, label, scale in RUN_ORDER:
        for suffix in ([""] + (["_sanity"] if include_sanity else [])):
            p = results_root / (run_id + suffix) / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            m = d.get("metrics", {})
            run = d.get("run", {})
            env = d.get("environment", {})
            op = m.get("operating_point", {})
            rows.append({
                "run_id": run_id + suffix,
                "resolution": label,
                "scale": scale,
                "area_scale": round(scale ** 2, 4),
                "sanity": bool(suffix),
                "mAP50": m.get("mAP50"),
                "mAP50_95": m.get("mAP50_95"),
                "mAP75": m.get("mAP75"),
                "AR50_95": m.get("AR50_95"),
                "AP_small": m.get("AP_small"),
                "AP_medium": m.get("AP_medium"),
                "AP_large": m.get("AP_large"),
                "AP50_small": m.get("AP50_small"),
                "AP50_medium": m.get("AP50_medium"),
                "AP50_large": m.get("AP50_large"),
                "precision": op.get("precision"),
                "recall": op.get("recall"),
                "f1": op.get("f1"),
                "operating_conf": op.get("conf"),
                "n_detections": m.get("n_detections"),
                "n_gt_all": (m.get("n_gt") or {}).get("all"),
                "n_gt_small": (m.get("n_gt") or {}).get("small"),
                "n_gt_medium": (m.get("n_gt") or {}).get("medium"),
                "n_gt_large": (m.get("n_gt") or {}).get("large"),
                "training_time_h": run.get("training_time_h"),
                "training_time_s": run.get("training_time_s"),
                "peak_gpu_memory_gb": run.get("peak_gpu_memory_gb"),
                "latency_ms_per_image": (m.get("timing") or {}).get("latency_ms_per_image"),
                "images_per_s": (m.get("timing") or {}).get("images_per_s"),
                "epochs": run.get("epochs"),
                "batch": run.get("batch"),
                "imgsz": run.get("imgsz"),
                "seed": run.get("seed"),
                "gpu": env.get("gpu_name"),
                "gpu_memory_gb": env.get("gpu_total_memory_gb"),
                "ultralytics_mAP50": (d.get("ultralytics_val") or {}).get("mAP50"),
                "cross_check_mAP50_abs_diff": d.get("cross_check_mAP50_abs_diff"),
                "cross_check_warning": d.get("cross_check_warning"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_ord"] = df["resolution"].map({l: i for i, (_, l, _) in enumerate(RUN_ORDER)})
        df = df.sort_values(["sanity", "_ord"]).drop(columns="_ord").reset_index(drop=True)
    return df


def add_degradation(df: pd.DataFrame, baseline_label: str = "original") -> pd.DataFrame:
    """Percentage drop of each metric relative to the baseline condition."""
    out = df.copy()
    base = out[(out["resolution"] == baseline_label) & (~out["sanity"])]
    if base.empty:
        # Without a baseline there is nothing to express a drop against. Leaving
        # the columns empty is correct; inventing a reference would not be.
        for c in METRIC_COLS:
            out[f"{c}_drop_pct"] = np.nan
        return out
    b = base.iloc[0]
    for c in METRIC_COLS:
        bv = b.get(c)
        out[f"{c}_drop_pct"] = out[c].apply(
            lambda v, bv=bv: np.nan if (bv in (None, 0) or pd.isna(bv) or pd.isna(v))
            else round((bv - v) / bv * 100, 2))
    return out


def make_plots(df: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    d = df[~df["sanity"]].dropna(subset=["mAP50"])
    if d.empty:
        return []
    x = d["scale"] * 100
    written = []

    def single(col, ylabel, title, fname, colour="#00798c"):
        if d[col].isna().all():
            return
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(x, d[col], "o-", lw=2.2, color=colour, markersize=7)
        for xi, yi in zip(x, d[col]):
            if pd.notna(yi):
                ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("resolution (% of original)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=140)
        plt.close(fig)
        written.append(fname)

    single("mAP50", "mAP@50", "Resolution vs mAP@50", "resolution_vs_map.png")
    single("mAP50_95", "mAP@50:95", "Resolution vs mAP@50:95",
           "resolution_vs_map5095.png")
    single("recall", "recall", "Resolution vs recall", "resolution_vs_recall.png",
           "#3b7dd8")
    single("precision", "precision", "Resolution vs precision",
           "resolution_vs_precision.png", "#8e44ad")
    single("AP_small", "AP (small)", "Resolution vs small-object AP",
           "resolution_vs_small_ap.png", BIN_COLOURS["small"])
    single("AP_medium", "AP (medium)", "Resolution vs medium-object AP",
           "resolution_vs_medium_ap.png", BIN_COLOURS["medium"])
    single("AP_large", "AP (large)", "Resolution vs large-object AP",
           "resolution_vs_large_ap.png", BIN_COLOURS["large"])

    # Combined size view: the study's headline figure.
    if not d[["AP_small", "AP_medium", "AP_large"]].isna().all().all():
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        ax = axes[0]
        for b in ("small", "medium", "large"):
            ax.plot(x, d[f"AP_{b}"], "o-", lw=2.2, label=b, color=BIN_COLOURS[b])
        ax.plot(x, d["mAP50_95"], "k--", lw=1.6, label="overall", alpha=0.7)
        ax.set_xlabel("resolution (% of original)")
        ax.set_ylabel("AP@50:95")
        ax.set_title("AP by fixed original-resolution size bin")
        ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

        ax = axes[1]
        for b in ("small", "medium", "large"):
            col = f"AP_{b}_drop_pct"
            if col in d:
                ax.plot(x, d[col], "o-", lw=2.2, label=b, color=BIN_COLOURS[b])
        if "mAP50_95_drop_pct" in d:
            ax.plot(x, d["mAP50_95_drop_pct"], "k--", lw=1.6, label="overall",
                    alpha=0.7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("resolution (% of original)")
        ax.set_ylabel("% drop vs baseline")
        ax.set_title("Degradation relative to original resolution")
        ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "resolution_x_object_size.png", dpi=140)
        plt.close(fig)
        written.append("resolution_x_object_size.png")

    # Computational cost.
    cost = d.dropna(subset=["training_time_h"])
    if not cost.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, col, lbl in [
                (axes[0], "training_time_h", "training time (h)"),
                (axes[1], "latency_ms_per_image", "inference latency (ms/img)"),
                (axes[2], "peak_gpu_memory_gb", "peak GPU memory (GB)")]:
            if col in cost and not cost[col].isna().all():
                ax.bar(cost["resolution"], cost[col], color="#3b7dd8")
                for i, v in enumerate(cost[col]):
                    if pd.notna(v):
                        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel(lbl)
            ax.set_title(lbl)
        fig.suptitle("Computational cost by condition "
                     "(network input size held constant at imgsz=1024)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / "training_cost.png", dpi=140)
        plt.close(fig)
        written.append("training_cost.png")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out", default=None, help="output directory (default: <pkg>/results)")
    ap.add_argument("--include-sanity", action="store_true")
    args = ap.parse_args()

    results_root = find_results_root(args.results_root)
    out_dir = Path(args.out) if args.out else PKG / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(results_root, args.include_sanity)
    if df.empty:
        raise SystemExit(f"No metrics.json found under {results_root}. "
                         f"Run scripts/run_all.py first.")

    df = add_degradation(df)
    present = [r for r in df.loc[~df["sanity"], "resolution"]]
    missing = [l for _, l, _ in RUN_ORDER if l not in present]

    df.to_csv(out_dir / "master_results.csv", index=False)
    (out_dir / "master_results.json").write_text(
        json.dumps({
            "conditions_present": present,
            "conditions_missing": missing,
            "complete_sweep": not missing,
            "baseline": "original" if "original" in present else None,
            "size_bins": {"method": "COCO area thresholds on ORIGINAL-resolution boxes",
                          "small": "< 1024 px^2", "medium": "1024-9216 px^2",
                          "large": ">= 9216 px^2",
                          "fixed_across_resolutions": True},
            "runs": json.loads(df.to_json(orient="records")),
        }, indent=2), encoding="utf-8")

    # Part Q table.
    cmp_cols = ["resolution", "scale", "area_scale", "mAP50", "mAP50_95",
                "precision", "recall", "f1",
                "mAP50_drop_pct", "mAP50_95_drop_pct",
                "recall_drop_pct", "precision_drop_pct"]
    df.loc[~df["sanity"], [c for c in cmp_cols if c in df]].to_csv(
        out_dir / "resolution_comparison.csv", index=False)

    size_cols = ["resolution", "scale", "mAP50_95",
                 "AP_small", "AP_medium", "AP_large",
                 "AP_small_drop_pct", "AP_medium_drop_pct", "AP_large_drop_pct",
                 "n_gt_small", "n_gt_medium", "n_gt_large"]
    df.loc[~df["sanity"], [c for c in size_cols if c in df]].to_csv(
        out_dir / "object_size_comparison.csv", index=False)

    cost_cols = ["resolution", "scale", "training_time_h", "latency_ms_per_image",
                 "images_per_s", "peak_gpu_memory_gb", "gpu", "gpu_memory_gb",
                 "epochs", "batch", "imgsz", "seed"]
    df.loc[~df["sanity"], [c for c in cost_cols if c in df]].to_csv(
        out_dir / "training_cost_comparison.csv", index=False)

    plots = make_plots(df, out_dir / "plots")

    print(f"Conditions found : {present or 'none'}")
    if missing:
        print(f"Conditions MISSING: {missing}  <- tables are a partial sweep")
    warn = df[df["cross_check_warning"].notna()]
    if not warn.empty:
        print("\n!! metric cross-check disagreement:")
        for _, r in warn.iterrows():
            print(f"   {r['run_id']}: {r['cross_check_warning']}")

    show = ["resolution", "mAP50", "mAP50_95", "AP_small", "AP_medium",
            "AP_large", "recall", "precision"]
    print("\n" + df.loc[~df["sanity"], [c for c in show if c in df]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nWrote {len(plots)} plot(s) and 5 table(s) to {out_dir}")


if __name__ == "__main__":
    main()
