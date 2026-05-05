"""
Visualization and figure generation for the research report.

Reads the summary and per-clip CSVs produced by evaluate.py and generates
publication-ready matplotlib figures:

1. Accuracy vs. Illumination (grouped bar / line chart)
   — X: lighting condition, Y: mean targeting error, series: method.
2. Trajectory Smoothing (2D floor-plan path plot)
   — Overlays GT path with each method's predicted path for a selected
     5-second window, showing relative jitter.
3. Qualitative Failure Modes (image grid)
   — Selects frames with highest error per method, renders the frame with
     the method's internal state (heatmap, flow field, bounding boxes) and
     the erroneous aim point annotated.

All figures are saved as both PNG (for the report) and PDF (for LaTeX).

Usage:
    python -m evaluation.visualize --results results/ --output figures/
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ORDER = ["ambient", "static_color", "slow_drift", "strobe", "moving_beam"]
CONDITION_LABELS = {
    "ambient": "Ambient",
    "static_color": "Static Color",
    "slow_drift": "Slow Drift",
    "strobe": "Strobe",
    "moving_beam": "Moving Beam",
}
METHOD_LABELS = {
    "frame_diff": "Frame Diff",
    "farneback": "Farneback Flow",
    "detection": "Detection Only",
    "hybrid": "Hybrid (Ours)",
}


def load_summary(summary_path: str) -> list[dict]:
    """Load the summary CSV from evaluate.py."""
    rows = []
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["mean_error_m"] = (float(row["mean_error_m"])
                                   if row["mean_error_m"] else None)
            row["jitter_m_per_sec"] = float(row["jitter_m_per_sec"])
            row["fps"] = float(row["fps"])
            rows.append(row)
    return rows


def load_per_clip_csv(path: str) -> list[dict]:
    """Load a per-clip result CSV."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def plot_accuracy_vs_illumination(summary: list[dict], output_dir: str):
    """
    Figure 1: Grouped bar chart — mean targeting error per condition per method.
    """
    methods = list(METHOD_LABELS.keys())
    conditions = [c for c in CONDITION_ORDER
                  if any(r["condition"] == c for r in summary)]

    n_conditions = len(conditions)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []
        for cond in conditions:
            matches = [r["mean_error_m"] for r in summary
                       if r["method"] == method and r["condition"] == cond
                       and r["mean_error_m"] is not None]
            values.append(float(np.mean(matches)) if matches else 0.0)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=METHOD_LABELS[method])

    ax.set_xlabel("Lighting Condition")
    ax.set_ylabel("Mean Targeting Error (m)")
    ax.set_title("Targeting Accuracy vs. Illumination Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"accuracy_vs_illumination.{ext}"),
                    dpi=150)
    plt.close(fig)
    print("  Saved: accuracy_vs_illumination.png/pdf")


def plot_jitter_comparison(summary: list[dict], output_dir: str):
    """
    Figure 1b: Grouped bar chart — jitter per condition per method.
    """
    methods = list(METHOD_LABELS.keys())
    conditions = [c for c in CONDITION_ORDER
                  if any(r["condition"] == c for r in summary)]

    n_conditions = len(conditions)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []
        for cond in conditions:
            matches = [r["jitter_m_per_sec"] for r in summary
                       if r["method"] == method and r["condition"] == cond]
            values.append(float(np.mean(matches)) if matches else 0.0)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=METHOD_LABELS[method])

    ax.set_xlabel("Lighting Condition")
    ax.set_ylabel("Jitter (m/s)")
    ax.set_title("Target Stability (Jitter) vs. Illumination Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"jitter_comparison.{ext}"), dpi=150)
    plt.close(fig)
    print("  Saved: jitter_comparison.png/pdf")


def plot_trajectory(results_dir: str, output_dir: str, clip_name: str,
                    start_frame: int = 0, num_frames: int = 150):
    """
    Figure 2: 2D trajectory plot on the floor plane.

    Overlays GT path with each method's predicted path over a time window.
    """
    methods = list(METHOD_LABELS.keys())
    fig, ax = plt.subplots(figsize=(8, 8))

    gt_plotted = False

    for method in methods:
        csv_path = os.path.join(results_dir, f"{clip_name}_{method}.csv")
        if not os.path.exists(csv_path):
            continue

        rows = load_per_clip_csv(csv_path)
        window = rows[start_frame:start_frame + num_frames]

        pred_x = [r["pred_x_m"] for r in window if r["pred_x_m"] is not None]
        pred_y = [r["pred_y_m"] for r in window if r["pred_y_m"] is not None]

        if pred_x:
            ax.plot(pred_x, pred_y, alpha=0.7, linewidth=1.2,
                    label=METHOD_LABELS[method])

        if not gt_plotted:
            gt_x = [r["gt_x_m"] for r in window if r["gt_x_m"] is not None]
            gt_y = [r["gt_y_m"] for r in window if r["gt_y_m"] is not None]
            if gt_x:
                ax.plot(gt_x, gt_y, "k-", linewidth=2.5, alpha=0.9,
                        label="Ground Truth")
                gt_plotted = True

    ax.set_xlabel("Floor X (m)")
    ax.set_ylabel("Floor Y (m)")
    ax.set_title(f"Target Trajectories — {clip_name} "
                 f"(frames {start_frame}–{start_frame + num_frames})")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"trajectory_{clip_name}.{ext}"),
                    dpi=150)
    plt.close(fig)
    print(f"  Saved: trajectory_{clip_name}.png/pdf")


def plot_fps_comparison(summary: list[dict], output_dir: str):
    """
    Supplementary figure: throughput bar chart per method.
    """
    methods = list(METHOD_LABELS.keys())

    fig, ax = plt.subplots(figsize=(7, 4))

    avg_fps = []
    labels = []
    for method in methods:
        fps_vals = [r["fps"] for r in summary if r["method"] == method]
        avg_fps.append(float(np.mean(fps_vals)) if fps_vals else 0.0)
        labels.append(METHOD_LABELS[method])

    bars = ax.bar(labels, avg_fps)
    ax.set_ylabel("Average FPS")
    ax.set_title("Method Throughput Comparison")
    ax.grid(axis="y", alpha=0.3)

    for bar, fps in zip(bars, avg_fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{fps:.0f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"fps_comparison.{ext}"), dpi=150)
    plt.close(fig)
    print("  Saved: fps_comparison.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results", type=str, default="results",
                        help="Directory containing evaluate.py output CSVs")
    parser.add_argument("--output", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--trajectory-clip", type=str, default=None,
                        help="Clip name for trajectory plot "
                             "(e.g., 'strobe_20260429_120000')")
    parser.add_argument("--trajectory-start", type=int, default=0,
                        help="Start frame for trajectory window")
    parser.add_argument("--trajectory-frames", type=int, default=150,
                        help="Number of frames in trajectory window")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    summary_path = os.path.join(args.results, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: summary.csv not found at {summary_path}")
        print("Run evaluation.evaluate first.")
        return

    summary = load_summary(summary_path)
    print(f"Loaded {len(summary)} summary entries.")
    print()

    print("Generating figures:")
    plot_accuracy_vs_illumination(summary, args.output)
    plot_jitter_comparison(summary, args.output)
    plot_fps_comparison(summary, args.output)

    if args.trajectory_clip:
        plot_trajectory(
            args.results, args.output,
            args.trajectory_clip,
            args.trajectory_start,
            args.trajectory_frames,
        )
    else:
        clips_in_summary = sorted(set(r["clip"] for r in summary))
        if clips_in_summary:
            plot_trajectory(args.results, args.output, clips_in_summary[0])

    print()
    print(f"All figures saved to: {args.output}/")


if __name__ == "__main__":
    main()
