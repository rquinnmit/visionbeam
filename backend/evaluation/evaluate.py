"""
Evaluation harness.

Runs each target-selection method (from evaluation.methods) on every
recorded clip, logs per-frame predictions to CSV, then computes metrics
by comparing predictions against ground truth. Outputs:

- Per-clip CSVs: frame, pred_x_px, pred_y_px, gt_x_px, gt_y_px, error_px
- Summary CSV:   clip, condition, method, mean_error_px,
                 jitter_px_per_sec, fps

All comparisons happen in pixel space. The deployed control loop maps
target pixels to fixture pan/tilt via the operator-fitted quadratic
calibration in visionbeam.aim; that mapping is irrelevant to whether one
target-selection method is more accurate than another, so the harness
stays in pixel space and reports raw Euclidean distance between predicted
and ground-truth marker pixels.

Metrics computed:
1. Targeting Accuracy — mean Euclidean distance (pixels) between predicted
   and ground-truth marker locations.
2. Target Stability (Jitter) — total path length of predictions per second.
3. Robustness Drop-off — accuracy ratio between each condition and baseline.
4. Throughput — average FPS per method on the evaluation hardware.

Usage:
    python -m evaluation.evaluate --clips data/clips/ --gt data/gt/ \
                                  --output results/
"""

import argparse
import csv
import json
import math
import os
import time

import cv2
import numpy as np

from visionbeam.tracker import HybridMethod
from evaluation.methods import (
    TargetMethod,
    FrameDiffMethod,
    FarnebackFlowMethod,
    DetectionMethod,
)


def build_methods() -> dict[str, TargetMethod]:
    """Instantiate all methods to evaluate.

    Hybrid runs with snap_to_feet=False so the evaluation compares the raw
    motion-heatmap peak against the other methods' pixel outputs (which all
    return body-located points). Deployment still uses snap_to_feet=True.
    """
    return {
        "frame_diff": FrameDiffMethod(),
        "farneback": FarnebackFlowMethod(),
        "detection": DetectionMethod(),
        "hybrid": HybridMethod(snap_to_feet=False),
    }


def load_ground_truth(gt_path: str) -> dict[int, tuple[float, float]]:
    """Load a GT CSV into a dict mapping frame number to (x_px, y_px)."""
    gt = {}
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[int(row["frame"])] = (float(row["x_px"]), float(row["y_px"]))
    return gt


def evaluate_clip(
    video_path: str,
    gt: dict[int, tuple[float, float]],
    method: TargetMethod,
) -> tuple[list[dict], float]:
    """
    Run a single method on a video clip and compute per-frame metrics.

    Returns (per-frame results list, average FPS).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    method.reset()
    results = []
    frame_num = 0
    total_process_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        pred_px = method.process_frame(frame)
        total_process_time += time.perf_counter() - t0

        gt_px = gt.get(frame_num)

        error_px = None
        if pred_px is not None and gt_px is not None:
            error_px = math.hypot(
                pred_px[0] - gt_px[0],
                pred_px[1] - gt_px[1],
            )

        results.append({
            "frame": frame_num,
            "pred_x_px": pred_px[0] if pred_px else None,
            "pred_y_px": pred_px[1] if pred_px else None,
            "gt_x_px": gt_px[0] if gt_px else None,
            "gt_y_px": gt_px[1] if gt_px else None,
            "error_px": error_px,
        })

        frame_num += 1

    cap.release()

    avg_fps = frame_num / total_process_time if total_process_time > 0 else 0.0
    return results, avg_fps


def compute_summary(
    results: list[dict],
    avg_fps: float,
    clip_name: str,
    condition: str,
    method_name: str,
    video_fps: float,
) -> dict:
    """Compute aggregate metrics from per-frame results."""
    errors = [r["error_px"] for r in results if r["error_px"] is not None]
    mean_error = float(np.mean(errors)) if errors else None

    path_length = 0.0
    prev = None
    for r in results:
        if r["pred_x_px"] is not None:
            cur = (r["pred_x_px"], r["pred_y_px"])
            if prev is not None:
                path_length += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            prev = cur

    duration_sec = len(results) / video_fps if video_fps > 0 else 1.0
    jitter = path_length / duration_sec if duration_sec > 0 else 0.0

    return {
        "clip": clip_name,
        "condition": condition,
        "method": method_name,
        "mean_error_px": round(mean_error, 4) if mean_error is not None else None,
        "jitter_px_per_sec": round(jitter, 4),
        "fps": round(avg_fps, 1),
    }


def save_per_clip_csv(results: list[dict], output_path: str):
    """Write per-frame results for one method+clip to CSV."""
    fieldnames = [
        "frame", "pred_x_px", "pred_y_px", "gt_x_px", "gt_y_px", "error_px",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items()}
            writer.writerow(row)


def find_clip_pairs(clips_dir: str, gt_dir: str) -> list[dict]:
    """
    Match video clips with their ground truth CSVs.

    Expects clip files named like 'external_static_20260429_120000.mp4'
    with sidecar JSON 'external_static_20260429_120000.json' and GT files
    named 'external_static_20260429_120000_gt.csv'.
    """
    pairs = []
    for fname in sorted(os.listdir(clips_dir)):
        if not fname.endswith(".mp4"):
            continue

        base = os.path.splitext(fname)[0]
        meta_path = os.path.join(clips_dir, f"{base}.json")
        gt_path = os.path.join(gt_dir, f"{base}_gt.csv")

        if not os.path.exists(gt_path):
            print(f"  Warning: No GT found for {fname}, skipping.")
            continue

        condition = "unknown"
        video_fps = 30.0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            condition = meta.get("condition", "unknown")
            video_fps = meta.get("fps", 30.0)

        pairs.append({
            "video_path": os.path.join(clips_dir, fname),
            "gt_path": gt_path,
            "clip_name": base,
            "condition": condition,
            "video_fps": video_fps,
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--clips", type=str, default="data/clips",
                        help="Directory containing video clips and metadata JSONs")
    parser.add_argument("--gt", type=str, default="data/gt",
                        help="Directory containing ground truth CSVs")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for result CSVs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    methods = build_methods()
    pairs = find_clip_pairs(args.clips, args.gt)

    if not pairs:
        print("No clip/GT pairs found. Check --clips and --gt paths.")
        return

    print(f"Found {len(pairs)} clips, evaluating {len(methods)} methods each.")
    print()

    all_summaries = []

    for pair in pairs:
        print(f"Clip: {pair['clip_name']} ({pair['condition']})")
        gt = load_ground_truth(pair["gt_path"])

        for method_name, method in methods.items():
            print(f"  Method: {method_name} ...", end=" ", flush=True)

            results, avg_fps = evaluate_clip(
                pair["video_path"], gt, method
            )

            per_clip_path = os.path.join(
                args.output, f"{pair['clip_name']}_{method_name}.csv"
            )
            save_per_clip_csv(results, per_clip_path)

            summary = compute_summary(
                results, avg_fps,
                pair["clip_name"], pair["condition"],
                method_name, pair["video_fps"],
            )
            all_summaries.append(summary)

            error_str = (f"{summary['mean_error_px']:.2f}px"
                         if summary["mean_error_px"] is not None else "N/A")
            print(f"error={error_str}, "
                  f"jitter={summary['jitter_px_per_sec']:.2f}px/s, "
                  f"fps={summary['fps']}")

        print()

    summary_path = os.path.join(args.output, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip", "condition", "method", "mean_error_px",
                           "jitter_px_per_sec", "fps"]
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
