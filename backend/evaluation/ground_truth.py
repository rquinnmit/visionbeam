"""
Ground truth extraction from recorded video.

Detects a known tracking marker (bright retroreflector or specific-color
LED) in each frame of a recorded clip and outputs the marker's (x, y)
pixel position per frame as a CSV. Supports two modes:

1. Color thresholding — HSV range isolation for a distinctly colored marker
   (e.g., bright green LED not present in stage lighting palette).
2. Brightness peak — For a retroreflector that appears as the brightest
   small blob in an otherwise dim scene.

When the marker is not detected in a frame (occlusion, failure), the
frame is flagged and linearly interpolated from neighbors. The output
CSV has columns: frame_number, timestamp_ms, x_px, y_px, detected (bool).

Usage:
    python -m evaluation.ground_truth --video data/clips/ambient_01.mp4 \
                                      --mode color --output data/gt/
"""

import argparse
import csv
import os

import cv2
import numpy as np


DEFAULT_HSV_LOW = (35, 100, 100)
DEFAULT_HSV_HIGH = (85, 255, 255)
MIN_CONTOUR_AREA = 30


def detect_color_marker(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int],
    hsv_high: tuple[int, int, int],
) -> tuple[float, float] | None:
    """Finds the centroid of the largest blob matching the HSV range."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def detect_brightness_marker(frame: np.ndarray) -> tuple[float, float] | None:
    """Finds the centroid of the brightest small blob in a dim scene."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def interpolate_gaps(records: list[dict]) -> list[dict]:
    """
    Linearly interpolates x_px and y_px for frames where detected=False,
    using the nearest detected neighbors on each side.
    """
    n = len(records)
    detected_indices = [i for i, r in enumerate(records) if r["detected"]]
    if len(detected_indices) < 2:
        return records
    for i in range(n):
        if records[i]["detected"]:
            continue
        prev_idx = None
        next_idx = None
        for di in detected_indices:
            if di < i:
                prev_idx = di
            elif di > i and next_idx is None:
                next_idx = di
        if prev_idx is not None and next_idx is not None:
            span = next_idx - prev_idx
            t = (i - prev_idx) / span
            records[i]["x_px"] = (
                records[prev_idx]["x_px"] * (1 - t) + records[next_idx]["x_px"] * t
            )
            records[i]["y_px"] = (
                records[prev_idx]["y_px"] * (1 - t) + records[next_idx]["y_px"] * t
            )
        elif prev_idx is not None:
            records[i]["x_px"] = records[prev_idx]["x_px"]
            records[i]["y_px"] = records[prev_idx]["y_px"]
        elif next_idx is not None:
            records[i]["x_px"] = records[next_idx]["x_px"]
            records[i]["y_px"] = records[next_idx]["y_px"]
    return records


def extract_ground_truth(
    video_path: str,
    mode: str = "color",
    hsv_low: tuple[int, int, int] = DEFAULT_HSV_LOW,
    hsv_high: tuple[int, int, int] = DEFAULT_HSV_HIGH,
    preview: bool = False,
) -> list[dict]:
    """
    Runs marker detection on every frame of a video and returns
    a list of per-frame records with interpolated gaps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    records = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "color":
            pos = detect_color_marker(frame, hsv_low, hsv_high)
        else:
            pos = detect_brightness_marker(frame)

        detected = pos is not None
        x, y = pos if detected else (0.0, 0.0)

        records.append({
            "frame": frame_num,
            "timestamp_ms": round(frame_num / fps * 1000, 1),
            "x_px": round(x, 2),
            "y_px": round(y, 2),
            "detected": detected,
        })

        if preview:
            if detected:
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), 2)
                cv2.drawMarker(
                    frame,
                    (int(x), int(y)),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    16,
                    2,
                )
            cv2.imshow("Ground Truth Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_num += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    total = len(records)
    detected_count = sum(1 for r in records if r["detected"])
    print(f"  Detected marker in {detected_count}/{total} frames "
          f"({detected_count / total * 100:.1f}%)" if total > 0 else "  No frames read.")

    records = interpolate_gaps(records)
    return records


def save_csv(records: list[dict], output_path: str):
    """Write records to a CSV file."""
    fieldnames = ["frame", "timestamp_ms", "x_px", "y_px", "detected"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {output_path}")


def parse_hsv(s: str) -> tuple[int, int, int]:
    """Parse a comma-separated HSV triplet string."""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV must be three comma-separated integers")
    return (parts[0], parts[1], parts[2])


def main():
    parser = argparse.ArgumentParser(
        description="Extract ground truth marker positions from recorded video"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to recorded video clip",
    )
    parser.add_argument(
        "--mode",
        choices=["color", "brightness"],
        default="color",
        help="Detection mode: 'color' (HSV) or 'brightness' (peak)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gt",
        help="Output directory for GT CSV files",
    )
    parser.add_argument(
        "--hsv-low",
        type=parse_hsv,
        default="35,100,100",
        help="Lower HSV bound as 'H,S,V' (color mode)",
    )
    parser.add_argument(
        "--hsv-high",
        type=parse_hsv,
        default="85,255,255",
        help="Upper HSV bound as 'H,S,V' (color mode)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview with marker overlay",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = os.path.join(args.output, f"{video_name}_gt.csv")

    print(f"Extracting ground truth from: {args.video}")
    print(f"Mode: {args.mode}")

    records = extract_ground_truth(
        args.video,
        mode=args.mode,
        hsv_low=args.hsv_low,
        hsv_high=args.hsv_high,
        preview=args.preview,
    )
    save_csv(records, output_path)


if __name__ == "__main__":
    main()