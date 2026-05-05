"""
Dataset recording tool.

Captures video clips from the webcam under each of the 5 controlled
lighting conditions (ambient, static color, slow drift, strobe, moving
beam). Saves each clip as a timestamped video file alongside a metadata
JSON containing the lighting condition label, camera index, resolution,
FPS, and duration. Optionally triggers DMX fixture states to automate
the lighting changes between conditions.

Usage:
    python -m evaluation.record --camera 0 --output data/clips/
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2


LIGHTING_CONDITIONS = [
    "ambient",
    "static_color",
    "slow_drift",
    "strobe",
    "moving_beam",
]


def record_clip(
    cap: cv2.VideoCapture,
    output_dir: str,
    condition: str,
    duration: float,
    fps: float,
) -> str:
    """
    Records a single video clip and saves it with a metadata sidecar.
    Returns the path to the saved video file.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{condition}_{timestamp}"
    video_path = os.path.join(output_dir, f"{filename}.mp4")
    meta_path = os.path.join(output_dir, f"{filename}.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    frame_interval = 1.0 / fps
    start_time = time.monotonic()
    frame_count = 0

    print(f"  Recording '{condition}' for {duration}s ...")

    while time.monotonic() - start_time < duration:
        loop_start = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)
        frame_count += 1

        cv2.putText(
            frame,
            f"REC {condition} | {time.monotonic() - start_time:.1f}s / {duration}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("     Recording interrupted by user.")
            break

        elapsed = time.monotonic() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    writer.release()
    actual_duration = time.monotonic() - start_time

    metadata = {
        "condition": condition,
        "filename": f"{filename}.mp4",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": round(actual_duration, 2),
        "recorded_at": timestamp,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved: {video_path} ({frame_count} frames, {actual_duration:.1f}s)")
    return video_path


def main():
    parser = argparse.ArgumentParser(
        description="Record evaluation dataset clips under different lighting conditions"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/clips",
        help="Output directory for clips and metadata",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="Duration of each clip in seconds",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Recording frame rate",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=LIGHTING_CONDITIONS,
        help="Lighting conditions to record",
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return
    print(f"Recording {len(args.conditions)} clips to {args.output}/")
    print(f"Duration: {args.duration}s each | FPS: {args.fps}")
    print()
    try:
        for i, condition in enumerate(args.conditions):
            print(f"[{i + 1}/{len(args.conditions)}] Condition: {condition}")
            input("  Press Enter when lighting is set, or Ctrl+C to abort...")
            record_clip(cap, args.output, condition, args.duration, args.fps)
            print()
    except KeyboardInterrupt:
        print("\nRecording session aborted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()