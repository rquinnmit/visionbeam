"""
VisionBeam entry point.

Launches either the calibration wizard or the live pipeline with the
PySide6 Director's Station UI (or an OpenCV fallback for development).
The live view renders tracked person bounding boxes, persistent IDs,
and the person-masked motion heatmap alongside light aim indicators.

Usage:
    python main.py                          # Launch Director's Station
    python main.py --calibrate              # Run calibration wizard
    python main.py --no-dmx                 # Run without DMX hardware
    python main.py --camera 1               # Use a specific camera index
    python main.py --fixture path/to.json   # Use a specific fixture profile
"""

import argparse
import queue
import sys
import cv2

from visionbeam.calibration import FloorCalibration
from visionbeam.dmx import DMXConnection, FixtureProfile
from visionbeam.ik import LightMount
from visionbeam.pipeline import Pipeline


DEFAULT_FIXTURE = "config/fixture_default.json"
DEFAULT_CALIBRATION = "calibration/homography.json"
DEFAULT_MOUNT = "calibration/mount.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionBeam — autonomous stage light")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the calibration wizard instead of the live pipeline")
    parser.add_argument("--no-dmx", action="store_true",
                        help="Run without DMX hardware (CV-only mode)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--fixture", type=str, default=DEFAULT_FIXTURE,
                        help="Path to fixture profile JSON")
    parser.add_argument("--dmx-port", type=str, default="/dev/ttyUSB0",
                        help="Serial port for USB-to-DMX adapter")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION,
                        help="Path to saved homography JSON")
    parser.add_argument("--mount", type=str, default=DEFAULT_MOUNT,
                        help="Path to saved light mount JSON")
    return parser.parse_args()

def run_live(args: argparse.Namespace):
    """Launch the live pipeline with an OpenCV preview window."""
    calibration = FloorCalibration.load(args.calibration)
    mount = LightMount.load(args.mount)
    dmx = None
    if not args.no_dmx:
        fixture = FixtureProfile(args.fixture)
        dmx = DMXConnection(args.dmx_port, fixture)
        dmx.set_defaults(dimmer=255)
        dmx.start()
    display_queue: queue.Queue = queue.Queue(maxsize=2)
    pipeline = Pipeline(
        camera_index=args.camera,
        calibration=calibration,
        mount=mount,
        dmx=dmx,
        display_queue=display_queue,
    )
    pipeline.start()
    print("VisionBeam running. Press 'q' to quit, 'a' to toggle auto/manual.")
    try:
        while True:
            try:
                payload = display_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            frame = payload["frame"]
            target_px = payload["target_px"]
            if target_px is not None:
                cx, cy = int(target_px[0]), int(target_px[1])
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)
            status = "AUTO" if payload["auto_enabled"] else "MANUAL"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if payload["pan"] is not None:
                aim_text = f"Pan: {payload['pan']:.1f}  Tilt: {payload['tilt']:.1f}"
                cv2.putText(frame, aim_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("VisionBeam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                pipeline.state.auto_enabled = not pipeline.state.auto_enabled
                pipeline.state.manual_target = None
    finally:
        pipeline.stop()
        if dmx is not None:
            dmx.blackout()
            dmx.stop()
        cv2.destroyAllWindows()


def run_calibration(args: argparse.Namespace):
    """Placeholder for the calibration wizard."""
    print("Calibration wizard not yet implemented.")
    print("Use FloorCalibration and triangulate_light from visionbeam.calibration")
    print("to generate calibration/homography.json and calibration/mount.json.")
    sys.exit(0)



def main():
    args = parse_args()
    if args.calibrate:
        run_calibration(args)
    else:
        run_live(args)
if __name__ == "__main__":
    main()
