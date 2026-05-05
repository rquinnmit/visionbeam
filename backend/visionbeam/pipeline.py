"""
Main pipeline loop.

Camera capture → person detection → multi-person tracking → motion
heatmap → spatial translation → DMX output. Runs on a dedicated thread,
pushes display data (frame, heatmap, tracked persons, aim state) to the
UI via queue.
"""

import queue
import threading
import time
import cv2

from visionbeam.calibration import FloorCalibration
from visionbeam.dmx import DMXConnection
from visionbeam.ik import LightMount, TargetSmoother, floor_to_pan_tilt
from visionbeam.tracker import HybridMethod


class PipelineState:
    """Shared mutable state between the pipeline thread and UI."""

    def __init__(self):
        self.running = False
        self.manual_target: tuple[float, float] | None = None
        self.auto_enabled = True


class Pipeline:
    """
    Threaded pipeline: camera -> tracker -> IK -> DMX.
    """

    def __init__(
        self,
        camera_index: int,
        calibration: FloorCalibration,
        mount: LightMount,
        dmx: DMXConnection | None,
        display_queue: queue.Queue,
        tracker: HybridMethod | None = None,
        smoother_alpha: float = 0.2,
        target_fps: float = 30.0,
    ):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        self._calibration = calibration
        self._mount = mount
        self._dmx = dmx
        self._display_queue = display_queue
        self._tracker = tracker or HybridMethod()
        self._smoother = TargetSmoother(alpha=smoother_alpha)
        self._frame_interval = 1.0 / target_fps

        self.state = PipelineState()
        self._thread: threading.Thread | None = None

    def start(self):
        if self.state.running:
            return
        self.state.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.state.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def _loop(self):
        while self.state.running:
            loop_start = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                continue

            target_px = None
            floor_target = None
            smoothed_target = None
            pan, tilt = None, None

            if self.state.auto_enabled and self.state.manual_target is None:
                target_px = self._tracker.process_frame(frame)
            elif self.state.manual_target is not None:
                target_px = self.state.manual_target

            if target_px is not None:
                floor_x, floor_y = self._calibration.pixel_to_floor(
                    target_px[0], target_px[1]
                )
                floor_target = (floor_x, floor_y)
                sx, sy = self._smoother.update(floor_x, floor_y)
                smoothed_target = (sx, sy)

                pan, tilt = floor_to_pan_tilt(sx, sy, self._mount)

                if self._dmx is not None:
                    self._dmx.aim(pan, tilt)

                self._tracker.set_beam_position(
                    int(target_px[0]), int(target_px[1])
                )

            display_payload = {
                "frame": frame,
                "target_px": target_px,
                "floor_target": floor_target,
                "smoothed_target": smoothed_target,
                "pan": pan,
                "tilt": tilt,
                "auto_enabled": self.state.auto_enabled,
            }

            try:
                self._display_queue.put_nowait(display_payload)
            except queue.Full:
                pass

            elapsed = time.monotonic() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
