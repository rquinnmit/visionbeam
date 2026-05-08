from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from visionbeam.base import TargetMethod


@dataclass
class Track:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def feet(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class HybridMethod(TargetMethod):
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.3,
        detect_every_n: int = 2,
        blur_ksize: int = 31,
        diff_threshold: int = 25,
        min_motion_area: int = 100,
        scale_width: int = 320,
        beam_mask_radius: int = 40,
        snap_to_feet: bool = True,
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.detect_every_n = detect_every_n
        self.blur_ksize = blur_ksize
        self.diff_threshold = diff_threshold
        self.min_motion_area = min_motion_area
        self.scale_width = scale_width
        self.beam_mask_radius = beam_mask_radius
        self.snap_to_feet = snap_to_feet

        self.prev_gray: np.ndarray | None = None
        self.frame_count: int = 0
        self.tracks: list[Track] = []
        self.beam_pos: tuple[int, int] | None = None
        self.locked_id: int | None = None

    def reset(self):
        self.prev_gray = None
        self.frame_count = 0
        self.tracks = []
        self.beam_pos = None

    def set_beam_position(self, x: int, y: int):
        self.beam_pos = (x, y)

    def lock_to_id(self, track_id: int | None):
        self.locked_id = track_id

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        h, w = frame.shape[:2]
        scale = self.scale_width / w
        small_h = int(h * scale)
        small = cv2.resize(frame, (self.scale_width, small_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        if self.frame_count % self.detect_every_n == 0 or not self.tracks:
            results = self.model.track(
                frame, persist=True, conf=self.confidence, classes=[0], verbose=False,
            )
            boxes = results[0].boxes
            self.tracks = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.arange(len(xyxy))
                )
                for i, box in enumerate(xyxy):
                    self.tracks.append(Track(
                        id=int(ids[i]),
                        x1=float(box[0]), y1=float(box[1]),
                        x2=float(box[2]), y2=float(box[3]),
                    ))

        if self.locked_id is not None:
            for t in self.tracks:
                if t.id == self.locked_id:
                    return t.feet
            return None

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, motion_mask = cv2.threshold(
            diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )

        if not self.tracks:
            return None

        person_mask = np.zeros((small_h, self.scale_width), dtype=np.uint8)
        for t in self.tracks:
            x1 = int(t.x1 * scale)
            y1 = int(t.y1 * scale)
            x2 = int(t.x2 * scale)
            y2 = int(t.y2 * scale)
            person_mask[y1:y2, x1:x2] = 255

        masked_motion = cv2.bitwise_and(motion_mask, person_mask)

        if self.beam_pos is not None:
            bx = int(self.beam_pos[0] * scale)
            by = int(self.beam_pos[1] * scale)
            cv2.circle(masked_motion, (bx, by), self.beam_mask_radius, 0, -1)

        if cv2.countNonZero(masked_motion) < self.min_motion_area:
            return None

        heatmap = cv2.GaussianBlur(
            masked_motion, (self.blur_ksize, self.blur_ksize), 0
        )

        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        peak_x = max_loc[0] / scale
        peak_y = max_loc[1] / scale

        if self.snap_to_feet:
            for t in self.tracks:
                if t.contains(peak_x, peak_y):
                    return t.feet
        return peak_x, peak_y
