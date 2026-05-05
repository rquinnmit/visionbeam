"""
Person detection, tracking, and motion analysis module.

YOLOv8-nano detects people in each frame (optionally every Nth frame for
throughput). ByteTrack associates detections across frames via Kalman
filtering and IoU matching, assigning persistent IDs. Frame differencing
produces a per-pixel motion signal that is masked to tracked person
bounding boxes, then Gaussian-blurred into a spatial heatmap. The peak
of the person-masked heatmap identifies the most active dancer.

When a track ID is locked (e.g. via the operator UI), the target snaps to
the centroid of that ID's bounding box instead of the heatmap peak.

Beam masking prevents the system from chasing its own light, and a
configurable min-motion threshold suppresses noise when the floor is idle.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from evaluation.methods import TargetMethod


@dataclass
class Track:
    """A single tracked person from the most recent detection pass."""
    id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class HybridMethod(TargetMethod):
    """
    YOLOv8 + ByteTrack + person-masked motion heatmap.

    Pipeline per frame:
        1. Run YOLOv8n detection (optionally every Nth frame).
        2. ByteTrack associates detections with persistent IDs.
        3. If locked to an ID, target = that box's centroid; return.
        4. Otherwise: frame difference, mask to tracked boxes,
           optionally subtract beam aim, blur, peak.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3, detect_every_n: int = 2, blur_ksize: int = 31,
                    diff_threshold: int = 25, min_motion_area: int = 100, scale_width: int = 320, beam_mask_radius: int = 40):
        self._model = YOLO(model_name)
        self._confidence = confidence
        self._detect_every_n = detect_every_n
        self._blur_ksize = blur_ksize
        self._diff_threshold = diff_threshold
        self._min_motion_area = min_motion_area
        self._scale_width = scale_width
        self._beam_mask_radius = beam_mask_radius

        self._prev_gray: np.ndarray | None = None
        self._frame_count: int = 0
        self._tracks: list[Track] = []
        self._beam_pos: tuple[int, int] | None = None
        self._locked_id: int | None = None

    def reset(self):
        self._prev_gray = None
        self._frame_count = 0
        self._tracks = []
        self._beam_pos = None

    def set_beam_position(self, x: int, y: int):
        """Update the current beam aim point (in full-res pixel coords)."""
        self._beam_pos = (x, y)

    def lock_to_id(self, track_id: int | None):
        """
        Lock onto a specific tracked person. Pass None to release.

        While locked, process_frame returns that person's bbox centroid
        instead of the motion heatmap peak.
        """
        self._locked_id = track_id

    @property
    def locked_id(self) -> int | None:
        return self._locked_id

    @property
    def tracks(self) -> list[Track]:
        """Most recently detected tracked persons (with persistent IDs)."""
        return self._tracks

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        h, w = frame.shape[:2]
        scale = self._scale_width / w
        small_h = int(h * scale)
        small = cv2.resize(frame, (self._scale_width, small_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        self._frame_count += 1
        if self._frame_count % self._detect_every_n == 0 or not self._tracks:
            results = self._model.track(
                frame, persist=True, conf=self._confidence, classes=[0], verbose=False,
            )
            boxes = results[0].boxes
            self._tracks = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.arange(len(xyxy))
                )
                for i, box in enumerate(xyxy):
                    self._tracks.append(Track(
                        id=int(ids[i]),
                        x1=float(box[0]), y1=float(box[1]),
                        x2=float(box[2]), y2=float(box[3]),
                    ))

        # Lock-on path: target = centroid of the locked track's box.
        if self._locked_id is not None:
            for t in self._tracks:
                if t.id == self._locked_id:
                    return t.center
            # Locked ID not in current detections — no target this frame.
            return None

        # Auto path: motion heatmap on person-masked frame diff.
        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        _, motion_mask = cv2.threshold(
            diff, self._diff_threshold, 255, cv2.THRESH_BINARY
        )

        if not self._tracks:
            return None

        person_mask = np.zeros((small_h, self._scale_width), dtype=np.uint8)
        for t in self._tracks:
            x1 = int(t.x1 * scale)
            y1 = int(t.y1 * scale)
            x2 = int(t.x2 * scale)
            y2 = int(t.y2 * scale)
            person_mask[y1:y2, x1:x2] = 255

        masked_motion = cv2.bitwise_and(motion_mask, person_mask)

        if self._beam_pos is not None:
            bx = int(self._beam_pos[0] * scale)
            by = int(self._beam_pos[1] * scale)
            cv2.circle(masked_motion, (bx, by), self._beam_mask_radius, 0, -1)

        if cv2.countNonZero(masked_motion) < self._min_motion_area:
            return None

        heatmap = cv2.GaussianBlur(
            masked_motion, (self._blur_ksize, self._blur_ksize), 0
        )

        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        sx, sy = max_loc

        return sx / scale, sy / scale
