"""
Person detection, tracking, and motion analysis module.

YOLOv8-nano detects people in each frame (optionally every Nth frame for
throughput). ByteTrack associates detections across frames via Kalman
filtering and IoU matching, assigning persistent IDs. Frame differencing
produces a per-pixel motion signal that is masked to tracked person
bounding boxes, then Gaussian-blurred into a spatial heatmap. The peak
of the person-masked heatmap identifies the most active dancer.

Beam masking prevents the system from chasing its own light, and a
configurable min-motion threshold suppresses noise when the floor is idle.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from evaluation.methods import TargetMethod
from visionbeam.viz import save_step


class HybridMethod(TargetMethod):
    """
    YOLOv8 + ByteTrack + person-masked motion heatmap.

    Pipeline per frame:
        1. Run YOLOv8n detection (optionally every Nth frame).
        2. ByteTrack associates detections with tracked bounding boxes.
        3. Compute frame difference on downscaled grayscale.
        4. Mask the diff to the union of tracked bounding boxes.
        5. Optionally mask out the beam's current aim point.
        6. Gaussian blur -> heatmap peak = target pixel.
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
        self._last_boxes: np.ndarray | None = None
        self._beam_pos: tuple[int, int] | None = None

    def reset(self):
        self._prev_gray = None
        self._frame_count = 0
        self._last_boxes = None
        self._beam_pos = None

    def set_beam_position(self, x: int, y: int):
        """Update the current beam aim point (in full-res pixel coords)."""
        self._beam_pos = (x, y)

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        h, w = frame.shape[:2]
        scale = self._scale_width / w
        small_h = int(h * scale)
        small = cv2.resize(frame, (self._scale_width, small_h))
        save_step(small, "02_downscaled")
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        save_step(gray, "03_grayscale")

        # Detection (every Nth frame)
        self._frame_count += 1
        if self._frame_count % self._detect_every_n == 0 or self._last_boxes is None:
            results = self._model.track(
                frame, persist=True, conf=self._confidence, classes=[0], verbose=False,
            )
            save_step(results[0].plot(), "04_detection")
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                self._last_boxes = boxes.xyxy.cpu().numpy()
            else:
                self._last_boxes = None

        # Frame differencing
        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        _, motion_mask = cv2.threshold(
            diff, self._diff_threshold, 255, cv2.THRESH_BINARY
        )

        # Person masking
        if self._last_boxes is None:
            return None

        person_mask = np.zeros((small_h, self._scale_width), dtype=np.uint8)
        for box in self._last_boxes:
            x1 = int(box[0] * scale)
            y1 = int(box[1] * scale)
            x2 = int(box[2] * scale)
            y2 = int(box[3] * scale)
            person_mask[y1:y2, x1:x2] = 255

        masked_motion = cv2.bitwise_and(motion_mask, person_mask)

        # Beam masking
        if self._beam_pos is not None:
            bx = int(self._beam_pos[0] * scale)
            by = int(self._beam_pos[1] * scale)
            cv2.circle(masked_motion, (bx, by), self._beam_mask_radius, 0, -1)

        # Min-motion gate
        if cv2.countNonZero(masked_motion) < self._min_motion_area:
            return None

        # Heatmap peak
        heatmap = cv2.GaussianBlur(
            masked_motion, (self._blur_ksize, self._blur_ksize), 0
        )

        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        sx, sy = max_loc
        
        return sx / scale, sy / scale
