import cv2
import numpy as np
from ultralytics import YOLO

from visionbeam.base import TargetMethod


def downscale_gray(frame: np.ndarray, scale_width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = scale_width / w
    small = cv2.resize(frame, (scale_width, int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray, scale


class FrameDiffMethod(TargetMethod):
    def __init__(self, blur_ksize: int = 31, threshold: int = 25, scale_width: int = 320):
        self.prev_gray: np.ndarray | None = None
        self.blur_ksize = blur_ksize
        self.threshold = threshold
        self.scale_width = scale_width

    def reset(self):
        self.prev_gray = None

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        gray, scale = downscale_gray(frame, self.scale_width)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        heatmap = cv2.GaussianBlur(mask, (self.blur_ksize, self.blur_ksize), 0)

        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < self.threshold:
            return None

        sx, sy = max_loc
        return sx / scale, sy / scale


class FarnebackFlowMethod(TargetMethod):
    def __init__(self, scale_width: int = 320, min_magnitude: float = 2.0):
        self.prev_gray: np.ndarray | None = None
        self.scale_width = scale_width
        self.min_magnitude = min_magnitude

    def reset(self):
        self.prev_gray = None

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        gray, scale = downscale_gray(frame, self.scale_width)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, flow=None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        self.prev_gray = gray

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mag_blurred = cv2.GaussianBlur(mag, (31, 31), 0)

        _, max_val, _, max_loc = cv2.minMaxLoc(mag_blurred)
        if max_val < self.min_magnitude:
            return None

        sx, sy = max_loc
        return sx / scale, sy / scale


class DetectionMethod(TargetMethod):
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3, detect_every_n: int = 1):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.detect_every_n = detect_every_n
        self.frame_count = 0
        self.last_target: tuple[float, float] | None = None

    def reset(self):
        self.frame_count = 0
        self.last_target = None

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        self.frame_count += 1

        if self.frame_count % self.detect_every_n != 0:
            return self.last_target

        results = self.model.track(
            frame, persist=True, conf=self.confidence, classes=[0], verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.last_target = None
            return None

        best_idx = 0
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            best_idx = int(np.argmin(ids))

        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2

        self.last_target = (float(cx), float(cy))
        return self.last_target
