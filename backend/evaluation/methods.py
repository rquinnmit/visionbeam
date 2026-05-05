"""
Target selection methods for comparative evaluation.

Each method implements the same interface: given a pair of consecutive
frames (and optionally prior state), return a predicted target (x, y)
in pixel space representing "where the light should aim." Methods:

1. FrameDiffMethod      — Raw frame differencing (cv2.absdiff), threshold,
                          Gaussian blur, peak of heatmap. No person filtering.
2. FarnebackFlowMethod  — Dense optical flow (cv2.calcOpticalFlowFarneback),
                          magnitude map, peak of motion energy.
3. DetectionMethod      — YOLOv8n + ByteTrack, target = center of most
                          persistent / most central tracked person.
4. HybridMethod         — YOLOv8n bounding boxes mask a frame-differencing
                          heatmap; peak of the person-masked heatmap. This is
                          the current VisionBeam design.

All methods return results in pixel coordinates; the evaluation harness
applies the homography transform to floor coordinates before computing
metrics against ground truth.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np
from ultralytics import YOLO


class TargetMethod(ABC):
    """
    Abstract base class for target selection methods.
    """

    @abstractmethod
    def reset(self):
        """Clear any internal state. Called between clips."""

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        """
        Process a single BGR frame and return the target pixel coordinate.
        """


class FrameDiffMethod(TargetMethod):
    """
    Baseline 1: Raw frame differencing.

    Computes per-pixel absolute differences between consecutive grayscale
    frames, thresholds, Gaussian-blurs into a heatmap, and returns the peak location.
    No person filtering - responds to all motion including lighting artifacts.
    """
    
    def __init__(self, blur_ksize: int = 31, threshold: int = 25, scale_width: int = 320):
        self._prev_gray: np.ndarray | None = None
        self._blur_ksize = blur_ksize
        self._threshold = threshold
        self._scale_width = scale_width

    def reset(self):
        self._prev_gray = None

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        h, w = frame.shape[:2]
        scale = self._scale_width / w
        small = cv2.resize(frame, (self._scale_width, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        _, mask = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)
        heatmap = cv2.GaussianBlur(mask, (self._blur_ksize, self._blur_ksize), 0)

        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < self._threshold:
            return None

        sx, sy = max_loc
        return sx / scale, sy / scale


class FarnebackFlowMethod(TargetMethod):
    """
    Baseline 2: Farneback dense optical flow.

    Computes dense flow between consecutive frames and returns the pixel 
        with the highest flow magnitude. 
    """

    def __init__(self, scale_width: int = 320, min_magnitude: float = 2.0):
        self._prev_gray: np.ndarray | None = None
        self._scale_width = scale_width
        self._min_magnitude = min_magnitude

    def reset(self):
        self._prev_gray = None

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        h, w = frame.shape[:2]
        scale = self._scale_width / w
        small = cv2.resize(frame, (self._scale_width, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, flow=None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self._prev_gray = gray

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mag_blurred = cv2.GaussianBlur(mag, (31, 31), 0)

        _, max_val, _, max_loc = cv2.minMaxLoc(mag_blurred)
        if max_val < self._min_magnitude:
            return None

        sx, sy = max_loc
        return sx / scale, sy / scale


class DetectionMethod(TargetMethod):
    """
    Baseline 3: Detection-only (YOLOv8n + ByteTrack)

    Returns the center of the tracked bounding box with the longest
    tracking age (most persistent person). Falls back to the detection nearest the
    frame center if no tracks have accumulated age. No motion signal - cannot distinguish
    a stationary person from an active dancer.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3, detect_every_n: int = 1):
        self._model = YOLO(model_name)
        self._confidence = confidence
        self._detect_every_n = detect_every_n
        self._frame_count = 0
        self._last_target: tuple[float, float] | None = None

    def reset(self):
        self._frame_count = 0
        self._last_target = None

    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        self._frame_count += 1
        
        if self._frame_count % self._detect_every_n != 0:
            return self._last_target

        results = self._model.track(
            frame, persist=True, conf=self._confidence, classes=[0], verbose=False
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self._last_target = None
            return None

        best_idx = 0
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy()
            best_idx = int(np.argmin(ids))

        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2

        self._last_target = (float(cx), float(cy))
        return self._last_target
