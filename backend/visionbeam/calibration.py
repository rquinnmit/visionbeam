"""
Spatial calibration module.

Step 1: Camera-to-floor homography via ArUco marker detection.
Step 2: Light position triangulation from known aim points.
"""

import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from visionbeam.ik import LightMount

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def generate_marker_image(marker_id: int, size_px: int = 200) -> np.ndarray:
    """
    Generates a single ArUco marker image for printing.

    Args:
        marker_id: Marker ID (0-49 for DICT_4x4_50).
        size_px: Output image width/height in pixels.

    Returns:
        Grayscale image (numpy array) of the marker.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)


def generate_marker_sheet(marker_ids: list[int], path: str, size_px: int = 200, margin: int = 40):
    """
    Generates and saves a printable sheet of ArUco markers.

    Creates a single image with the specified markers arranged in a row,
    each labeled with its ID. Saves to disk as a PNG.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    cell = size_px + margin * 2
    sheet = np.ones((cell + 40, cell * len(marker_ids)), dtype=np.uint8) * 255

    for i, mid in enumerate(marker_ids):
        marker = cv2.aruco.generateImageMarker(aruco_dict, mid, size_px)
        x_offset = i * cell + margin
        sheet[margin:margin+size_px, x_offset:x_offset+size_px] = marker
        cv2.putText(
            sheet, f"ID {mid}",
            (x_offset + size_px // 3, margin + size_px + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2,
        )
    cv2.imwrite(path, sheet)



class FloorCalibration:
    """
    Camera-to-floor coordinate mapping (via homography).

    Uses ArUco markers placed at known floor positions to compute a perspective
    transform from camera pixels to floor coordinates.
    """
    def __init__(self):
        self.homography: np.ndarray | None = None
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    def detect_markers(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        """
        Detects ArUco markers in a camera frame.

        Returns:
            Dictionary mapping marker ID to its center pixel coordinate (x, y).
        """
        corners, ids, _ = self._detector.detectMarkers(frame)
        if ids is None:
            return {}

        markers = {}
        for i, marker_id in enumerate(ids.flatten()):
            center = corners[i][0].mean(axis=0)
            markers[int(marker_id)] = center
        return markers

    def compute_homography(
        self,
        pixel_points: np.ndarray,
        floor_points: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the perspective transform from pixels to floor coordinates.

        Args:
            pixel_points: Nx2 array of marker centers in pixel space.
            floor_points: Nx2 array of corresponding positions in meters.

        Returns:
            3x3 homography matrix.

        Raises:
            ValueError: If fewer than 4 point pairs are provided.
        """
        if len(pixel_points) < 4:
            raise ValueError(f"Need at least 4 points, got {len(pixel_points)}")

        H, _ = cv2.findHomography(pixel_points, floor_points)
        if H is None:
            raise ValueError("Homography computation failed — points may be collinear")

        self.homography = H
        return H

    def pixel_to_floor(self, px: float, py: float) -> tuple[float, float]:
        """Transforms a single pixel coordinate to floor coordinates (meters)."""
        if self.homography is None:
            raise RuntimeError("Homography not computed — run compute_homography first")

        pt = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def warp_frame(self, frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        """Warps a camera frame to the top-down floor view (for the UI)."""
        if self.homography is None:
            raise RuntimeError("Homography not computed — run compute_homography first")
        return cv2.warpPerspective(frame, self.homography, output_size)

    def save(self, path: str):
        """Persists the homography matrix to a JSON file."""
        if self.homography is None:
            raise RuntimeError("Nothing to save — homography not computed")

        with open(path, "w") as f:
            json.dump({"homography": self.homography.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FloorCalibration":
        """Loads a previously saved homography from JSON."""
        cal = cls()
        with open(path) as f:
            data = json.load(f)
        cal.homography = np.array(data["homography"])
        return cal


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference in degrees, handling wrapping."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def triangulate_light(
    aim_points: list[dict],
    initial_position: tuple[float, float, float] = (0.0, 0.0, 3.0),
) -> LightMount:
    """
    Solves for the light's 3D position and angular offsets.

    The operator aims the light at 3+ known floor positions and records
    the fixture's pan/tilt readings at each. This function finds the
    mount position and offsets that best explain those readings.

    Args:
        aim_points: List of dicts, each with keys:
            "floor_x", "floor_y" — known target position in meters.
            "pan_deg", "tilt_deg" — fixture's pan/tilt reading at that target.
        initial_position: Starting guess for (x, y, z) of the light in meters.

    Returns:
        A fully calibrated LightMount.

    Raises:
        ValueError: If fewer than 3 aim points are provided.
    """
    if len(aim_points) < 3:
        raise ValueError(f"Need at least 3 aim points, got {len(aim_points)}")

    targets = np.array([(p["floor_x"], p["floor_y"]) for p in aim_points])
    pan_readings = np.array([p["pan_deg"] for p in aim_points])
    tilt_readings = np.array([p["tilt_deg"] for p in aim_points])

    def residuals(params):
        lx, ly, lz, pan_off, tilt_off = params
        res = []
        for i in range(len(targets)):
            dx = targets[i, 0] - lx
            dy = targets[i, 1] - ly
            h_dist = np.hypot(dx, dy)

            pred_pan = np.degrees(np.arctan2(dy, dx)) + pan_off
            pred_tilt = np.degrees(np.arctan2(h_dist, lz)) + tilt_off

            res.append(_angle_diff(pred_pan, pan_readings[i]))
            res.append(_angle_diff(pred_tilt, tilt_readings[i]))
        return res

    x0 = [
        initial_position[0],
        initial_position[1],
        initial_position[2],
        270.0,
        135.0,
    ]

    result = least_squares(
        residuals,
        x0,
        bounds=(
            [-np.inf, -np.inf, 0.5, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
    )

    lx, ly, lz, pan_off, tilt_off = result.x
    return LightMount(x=lx, y=ly, z=lz, pan_offset=pan_off, tilt_offset=tilt_off)
