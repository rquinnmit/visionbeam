"""
Pixel -> pan/tilt aim calibration.

The operator drives the fixture to a sequence of pan/tilt setpoints and
clicks the beam dot in the camera image at each. Those samples train a
quadratic least-squares fit:

    pan  = a0 + a1*px + a2*py + a3*px*py + a4*px^2 + a5*py^2
    tilt = b0 + b1*px + b2*py + b3*px*py + b4*px^2 + b5*py^2

Off-screen samples (where the operator hits "off-screen") are stored
for completeness but excluded from the fit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

MIN_SAMPLES_FOR_FIT = 6
N_FEATURES = 6


@dataclass
class AimSample:
    pan: float
    tilt: float
    px: float | None
    py: float | None


def _features(px: float, py: float) -> np.ndarray:
    return np.array([1.0, px, py, px * py, px * px, py * py])


class PixelAimCalibration:
    """Quadratic fit of pixel coords to fixture pan/tilt angles."""

    def __init__(self):
        self._samples: list[AimSample] = []
        self._coeffs_pan: np.ndarray | None = None
        self._coeffs_tilt: np.ndarray | None = None
        self._rms_pan: float | None = None
        self._rms_tilt: float | None = None

    @property
    def fitted(self) -> bool:
        return self._coeffs_pan is not None and self._coeffs_tilt is not None

    @property
    def samples(self) -> list[AimSample]:
        return list(self._samples)

    @property
    def n_in_frame(self) -> int:
        return sum(1 for s in self._samples if s.px is not None)

    def add_sample(self, pan: float, tilt: float, px: float | None, py: float | None) -> None:
        self._samples.append(AimSample(pan=pan, tilt=tilt, px=px, py=py))

    def clear(self) -> None:
        self._samples.clear()
        self._coeffs_pan = None
        self._coeffs_tilt = None
        self._rms_pan = None
        self._rms_tilt = None

    def fit(self) -> dict[str, Any]:
        in_frame = [s for s in self._samples if s.px is not None and s.py is not None]
        if len(in_frame) < MIN_SAMPLES_FOR_FIT:
            return {
                "fitted": False,
                "reason": f"need >={MIN_SAMPLES_FOR_FIT} in-frame samples, have {len(in_frame)}",
                "n_samples": len(in_frame),
            }

        X = np.stack([_features(s.px, s.py) for s in in_frame])  # type: ignore[arg-type]
        pan_vec = np.array([s.pan for s in in_frame])
        tilt_vec = np.array([s.tilt for s in in_frame])

        a, *_ = np.linalg.lstsq(X, pan_vec, rcond=None)
        b, *_ = np.linalg.lstsq(X, tilt_vec, rcond=None)

        self._coeffs_pan = a
        self._coeffs_tilt = b
        self._rms_pan = float(np.sqrt(np.mean((X @ a - pan_vec) ** 2)))
        self._rms_tilt = float(np.sqrt(np.mean((X @ b - tilt_vec) ** 2)))

        return {
            "fitted": True,
            "n_samples": len(in_frame),
            "rms_pan_deg": self._rms_pan,
            "rms_tilt_deg": self._rms_tilt,
        }

    def predict(self, px: float, py: float) -> tuple[float, float] | None:
        if not self.fitted:
            return None
        feats = _features(px, py)
        pan = float(feats @ self._coeffs_pan)  # type: ignore[operator]
        tilt = float(feats @ self._coeffs_tilt)  # type: ignore[operator]
        return pan, tilt

    def status(self) -> dict[str, Any]:
        return {
            "n_samples": len(self._samples),
            "n_in_frame": self.n_in_frame,
            "fitted": self.fitted,
            "rms_pan_deg": self._rms_pan,
            "rms_tilt_deg": self._rms_tilt,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [asdict(s) for s in self._samples],
            "coeffs_pan": self._coeffs_pan.tolist() if self._coeffs_pan is not None else None,
            "coeffs_tilt": self._coeffs_tilt.tolist() if self._coeffs_tilt is not None else None,
            "rms_pan_deg": self._rms_pan,
            "rms_tilt_deg": self._rms_tilt,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PixelAimCalibration":
        cal = cls()
        if not os.path.exists(path):
            return cal
        with open(path) as f:
            data = json.load(f)
        for s in data.get("samples", []):
            cal._samples.append(AimSample(**s))
        if data.get("coeffs_pan") is not None:
            cal._coeffs_pan = np.array(data["coeffs_pan"])
        if data.get("coeffs_tilt") is not None:
            cal._coeffs_tilt = np.array(data["coeffs_tilt"])
        cal._rms_pan = data.get("rms_pan_deg")
        cal._rms_tilt = data.get("rms_tilt_deg")
        return cal
