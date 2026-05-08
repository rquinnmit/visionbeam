from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

MIN_SAMPLES_FOR_FIT = 6


@dataclass
class AimSample:
    pan: float
    tilt: float
    px: float | None
    py: float | None


def features(px: float, py: float) -> np.ndarray:
    return np.array([1.0, px, py, px * py, px * px, py * py])


class PixelAimCalibration:
    def __init__(self):
        self.samples: list[AimSample] = []
        self.coeffs_pan: np.ndarray | None = None
        self.coeffs_tilt: np.ndarray | None = None
        self.rms_pan: float | None = None
        self.rms_tilt: float | None = None

    @property
    def fitted(self) -> bool:
        return self.coeffs_pan is not None and self.coeffs_tilt is not None

    @property
    def n_in_frame(self) -> int:
        return sum(1 for s in self.samples if s.px is not None)

    def add_sample(self, pan: float, tilt: float, px: float | None, py: float | None) -> None:
        self.samples.append(AimSample(pan=pan, tilt=tilt, px=px, py=py))

    def clear(self) -> None:
        self.samples.clear()
        self.coeffs_pan = None
        self.coeffs_tilt = None
        self.rms_pan = None
        self.rms_tilt = None

    def fit(self) -> dict[str, Any]:
        in_frame = [s for s in self.samples if s.px is not None and s.py is not None]
        if len(in_frame) < MIN_SAMPLES_FOR_FIT:
            return {
                "fitted": False,
                "reason": f"need >={MIN_SAMPLES_FOR_FIT} in-frame samples, have {len(in_frame)}",
                "n_samples": len(in_frame),
            }

        X = np.stack([features(s.px, s.py) for s in in_frame])  # type: ignore[arg-type]
        pan_vec = np.array([s.pan for s in in_frame])
        tilt_vec = np.array([s.tilt for s in in_frame])

        a, *_ = np.linalg.lstsq(X, pan_vec, rcond=None)
        b, *_ = np.linalg.lstsq(X, tilt_vec, rcond=None)

        self.coeffs_pan = a
        self.coeffs_tilt = b
        self.rms_pan = float(np.sqrt(np.mean((X @ a - pan_vec) ** 2)))
        self.rms_tilt = float(np.sqrt(np.mean((X @ b - tilt_vec) ** 2)))

        return {
            "fitted": True,
            "n_samples": len(in_frame),
            "rms_pan_deg": self.rms_pan,
            "rms_tilt_deg": self.rms_tilt,
        }

    def predict(self, px: float, py: float) -> tuple[float, float] | None:
        if not self.fitted:
            return None
        feats = features(px, py)
        pan = float(feats @ self.coeffs_pan)  # type: ignore[operator]
        tilt = float(feats @ self.coeffs_tilt)  # type: ignore[operator]
        return pan, tilt

    def status(self) -> dict[str, Any]:
        return {
            "n_samples": len(self.samples),
            "n_in_frame": self.n_in_frame,
            "fitted": self.fitted,
            "rms_pan_deg": self.rms_pan,
            "rms_tilt_deg": self.rms_tilt,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [asdict(s) for s in self.samples],
            "coeffs_pan": self.coeffs_pan.tolist() if self.coeffs_pan is not None else None,
            "coeffs_tilt": self.coeffs_tilt.tolist() if self.coeffs_tilt is not None else None,
            "rms_pan_deg": self.rms_pan,
            "rms_tilt_deg": self.rms_tilt,
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
            cal.samples.append(AimSample(**s))
        if data.get("coeffs_pan") is not None:
            cal.coeffs_pan = np.array(data["coeffs_pan"])
        if data.get("coeffs_tilt") is not None:
            cal.coeffs_tilt = np.array(data["coeffs_tilt"])
        cal.rms_pan = data.get("rms_pan_deg")
        cal.rms_tilt = data.get("rms_tilt_deg")
        return cal
