"""
Spatial translation module.

Converts floor coordinates to pan/tilt angles given
the light's known 3D mount position.
"""

import math
import json
from dataclasses import dataclass, asdict


@dataclass
class LightMount:
    """
    The light's position and orientation in the floor coordinate system.

    x, y, z: Mount position in meters (z = height above floor).
    pan_offset: Fixture pan angle (degrees) corresponding to world 0 degrees.
    tilt_offset: Fixture tilt angle (degrees) corresponding to straight down.

    Defaults to placing the fixture at the center of its pan/tilt range.
    """
    x: float
    y: float
    z: float
    pan_offset: float = 270.0
    tilt_offset: float = 135.0

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LightMount":
        with open(path) as f:
            return cls(**json.load(f))


class TargetSmoother:
    """
    Exponential moving average on floor coordinates.

    Prevents frame-to-frame jitter from reaching the light.
    Lower alpha = smoother/slower, higher alpha = more responsive.
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        
        return self._x, self._y

    def reset(self):
        self._x = None
        self._y = None


def floor_to_pan_tilt(
    target_x: float,
    target_y: float,
    mount: LightMount
) -> tuple[float, float]:
    """
    Converts a floor coordinate to fixture pan/tilt angles.

    Args:
        target_x: Target position on the floor (meters).
        target_y: Target position on the floor (meters).
        mount: The light's calibrated mount parameters.

    Returns:
        (pan, tilt) in degrees
    """
    dx = target_x - mount.x
    dy = target_y - mount.y
    horizontal_distance = math.hypot(dx, dy)

    world_pan = math.degrees(math.atan2(dy, dx))
    world_tilt = math.degrees(math.atan2(horizontal_distance, mount.z))

    fixture_pan = world_pan + mount.pan_offset
    fixture_tilt = world_tilt + mount.tilt_offset

    return fixture_pan, fixture_tilt
