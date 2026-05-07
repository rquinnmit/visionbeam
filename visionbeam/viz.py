"""
Tiny visualization helper.

Drop `save_step(image, name)` anywhere in the pipeline to dump that
image to `data/viz_steps/`. Intended for generating figures and debugging
the CV pipeline; not meant for production output.
"""

from collections import defaultdict
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


VIZ_DIR = Path(__file__).resolve().parent.parent / "data" / "viz_steps"

_counters: dict[str, int] = defaultdict(int)
_lock = Lock()


def save_step(image: np.ndarray | None, name: str) -> None:
    """Save `image` to `data/viz_steps/{name}_{counter:06d}.png`.

    Per-name monotonic counter so repeated calls produce a sequence
    instead of overwriting. No-op if `image` is None.
    """
    if image is None:
        return
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        _counters[name] += 1
        idx = _counters[name]
    cv2.imwrite(str(VIZ_DIR / f"{name}_{idx:06d}.png"), image)
