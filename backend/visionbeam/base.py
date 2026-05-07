"""
Shared abstract base class for target-selection methods.

Both the production tracker (visionbeam.tracker.HybridMethod) and the
research baselines (evaluation.methods.*) implement this interface so the
evaluation harness can compare them apples-to-apples on the same clips.
"""

from abc import ABC, abstractmethod

import numpy as np


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
