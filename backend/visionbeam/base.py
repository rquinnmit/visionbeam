from abc import ABC, abstractmethod

import numpy as np


class TargetMethod(ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        ...
