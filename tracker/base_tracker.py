"""Tracker abstraction."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.types import Detection, Track


class BaseTracker(ABC):
    """Common interface for tracking backends."""

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        """Update tracker state and return tracks visible in the current frame."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all tracker state."""
