"""Simple hysteresis state machine for congestion levels."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

LEVEL_ORDER = {"normal": 0, "slow": 1, "congested": 2, "severe": 3}


@dataclass(frozen=True)
class LevelThreshold:
    """Separate enter and exit thresholds for one congestion level."""

    enter: float
    exit: float


class CongestionStateMachine:
    """Stabilizes congestion level changes using hysteresis and duration checks."""

    def __init__(
        self,
        thresholds: Optional[Mapping[str, LevelThreshold]] = None,
        min_enter_frames: int = 2,
        min_exit_frames: int = 2,
        initial_level: str = "normal",
    ) -> None:
        if initial_level not in LEVEL_ORDER:
            raise ValueError("unknown initial_level")
        if min_enter_frames <= 0 or min_exit_frames <= 0:
            raise ValueError("minimum frame counts must be greater than zero")

        self.thresholds: Dict[str, LevelThreshold] = dict(
            thresholds
            if thresholds is not None
            else {
                "slow": LevelThreshold(enter=0.30, exit=0.20),
                "congested": LevelThreshold(enter=0.60, exit=0.45),
                "severe": LevelThreshold(enter=0.85, exit=0.70),
            }
        )
        self.min_enter_frames = min_enter_frames
        self.min_exit_frames = min_exit_frames
        self.current_level = initial_level
        self._candidate_level = initial_level
        self._candidate_frames = 0
        self._exit_frames = 0

    def update(self, score: float) -> str:
        """Update by congestion score and return the stabilized level."""

        desired_level = self._level_for_enter_score(score)
        if LEVEL_ORDER[desired_level] > LEVEL_ORDER[self.current_level]:
            self._exit_frames = 0
            if desired_level == self._candidate_level:
                self._candidate_frames += 1
            else:
                self._candidate_level = desired_level
                self._candidate_frames = 1

            if self._candidate_frames >= self.min_enter_frames:
                self.current_level = desired_level
                self._candidate_frames = 0
            return self.current_level

        self._candidate_level = self.current_level
        self._candidate_frames = 0

        if self.current_level != "normal" and score < self.thresholds[self.current_level].exit:
            self._exit_frames += 1
            if self._exit_frames >= self.min_exit_frames:
                self.current_level = self._level_for_enter_score(score)
                self._exit_frames = 0
        else:
            self._exit_frames = 0

        return self.current_level

    def reset(self, level: str = "normal") -> None:
        if level not in LEVEL_ORDER:
            raise ValueError("unknown level")
        self.current_level = level
        self._candidate_level = level
        self._candidate_frames = 0
        self._exit_frames = 0

    def _level_for_enter_score(self, score: float) -> str:
        level = "normal"
        for name, threshold in self.thresholds.items():
            if score >= threshold.enter and LEVEL_ORDER[name] > LEVEL_ORDER[level]:
                level = name
        return level
