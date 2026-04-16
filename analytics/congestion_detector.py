"""Rule-based traffic congestion detection."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

LEVEL_ORDER = {"normal": 0, "slow": 1, "congested": 2, "severe": 3}


@dataclass(frozen=True)
class CongestionRule:
    """Thresholds that must all be met before a congestion level is active."""

    min_roi_vehicle_count: int
    min_low_speed_ratio: float
    min_occupancy_ratio: float
    min_duration_frames: int
    level: str


class CongestionDetector:
    """Matches frame metrics against congestion rules from high to low severity."""

    def __init__(self, rules: Optional[Iterable[CongestionRule]] = None) -> None:
        self.rules: List[CongestionRule] = sorted(
            list(rules) if rules is not None else self.default_rules(),
            key=lambda rule: LEVEL_ORDER.get(rule.level, -1),
            reverse=True,
        )
        self._candidate_level = "normal"
        self._candidate_frames = 0
        self._active_level = "normal"

    @staticmethod
    def default_rules() -> List[CongestionRule]:
        return [
            CongestionRule(5, 0.20, 0.25, 1, "slow"),
            CongestionRule(8, 0.40, 0.45, 2, "congested"),
            CongestionRule(12, 0.65, 0.70, 3, "severe"),
        ]

    def update(
        self,
        roi_vehicle_count: int,
        low_speed_ratio: float,
        occupancy_ratio: float,
    ) -> Tuple[str, bool]:
        """Return the current congestion level and whether a warning is active."""

        matched_rule = self._match_rule(
            roi_vehicle_count=roi_vehicle_count,
            low_speed_ratio=low_speed_ratio,
            occupancy_ratio=occupancy_ratio,
        )

        if matched_rule is None:
            self._candidate_level = "normal"
            self._candidate_frames = 0
            self._active_level = "normal"
            return "normal", False

        if matched_rule.level == self._candidate_level:
            self._candidate_frames += 1
        else:
            self._candidate_level = matched_rule.level
            self._candidate_frames = 1

        if self._candidate_frames >= max(1, matched_rule.min_duration_frames):
            self._active_level = matched_rule.level

        return self._active_level, self._active_level != "normal"

    def reset(self) -> None:
        self._candidate_level = "normal"
        self._candidate_frames = 0
        self._active_level = "normal"

    def _match_rule(
        self,
        roi_vehicle_count: int,
        low_speed_ratio: float,
        occupancy_ratio: float,
    ) -> Optional[CongestionRule]:
        for rule in self.rules:
            if (
                roi_vehicle_count >= rule.min_roi_vehicle_count
                and low_speed_ratio >= rule.min_low_speed_ratio
                and occupancy_ratio >= rule.min_occupancy_ratio
            ):
                return rule
        return None


def congestion_level(density: float, speed_ratio: float) -> float:
    """Backward-compatible numeric congestion score helper."""

    return max(0.0, min(1.0, 0.6 * density + 0.4 * (1.0 - speed_ratio)))
