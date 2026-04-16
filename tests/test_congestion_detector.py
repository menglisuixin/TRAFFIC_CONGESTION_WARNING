import pytest

from analytics.congestion_detector import CongestionDetector, CongestionRule, congestion_level
from analytics.state_machine import CongestionStateMachine, LevelThreshold


def test_congestion_level_is_clamped() -> None:
    assert congestion_level(2.0, -1.0) == 1.0
    assert congestion_level(-1.0, 2.0) == 0.0


def test_detector_returns_normal_when_no_rule_matches() -> None:
    detector = CongestionDetector()

    level, warning = detector.update(
        roi_vehicle_count=1,
        low_speed_ratio=0.0,
        occupancy_ratio=0.1,
    )

    assert level == "normal"
    assert warning is False


def test_detector_switches_slow_congested_severe_by_highest_matching_rule() -> None:
    detector = CongestionDetector(
        rules=[
            CongestionRule(1, 0.10, 0.10, 1, "slow"),
            CongestionRule(2, 0.30, 0.30, 1, "congested"),
            CongestionRule(3, 0.60, 0.60, 1, "severe"),
        ]
    )

    assert detector.update(1, 0.10, 0.10) == ("slow", True)
    assert detector.update(2, 0.30, 0.30) == ("congested", True)
    assert detector.update(3, 0.60, 0.60) == ("severe", True)


def test_detector_requires_min_duration_frames() -> None:
    detector = CongestionDetector(
        rules=[CongestionRule(3, 0.60, 0.60, 3, "severe")]
    )

    assert detector.update(3, 0.60, 0.60) == ("normal", False)
    assert detector.update(3, 0.60, 0.60) == ("normal", False)
    assert detector.update(3, 0.60, 0.60) == ("severe", True)


def test_detector_duration_resets_when_metrics_drop_to_normal() -> None:
    detector = CongestionDetector(
        rules=[CongestionRule(3, 0.60, 0.60, 2, "severe")]
    )

    assert detector.update(3, 0.60, 0.60) == ("normal", False)
    assert detector.update(0, 0.0, 0.0) == ("normal", False)
    assert detector.update(3, 0.60, 0.60) == ("normal", False)
    assert detector.update(3, 0.60, 0.60) == ("severe", True)


def test_state_machine_requires_enter_duration() -> None:
    machine = CongestionStateMachine(
        thresholds={"slow": LevelThreshold(enter=0.5, exit=0.3)},
        min_enter_frames=2,
        min_exit_frames=1,
    )

    assert machine.update(0.6) == "normal"
    assert machine.update(0.6) == "slow"


def test_state_machine_uses_exit_hysteresis_and_duration() -> None:
    machine = CongestionStateMachine(
        thresholds={"slow": LevelThreshold(enter=0.5, exit=0.3)},
        min_enter_frames=1,
        min_exit_frames=2,
    )

    assert machine.update(0.6) == "slow"
    assert machine.update(0.4) == "slow"
    assert machine.update(0.2) == "slow"
    assert machine.update(0.2) == "normal"


def test_state_machine_can_step_down_from_severe_to_congested() -> None:
    machine = CongestionStateMachine(
        thresholds={
            "slow": LevelThreshold(enter=0.3, exit=0.2),
            "congested": LevelThreshold(enter=0.6, exit=0.45),
            "severe": LevelThreshold(enter=0.85, exit=0.70),
        },
        min_enter_frames=1,
        min_exit_frames=1,
    )

    assert machine.update(0.9) == "severe"
    assert machine.update(0.65) == "congested"


def test_state_machine_rejects_invalid_min_frames() -> None:
    with pytest.raises(ValueError):
        CongestionStateMachine(min_enter_frames=0)
