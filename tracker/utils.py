"""Tracker utility classes and IoU matching helpers."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from core.types import BBox, Detection, Track


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    cls_id: int
    label: str
    score: float
    frame_index: int
    missing: int = 0

    def to_track(self, frame_index: int) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            cls_id=self.cls_id,
            label=self.label,
            score=self.score,
            frame_index=frame_index,
        )


class IoUTrackerCore:
    """Small reusable IoU tracker for fallback and ByteTrack-style matching."""

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 30) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missing < 0:
            raise ValueError("max_missing must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_track_id = 1
        self.states: Dict[int, TrackState] = {}

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        return_missing: bool = False,
    ) -> List[Track]:
        matched_state_ids: Set[int] = set()
        output_tracks: List[Track] = []

        for detection in detections:
            track_id = self.match_detection(detection, matched_state_ids, self.iou_threshold)
            if track_id is None:
                track_id = self.allocate_track_id()
            state = self.update_state(track_id, detection, frame_index)
            matched_state_ids.add(track_id)
            output_tracks.append(state.to_track(frame_index))

        missing_tracks = self.age_unmatched(matched_state_ids, frame_index)
        if return_missing:
            output_tracks.extend(missing_tracks)
        return output_tracks

    def update_byte_style(
        self,
        high_detections: Sequence[Detection],
        low_detections: Sequence[Detection],
        frame_index: int,
        low_iou_threshold: float,
        return_missing: bool = False,
    ) -> List[Track]:
        matched_state_ids: Set[int] = set()
        output_tracks: List[Track] = []

        for detection in high_detections:
            track_id = self.match_detection(detection, matched_state_ids, self.iou_threshold)
            if track_id is None:
                track_id = self.allocate_track_id()
            state = self.update_state(track_id, detection, frame_index)
            matched_state_ids.add(track_id)
            output_tracks.append(state.to_track(frame_index))

        for detection in low_detections:
            track_id = self.match_detection(detection, matched_state_ids, low_iou_threshold)
            if track_id is None:
                continue
            state = self.update_state(track_id, detection, frame_index)
            matched_state_ids.add(track_id)
            output_tracks.append(state.to_track(frame_index))

        missing_tracks = self.age_unmatched(matched_state_ids, frame_index)
        if return_missing:
            output_tracks.extend(missing_tracks)
        return output_tracks

    def match_detection(
        self,
        detection: Detection,
        excluded_track_ids: Set[int],
        threshold: float,
    ) -> Optional[int]:
        best_track_id: Optional[int] = None
        best_iou = threshold
        for track_id, state in self.states.items():
            if track_id in excluded_track_ids:
                continue
            if state.cls_id != detection.cls_id:
                continue
            iou = bbox_iou(detection.bbox, state.bbox)
            if iou >= best_iou:
                best_iou = iou
                best_track_id = track_id
        return best_track_id

    def update_state(self, track_id: int, detection: Detection, frame_index: int) -> TrackState:
        state = TrackState(
            track_id=track_id,
            bbox=detection.bbox,
            cls_id=detection.cls_id,
            label=detection.label,
            score=detection.conf,
            frame_index=frame_index,
            missing=0,
        )
        self.states[track_id] = state
        return state

    def age_unmatched(self, matched_track_ids: Set[int], frame_index: int) -> List[Track]:
        stale_ids: List[int] = []
        missing_tracks: List[Track] = []
        for track_id, state in self.states.items():
            if track_id in matched_track_ids:
                continue
            state.missing += 1
            if state.missing > self.max_missing:
                stale_ids.append(track_id)
            else:
                missing_tracks.append(state.to_track(frame_index))
        for track_id in stale_ids:
            del self.states[track_id]
        return missing_tracks

    def allocate_track_id(self) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def clear(self) -> None:
        self.next_track_id = 1
        self.states.clear()


def split_detections_by_confidence(
    detections: Sequence[Detection],
    high_threshold: float,
    low_threshold: float,
) -> Tuple[List[Detection], List[Detection]]:
    high: List[Detection] = []
    low: List[Detection] = []
    for detection in detections:
        if detection.conf >= high_threshold:
            high.append(detection)
        elif detection.conf >= low_threshold:
            low.append(detection)
    return high, low


def bbox_iou(first: BBox, second: BBox) -> float:
    inter_x1 = max(first.x1, second.x1)
    inter_y1 = max(first.y1, second.y1)
    inter_x2 = min(first.x2, second.x2)
    inter_y2 = min(first.y2, second.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = first.area + second.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area
