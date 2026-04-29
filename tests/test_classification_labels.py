from analytics.density_estimator import ROIShape, classify_detection as classify_density_detection, compute_frame_density
from analytics.occupancy_estimator import ROI, classify_detection as classify_occupancy_detection, compute_frame_occupancy
from core.frame_processor import classify_track
from core.types import BBox, Track


def test_custom_traffic_labels_are_classified_by_label_before_class_id() -> None:
    motor = Track(track_id=1, bbox=BBox(0, 0, 10, 10), cls_id=0, label="Motor Vehicle", score=0.9, frame_index=0)
    non_motor = Track(track_id=2, bbox=BBox(0, 0, 10, 10), cls_id=1, label="Non_motorized Vehicle", score=0.9, frame_index=0)
    pedestrian = Track(track_id=3, bbox=BBox(0, 0, 10, 10), cls_id=2, label="Pedestrian", score=0.9, frame_index=0)
    traffic_light = Track(track_id=4, bbox=BBox(0, 0, 10, 10), cls_id=5, label="Traffic Light-Green Light", score=0.9, frame_index=0)

    assert classify_track(motor) == "motor_vehicle"
    assert classify_track(non_motor) == "non_motor"
    assert classify_track(pedestrian) == "pedestrian"
    assert classify_track(traffic_light) == "other"


def test_density_estimator_ignores_custom_traffic_light_classes() -> None:
    roi = ROIShape([(0, 0), (100, 0), (100, 100), (0, 100)])
    frame = {
        "frame_index": 0,
        "detections": [
            {"class_id": 0, "label": "Motor Vehicle", "bbox": [10, 10, 20, 20]},
            {"class_id": 1, "label": "Non_motorized Vehicle", "bbox": [30, 10, 40, 20]},
            {"class_id": 2, "label": "Pedestrian", "bbox": [50, 10, 60, 20]},
            {"class_id": 5, "label": "Traffic Light-Green Light", "bbox": [70, 10, 80, 20]},
        ],
    }

    stats = compute_frame_density(frame, roi)

    assert classify_density_detection(frame["detections"][3]) == "other"
    assert stats.vehicle_count == 1
    assert stats.non_motor_count == 1
    assert stats.pedestrian_count == 1


def test_occupancy_estimator_ignores_custom_traffic_light_classes() -> None:
    roi = ROI([(0, 0), (100, 0), (100, 100), (0, 100)])
    frame = {
        "frame_index": 0,
        "detections": [
            {"class_id": 0, "label": "Motor Vehicle", "bbox": [10, 10, 20, 20]},
            {"class_id": 5, "label": "Traffic Light-Green Light", "bbox": [70, 10, 80, 20]},
        ],
    }

    stats = compute_frame_occupancy(frame, roi)

    assert classify_occupancy_detection(frame["detections"][1]) == "other"
    assert stats.vehicle_count == 1
    assert stats.non_motor_count == 0
    assert stats.pedestrian_count == 0
