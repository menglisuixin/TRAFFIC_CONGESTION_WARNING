from pathlib import Path

import torch

from core.pipeline import resolve_model_classes


class DummyModel:
    def __init__(self, names):
        self.names = names


def save_dummy_weights(path: Path, names) -> None:
    torch.save({"model": DummyModel(names)}, path)


def test_resolve_model_classes_keeps_coco_car_truck_filter(tmp_path: Path) -> None:
    weights = tmp_path / "coco.pt"
    save_dummy_weights(weights, {2: "car", 7: "truck"})
    config = {"model": {"auto_classes": True, "classes": [2, 7]}}

    assert resolve_model_classes(config, str(weights)) == [2, 7]


def test_resolve_model_classes_keeps_only_motor_vehicle_for_custom_traffic_weights(tmp_path: Path) -> None:
    weights = tmp_path / "custom.pt"
    save_dummy_weights(weights, {0: "Motor Vehicle", 1: "Non_motorized Vehicle", 2: "Pedestrian"})
    config = {"model": {"auto_classes": True, "classes": [2, 7]}}

    assert resolve_model_classes(config, str(weights)) == [0]


def test_resolve_model_classes_can_force_filter_for_custom_weights(tmp_path: Path) -> None:
    weights = tmp_path / "custom.pt"
    save_dummy_weights(weights, {0: "Motor Vehicle", 1: "Non_motorized Vehicle", 2: "Pedestrian"})
    config = {"model": {"auto_classes": False, "classes": [0, 1]}}

    assert resolve_model_classes(config, str(weights)) == [0, 1]
