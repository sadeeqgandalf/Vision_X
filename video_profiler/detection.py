# COCO detection via Faster R-CNN ResNet50 FPN. Per-frame (class, score, bbox).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torchvision


# COCO 80 classes; index 0 = background
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class FrameDetections:
    frame_index: int
    timestamp_sec: float
    detections: list[Detection]


def _get_model(device: torch.device, min_score: float = 0.5) -> torch.nn.Module:
    """Faster R-CNN ResNet50 FPN, COCO weights."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    model.to(device)
    return model


def _preprocess(frame: np.ndarray) -> torch.Tensor:
    """BGR HWC uint8 -> RGB CHW float [0,1], batch dim 1."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def run_detection(
    frames: list[Any],
    get_frame: Any = None,
    get_index: Any = None,
    get_timestamp: Any = None,
    device: str | torch.device | None = None,
    min_score: float = 0.5,
    classes_of_interest: list[str] | None = None,
) -> list[FrameDetections]:
    """Run COCO detection per frame. device: cuda|cpu|None (auto). min_score: conf threshold."""
    if get_frame is None:
        get_frame = lambda o: o.frame
    if get_index is None:
        get_index = lambda o: o.index
    if get_timestamp is None:
        get_timestamp = lambda o: o.timestamp_sec

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = _get_model(device, min_score)

    if classes_of_interest is None:
        classes_of_interest = set(COCO_CLASSES)
    else:
        classes_of_interest = set(c.lower() for c in classes_of_interest)

    result: list[FrameDetections] = []
    for obj in frames:
        frame = get_frame(obj)
        idx = get_index(obj)
        ts = get_timestamp(obj)
        x = _preprocess(frame).to(device)
        with torch.no_grad():
            preds = model(x)
        boxes = preds[0]["boxes"].cpu().numpy()
        labels = preds[0]["labels"].cpu().numpy()
        scores = preds[0]["scores"].cpu().numpy()
        dets = []
        for box, label, score in zip(boxes, labels, scores):
            if score < min_score:
                continue
            if label < 1 or label > len(COCO_CLASSES):  # 1-indexed COCO
                continue
            class_name = COCO_CLASSES[label - 1]
            if class_name not in classes_of_interest:
                continue
            dets.append(
                Detection(
                    class_name=class_name,
                    score=float(score),
                    bbox=tuple(float(x) for x in box),
                )
            )
        result.append(
            FrameDetections(frame_index=idx, timestamp_sec=ts, detections=dets)
        )
    return result
