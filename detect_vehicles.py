from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    DETECTION_CONFIDENCE,
    INPUT_SIZE,
    NMS_CONFIDENCE,
    NMS_THRESHOLD,
    VEHICLE_CLASSES,
)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[int]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
        }


class VehicleDetector:
    def __init__(self, net, classes: list[str]):
        self.net = net
        self.classes = classes
        self.output_layers = net.getUnconnectedOutLayersNames()

    def detect(self, image) -> list[dict]:
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=INPUT_SIZE,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        labels: list[str] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                label = self.classes[class_id]

                if confidence < DETECTION_CONFIDENCE or label not in VEHICLE_CLASSES:
                    continue

                center_x, center_y, box_w, box_h = (
                    detection[:4] * np.array([width, height, width, height])
                ).astype(int)
                x = max(0, int(center_x - box_w / 2))
                y = max(0, int(center_y - box_h / 2))
                box_w = min(int(box_w), width - x)
                box_h = min(int(box_h), height - y)

                boxes.append([x, y, box_w, box_h])
                confidences.append(confidence)
                labels.append(label)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, NMS_CONFIDENCE, NMS_THRESHOLD)
        if len(indices) == 0:
            return []

        return [
            Detection(labels[int(i)], confidences[int(i)], boxes[int(i)]).to_dict()
            for i in np.array(indices).flatten()
        ]
