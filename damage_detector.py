from pathlib import Path

from config import ACCIDENT_DETECTION_CONFIDENCE, ACCIDENT_MODEL_PATHS


def _find_default_model_path() -> Path:
    for path in ACCIDENT_MODEL_PATHS:
        if path.exists():
            return path
    return ACCIDENT_MODEL_PATHS[0]


class DamageDetector:
    """Accident-region detector trained from the Hugging Face dataset.

    The Hugging Face dataset uses accident/non_accident region labels. This class
    keeps inference optional: if no trained YOLO model exists, the app still runs
    with vehicle reasoning only.
    """

    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else _find_default_model_path()
        self.model = None

        if self.model_path.exists():
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def detect(self, image) -> list[dict]:
        if not self.enabled:
            return []

        results = self.model.predict(image, conf=ACCIDENT_DETECTION_CONFIDENCE, verbose=False)
        detections = []

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "label": label,
                        "confidence": round(confidence, 4),
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2 - x1),
                            int(y2 - y1),
                        ],
                    }
                )

        return detections
