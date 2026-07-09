from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_CFG = PROJECT_ROOT / "yolov4.cfg"
YOLO_WEIGHTS = PROJECT_ROOT / "yolov4.weights"
COCO_NAMES = PROJECT_ROOT / "coco.names"

OUTPUT_IMAGE_DIR = PROJECT_ROOT / "outputs" / "images"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
DATA_DIR = PROJECT_ROOT / "data"
HF_DATASET_ID = "justjuu/traffic-accident-cctv-object-detection"
HF_YOLO_DATASET_DIR = DATA_DIR / "hf_accident_yolo"
ACCIDENT_MODEL_PATHS = [
    PROJECT_ROOT / "runs" / "detect" / "accident_detector" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "accident_detector-2" / "weights" / "best.pt",
]

YOLO_URLS = {
    YOLO_CFG: "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    YOLO_WEIGHTS: "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
    COCO_NAMES: "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}
ACCIDENT_CLASS_NAMES = ["accident", "non_accident"]

INPUT_SIZE = (608, 608)
DETECTION_CONFIDENCE = 0.3
NMS_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.4
ACCIDENT_THRESHOLD = 0.55
ACCIDENT_DETECTION_CONFIDENCE = 0.2
