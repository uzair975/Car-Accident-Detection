import os
from pathlib import Path

import cv2
import requests

from config import COCO_NAMES, YOLO_CFG, YOLO_URLS, YOLO_WEIGHTS


def download_file(url: str, path: Path) -> None:
    if path.exists():
        return

    print(f"Downloading {path.name}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def ensure_yolo_files() -> None:
    for path, url in YOLO_URLS.items():
        download_file(url, path)


def load_classes(names_path: Path = COCO_NAMES) -> list[str]:
    if not names_path.exists():
        raise FileNotFoundError(f"Missing class file: {names_path}")

    return names_path.read_text(encoding="utf-8").strip().splitlines()


def load_yolo_model():
    ensure_yolo_files()

    if not YOLO_WEIGHTS.exists() or not YOLO_CFG.exists():
        raise FileNotFoundError("YOLO model files are missing.")

    net = cv2.dnn.readNet(str(YOLO_WEIGHTS), str(YOLO_CFG))

    if os.getenv("OPENCV_DNN_CUDA") == "1":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return net, load_classes()
