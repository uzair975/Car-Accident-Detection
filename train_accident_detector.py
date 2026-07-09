import argparse
from pathlib import Path

from config import HF_YOLO_DATASET_DIR


def check_training_environment() -> None:
    import numpy as np
    import torch

    major_version = int(np.__version__.split(".", maxsplit=1)[0])
    if major_version >= 2 and torch.__version__.startswith("2.2."):
        raise RuntimeError(
            "This Torch build is incompatible with NumPy 2.x. "
            'Run: python -m pip install "numpy<2" --force-reinstall'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO accident-region detector.")
    parser.add_argument("--data", default=str(HF_YOLO_DATASET_DIR / "data.yaml"), help="YOLO data.yaml path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="accident_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_training_environment()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python prepare_hf_dataset.py"
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )


if __name__ == "__main__":
    main()
