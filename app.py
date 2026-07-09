import argparse
import json
from pathlib import Path

import cv2

from accident_classifier import classify_accident
from config import OUTPUT_IMAGE_DIR, OUTPUT_REPORT_DIR
from damage_detector import DamageDetector
from detect_vehicles import VehicleDetector
from model_loader import load_yolo_model
from visualize import draw_results


def run_image(
    image_path: Path,
    output_path: Path | None = None,
    report_path: Path | None = None,
    accident_model: Path | None = None,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    net, classes = load_yolo_model()
    vehicles = VehicleDetector(net, classes).detect(image)
    damage_detector = DamageDetector(accident_model)
    damage_regions = damage_detector.detect(image)
    report = classify_accident(vehicles, damage_regions, image.shape)
    report["model_status"] = {
        "vehicle_detector": "yolov4_coco",
        "accident_region_detector": (
            str(damage_detector.model_path) if damage_detector.enabled else "not_trained_or_not_found"
        ),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), draw_results(image, report))

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def default_output_paths(image_path: Path) -> tuple[Path, Path]:
    stem = image_path.stem
    return OUTPUT_IMAGE_DIR / f"{stem}_result.jpg", OUTPUT_REPORT_DIR / f"{stem}_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect likely vehicle accident evidence in an image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output", help="Path to save the annotated image.")
    parser.add_argument("--json", help="Path to save the JSON report.")
    parser.add_argument("--accident-model", help="Path to trained YOLO accident-region weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    default_image, default_report = default_output_paths(image_path)
    output_path = Path(args.output) if args.output else default_image
    report_path = Path(args.json) if args.json else default_report

    accident_model = Path(args.accident_model) if args.accident_model else None
    report = run_image(image_path, output_path, report_path, accident_model)
    print(json.dumps(report, indent=2))
    print(f"Annotated image: {output_path}")
    print(f"JSON report: {report_path}")


if __name__ == "__main__":
    main()
