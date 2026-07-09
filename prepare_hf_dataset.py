import argparse
import json
from pathlib import Path

import yaml

from config import ACCIDENT_CLASS_NAMES, HF_DATASET_ID, HF_YOLO_DATASET_DIR


SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def coco_xywh_to_yolo(box: list[float], image_width: int, image_height: int) -> list[float]:
    x, y, width, height = box
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    return [
        x_center,
        y_center,
        width / image_width,
        height / image_height,
    ]


def clamp_yolo_box(values: list[float]) -> list[float]:
    return [min(1.0, max(0.0, value)) for value in values]


def write_yolo_example(example: dict, output_dir: Path, split: str, index: int) -> int:
    image = example["image"].convert("RGB")
    image_width, image_height = image.size
    split_name = SPLIT_MAP[split]
    image_path = output_dir / "images" / split_name / f"{index:06d}.jpg"
    label_path = output_dir / "labels" / split_name / f"{index:06d}.txt"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(image_path, quality=95)

    boxes = example["objects"]["bbox"]
    categories = example["objects"]["category"]
    lines = []

    for category, box in zip(categories, boxes):
        yolo_box = clamp_yolo_box(coco_xywh_to_yolo(box, image_width, image_height))
        line = f"{int(category)} " + " ".join(f"{value:.6f}" for value in yolo_box)
        lines.append(line)

    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def write_data_yaml(output_dir: Path) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(ACCIDENT_CLASS_NAMES)},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def prepare_dataset(output_dir: Path, max_items: int | None = None) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET_ID)
    summary = {
        "dataset_id": HF_DATASET_ID,
        "output_dir": str(output_dir),
        "classes": ACCIDENT_CLASS_NAMES,
        "splits": {},
    }

    for split in SPLIT_MAP:
        written_images = 0
        written_boxes = 0
        split_dataset = dataset[split]
        limit = min(max_items, len(split_dataset)) if max_items else len(split_dataset)

        for index, example in enumerate(split_dataset.select(range(limit))):
            written_boxes += write_yolo_example(example, output_dir, split, index)
            written_images += 1

        summary["splits"][SPLIT_MAP[split]] = {
            "images": written_images,
            "boxes": written_boxes,
        }

    write_data_yaml(output_dir)
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face accident dataset and export YOLO format.")
    parser.add_argument("--output-dir", default=str(HF_YOLO_DATASET_DIR), help="YOLO dataset output directory.")
    parser.add_argument("--max-items", type=int, help="Limit examples per split for quick smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(Path(args.output_dir), args.max_items)
    print(json.dumps(summary, indent=2))
    print(f"YOLO data config: {Path(args.output_dir) / 'data.yaml'}")


if __name__ == "__main__":
    main()
