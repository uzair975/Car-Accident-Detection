from itertools import combinations
from math import sqrt

from config import ACCIDENT_THRESHOLD


def _area(box: list[int]) -> int:
    return max(0, box[2]) * max(0, box[3])


def _intersection(box_a: list[int], box_b: list[int]) -> list[int] | None:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2 - x1, y2 - y1]


def _iou(box_a: list[int], box_b: list[int]) -> float:
    overlap = _intersection(box_a, box_b)
    if overlap is None:
        return 0.0

    intersection_area = _area(overlap)
    union_area = _area(box_a) + _area(box_b) - intersection_area
    return intersection_area / union_area if union_area else 0.0


def _center_distance(box_a: list[int], box_b: list[int], image_shape) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    center_a = (ax + aw / 2, ay + ah / 2)
    center_b = (bx + bw / 2, by + bh / 2)
    distance = sqrt((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2)
    image_h, image_w = image_shape[:2]
    diagonal = sqrt(image_w**2 + image_h**2)
    return distance / diagonal if diagonal else 1.0


def classify_accident(
    vehicles: list[dict],
    damage_regions: list[dict],
    image_shape,
) -> dict:
    pair_features = []
    max_iou = 0.0
    min_center_distance = 1.0
    contact_regions = []

    for vehicle_a, vehicle_b in combinations(vehicles, 2):
        box_a = vehicle_a["bbox"]
        box_b = vehicle_b["bbox"]
        iou = _iou(box_a, box_b)
        distance = _center_distance(box_a, box_b, image_shape)
        overlap = _intersection(box_a, box_b)

        max_iou = max(max_iou, iou)
        min_center_distance = min(min_center_distance, distance)

        if overlap:
            contact_regions.append(
                {
                    "bbox": overlap,
                    "vehicles": [vehicle_a["label"], vehicle_b["label"]],
                    "iou": round(iou, 4),
                }
            )

        pair_features.append(
            {
                "vehicles": [vehicle_a["label"], vehicle_b["label"]],
                "iou": round(iou, 4),
                "center_distance": round(distance, 4),
            }
        )

    score = 0.0
    evidence = []

    if len(vehicles) >= 2:
        score += 0.15
        evidence.append("multiple vehicles detected")

    if max_iou > 0.03:
        score += min(0.35, max_iou * 1.5)
        evidence.append("vehicle bounding boxes overlap")
    elif len(vehicles) >= 2 and min_center_distance < 0.22:
        score += 0.18
        evidence.append("vehicles are unusually close")

    accident_regions = [
        region for region in damage_regions if region.get("label") in {"accident", "damage", "debris"}
    ]

    if accident_regions:
        best_accident_confidence = max(region.get("confidence", 0.0) for region in accident_regions)
        avg_confidence = sum(region.get("confidence", 0.0) for region in accident_regions) / len(accident_regions)
        count = len(accident_regions)

        # Give stronger weight to trained accident-region detections so a trained
        # model can decisively influence the final accident probability.
        score += min(0.8, 0.35 + best_accident_confidence * 0.5 + avg_confidence * 0.1 + 0.03 * count)
        evidence.append("trained accident-region model detected accident evidence")
    elif damage_regions:
        evidence.append("trained model detected only non-accident regions")

    accident_probability = round(min(score, 0.99), 4)
    accident_detected = accident_probability >= ACCIDENT_THRESHOLD

    if not evidence:
        evidence.append("no strong accident evidence detected")

    return {
        "accident_detected": accident_detected,
        "accident_probability": accident_probability,
        "threshold": ACCIDENT_THRESHOLD,
        "vehicles_involved": vehicles,
        "damage_regions": damage_regions,
        "contact_regions": contact_regions,
        "pair_features": pair_features,
        "explanation": "; ".join(evidence),
    }
