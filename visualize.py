import cv2
import numpy as np

from config import ACCIDENT_DETECTION_CONFIDENCE, NMS_THRESHOLD


def _draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y = max(18, y)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _filter_damage_regions(regions: list[dict]) -> list[dict]:
    if not regions:
        return []
    boxes = [r["bbox"] for r in regions]
    confidences = [float(r.get("confidence", 0.0)) for r in regions]

    # cv2.dnn.NMSBoxes expects boxes as [x,y,w,h] and confidences
    try:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, float(ACCIDENT_DETECTION_CONFIDENCE), float(NMS_THRESHOLD))
    except Exception:
        # If NMS fails for any reason, fall back to simple filtering
        filtered = [r for r in regions if r.get("confidence", 0.0) >= ACCIDENT_DETECTION_CONFIDENCE]
        kept = sorted(filtered, key=lambda r: r.get("confidence", 0.0), reverse=True)
        # If multiple overlapping detections remain, merge them into a single bbox
        if len(kept) <= 1:
            return kept[:1]
        xs = [r["bbox"][0] for r in kept]
        ys = [r["bbox"][1] for r in kept]
        x2s = [r["bbox"][0] + r["bbox"][2] for r in kept]
        y2s = [r["bbox"][1] + r["bbox"][3] for r in kept]
        merged = {
            "bbox": [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)],
            "confidence": max(r.get("confidence", 0.0) for r in kept),
            "label": ",".join(sorted({r.get("label", "damage") for r in kept})),
        }
        return [merged]

    if len(indices) == 0:
        return []

    # indices can be a list of lists or a flat list depending on OpenCV build
    flat = np.array(indices).flatten().tolist()
    kept = [regions[i] for i in flat]
    # keep strongest up to 5 before merging to be conservative
    kept = sorted(kept, key=lambda r: r.get("confidence", 0.0), reverse=True)[:5]

    if len(kept) <= 1:
        return kept

    # Merge multiple nearby/overlapping boxes into a single bounding box
    xs = [r["bbox"][0] for r in kept]
    ys = [r["bbox"][1] for r in kept]
    x2s = [r["bbox"][0] + r["bbox"][2] for r in kept]
    y2s = [r["bbox"][1] + r["bbox"][3] for r in kept]

    merged_bbox = [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)]
    merged_conf = max(r.get("confidence", 0.0) for r in kept)
    merged_label = ",".join(sorted({r.get("label", "damage") for r in kept}))

    return [{
        "bbox": [int(merged_bbox[0]), int(merged_bbox[1]), int(merged_bbox[2]), int(merged_bbox[3])],
        "confidence": round(float(merged_conf), 4),
        "label": merged_label,
    }]


def draw_results(image, report: dict):
    output = image.copy()

    # Only draw vehicle boxes when accident is detected to keep the view clean
    if report.get("accident_detected"):
        for vehicle in report.get("vehicles_involved", []):
            x, y, w, h = vehicle["bbox"]
            label = f'{vehicle["label"]} {vehicle["confidence"]:.2f}'
            cv2.rectangle(output, (x, y), (x + w, y + h), (40, 180, 40), 2)
            _draw_label(output, label, x, y - 8, (40, 180, 40))

    # Filter damage regions (remove duplicates / low-confidence & limit count)
    damage_regions = _filter_damage_regions(report.get("damage_regions", []))
    for region in damage_regions:
        x, y, w, h = region["bbox"]
        label = f'{region.get("label", "damage")} {region.get("confidence", 0):.2f}'
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 140, 255), 2)
        _draw_label(output, label, x, y - 8, (0, 140, 255))

    # Only draw contact evidence if accident detected
    if report.get("accident_detected"):
        for contact in report.get("contact_regions", []):
            x, y, w, h = contact["bbox"]
            overlay = output.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
            _draw_label(output, "contact evidence", x, y - 8, (0, 0, 255))

    status = "ACCIDENT LIKELY" if report.get("accident_detected") else "NO ACCIDENT"
    title = f'{status} ({report.get("accident_probability", 0.0):.2f})'
    cv2.rectangle(output, (10, 10), (330, 48), (0, 0, 0), -1)
    cv2.putText(output, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return output
