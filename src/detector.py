"""
src/detector.py
YOLOv8 detection helpers: ROI filtering, minimum-height filtering,
and the NMS pre-merge that fixes the pillar-split double-detection problem.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def load_model(model_name: str, device: str) -> YOLO:
    """Load YOLOv8 model onto the specified device."""
    model = YOLO(model_name)
    model.to(device)
    print(f"✅ {model_name} loaded on {device.upper()}")
    return model


def detect_persons(model: YOLO, frame: np.ndarray,
                   conf_thresh: float, iou_thresh: float) -> sv.Detections:
    """Run YOLOv8 detection for class 0 (person) on a single frame."""
    results = model(frame, classes=[0], conf=conf_thresh,
                    iou=iou_thresh, verbose=False)[0]
    return sv.Detections.from_ultralytics(results)


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Return True if (x, y) is inside the given polygon."""
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def filter_roi(detections: sv.Detections, roi_polygon: np.ndarray) -> sv.Detections:
    """Keep only detections whose centre point falls inside roi_polygon."""
    if len(detections) == 0:
        return detections
    cx = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
    cy = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
    in_roi = np.array([
        point_in_polygon(cx[i], cy[i], roi_polygon)
        for i in range(len(detections))
    ])
    return detections[in_roi]


def filter_min_height(detections: sv.Detections,
                      min_height: int) -> sv.Detections:
    """
    Remove detections shorter than min_height pixels.
    This eliminates ghost/partial-body tracks caused by reflections or
    partially-visible persons at the frame edge.
    """
    if len(detections) == 0:
        return detections
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    return detections[heights >= min_height]


def merge_overlapping_detections(detections: sv.Detections,
                                 iou_thresh: float) -> sv.Detections:
    """
    Merge overlapping bounding boxes BEFORE passing to the tracker.

    This is the primary fix for the pillar-split problem: when a door pillar
    partially occludes a person, YOLO may generate two boxes for the same
    individual. Any pair of boxes with IoU > iou_thresh are merged into a
    single bounding box that is the union of the two.

    Args:
        detections:  Supervision Detections object.
        iou_thresh:  IoU above which two boxes are considered the same person.

    Returns:
        New Detections object with merged boxes.
    """
    if len(detections) <= 1:
        return detections

    boxes = detections.xyxy.copy()
    confs = detections.confidence.copy()
    suppressed: set = set()
    merged_list = []

    for i in range(len(boxes)):
        if i in suppressed:
            continue
        merged = boxes[i].copy()
        best_conf = confs[i]

        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            xi1 = max(boxes[i][0], boxes[j][0])
            yi1 = max(boxes[i][1], boxes[j][1])
            xi2 = min(boxes[i][2], boxes[j][2])
            yi2 = min(boxes[i][3], boxes[j][3])
            inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            iou = inter / (area_i + area_j - inter + 1e-6)

            if iou > iou_thresh:
                merged[0] = min(merged[0], boxes[j][0])
                merged[1] = min(merged[1], boxes[j][1])
                merged[2] = max(merged[2], boxes[j][2])
                merged[3] = max(merged[3], boxes[j][3])
                best_conf = max(best_conf, confs[j])
                suppressed.add(j)

        merged_list.append((merged, best_conf))

    new_xyxy = np.array([m[0] for m in merged_list], dtype=np.float32)
    new_conf = np.array([m[1] for m in merged_list], dtype=np.float32)
    new_cls  = np.zeros(len(merged_list), dtype=int)
    return sv.Detections(xyxy=new_xyxy, confidence=new_conf, class_id=new_cls)


def run_detection_pipeline(model: YOLO, frame: np.ndarray,
                            roi_polygon: np.ndarray,
                            conf_thresh: float, iou_thresh: float,
                            min_box_height: int,
                            nms_merge_iou: float) -> sv.Detections:
    """
    Full detection pipeline for one frame:
      YOLO detect → ROI filter → min-height filter → NMS merge
    """
    detections = detect_persons(model, frame, conf_thresh, iou_thresh)
    detections = filter_roi(detections, roi_polygon)
    detections = filter_min_height(detections, min_box_height)
    if len(detections) > 1:
        detections = merge_overlapping_detections(detections, nms_merge_iou)
    return detections
