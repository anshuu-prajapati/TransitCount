"""
src/annotator.py
Drawing helpers: HUD overlay, ROI polygon, baseline line, track annotations.
"""

import cv2
import numpy as np
import supervision as sv


def build_annotators(trail_length: int, show_trail: bool):
    """Create and return supervision annotator instances."""
    box_annotator   = sv.BoundingBoxAnnotator(
        thickness=2, color=sv.ColorPalette.DEFAULT
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=3,
        color=sv.ColorPalette.DEFAULT
    )
    trace_annotator = (
        sv.TraceAnnotator(
            thickness=2, trace_length=trail_length,
            color=sv.ColorPalette.DEFAULT
        ) if show_trail else None
    )
    return box_annotator, label_annotator, trace_annotator


def draw_hud(frame: np.ndarray, entry_count: int, exit_count: int,
             total_count: int, fps_display: float) -> np.ndarray:
    """
    Draw Entry / Exit / Total HUD in the top-left corner.
    TOTAL = ENTRY + EXIT (unique persons, no double-count).
    """
    cv2.rectangle(frame, (10, 10), (340, 135), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (340, 135), (255, 255, 255), 1)
    cv2.putText(frame, f"ENTRY  : {entry_count}",
                (20, 48),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),   2)
    cv2.putText(frame, f"EXIT   : {exit_count}",
                (20, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
    cv2.putText(frame, f"TOTAL  : {total_count}",
                (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps_display:.1f}",
                (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    return frame


def draw_roi(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Draw a semi-transparent ROI polygon overlay."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (0, 255, 200))
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    cv2.polylines(frame, [polygon], True, (0, 255, 200), 2)
    return frame


def draw_baseline(frame: np.ndarray, baseline_x: int,
                  is_top_view: bool = False) -> np.ndarray:
    """Draw the counting baseline and ENTRY/EXIT labels."""
    h = frame.shape[0]
    cv2.line(frame, (baseline_x, 0), (baseline_x, h), (0, 255, 255), 3)
    if is_top_view:
        cv2.putText(frame, "EXIT <-",  (baseline_x - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        cv2.putText(frame, "-> ENTRY", (baseline_x + 10,  25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
    else:
        cv2.putText(frame, "ENTRY <-", (baseline_x - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),   2)
        cv2.putText(frame, "-> EXIT",  (baseline_x + 10,  25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    return frame


def annotate_detections(frame: np.ndarray, detections: sv.Detections,
                        labels: list,
                        box_annotator, label_annotator,
                        trace_annotator) -> np.ndarray:
    """Apply box, label, and trace annotations to a frame."""
    if len(detections) == 0:
        return frame
    if trace_annotator:
        frame = trace_annotator.annotate(frame, detections)
    frame = box_annotator.annotate(frame, detections)
    if labels:
        frame = label_annotator.annotate(
            frame, detections, labels=labels[:len(detections)]
        )
    return frame
