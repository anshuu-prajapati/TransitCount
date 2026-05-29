"""
scripts/visualise_config.py
Preview the counting baseline and ROI polygon on a sample frame.
Run this BEFORE the main processing loop to verify BASELINE_X and ROI_POLYGON.

Usage:
    python scripts/visualise_config.py
"""

import cv2
import importlib.util
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def load_config(path: str = "config/default_config.py"):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def visualise(video_path: str, baseline_x: int, roi_polygon,
              label: str, is_top_view: bool = False,
              frame_num: int = 30) -> None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"⚠️  Could not read frame from {video_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(16, 9))
    ax.imshow(frame_rgb)
    ax.axvline(x=baseline_x, color='yellow', linewidth=3, linestyle='--',
               label=f'Baseline x={baseline_x}')

    if is_top_view:
        ax.text(baseline_x - 130, 40, 'EXIT ←',  color='red',  fontsize=13, fontweight='bold')
        ax.text(baseline_x + 15,  40, '→ ENTRY', color='lime', fontsize=13, fontweight='bold')
    else:
        ax.text(baseline_x - 130, 40, '← ENTRY', color='lime', fontsize=13, fontweight='bold')
        ax.text(baseline_x + 15,  40, 'EXIT →',  color='red',  fontsize=13, fontweight='bold')

    poly = MplPolygon(roi_polygon, closed=True, fill=True,
                      facecolor='cyan', alpha=0.15,
                      edgecolor='cyan', linewidth=2)
    ax.add_patch(poly)
    ax.set_title(f'{label} — adjust BASELINE_X and ROI_POLYGON in config until correct')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def main():
    cfg = load_config()
    visualise(cfg.VIDEO_PATH_SIDE, cfg.BASELINE_X,     cfg.ROI_POLYGON,
              "Side-view (1.mp4)", is_top_view=False)
    visualise(cfg.VIDEO_PATH_TOP,  cfg.BASELINE_X_TOP, cfg.ROI_POLYGON_TOP,
              "Top-view  (2.mp4)", is_top_view=True)


if __name__ == "__main__":
    main()
