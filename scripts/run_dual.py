"""
scripts/run_dual.py
Process both side-view and top-view videos, then merge into a side-by-side output.

Usage:
    python scripts/run_dual.py
"""

import os
import sys
import cv2
import torch
import functools
import numpy as np
import importlib.util


def load_config(path: str = "config/default_config.py"):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def merge_side_by_side(path_side: str, path_top: str,
                       output_path: str, fps: float) -> None:
    """Merge two videos horizontally into a single side-by-side output."""
    print("Merging side-by-side ...")
    cap_s = cv2.VideoCapture(path_side)
    cap_t = cv2.VideoCapture(path_top)
    W_s = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_s = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_t = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_t = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))

    DUAL_H   = H_s
    W_t_new  = int(W_t * (DUAL_H / H_t))
    DUAL_W   = W_s + W_t_new

    out_dual = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (DUAL_W, DUAL_H)
    )

    while True:
        ret_s, fs = cap_s.read()
        ret_t, ft = cap_t.read()
        if not ret_s or not ret_t:
            break
        ft_r = cv2.resize(ft, (W_t_new, DUAL_H))
        cv2.putText(fs,   "CAM1: SIDE", (10, H_s   - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(ft_r, "CAM2: TOP",  (10, DUAL_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        combined = np.hstack([fs, ft_r])
        cv2.line(combined, (W_s, 0), (W_s, DUAL_H), (255, 255, 255), 3)
        out_dual.write(combined)

    cap_s.release()
    cap_t.release()
    out_dual.release()
    print(f"✅ Dual video saved → {output_path}  ({DUAL_W}x{DUAL_H})")


def main():
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/default_config.py"
    cfg = load_config(config_path)

    _orig = torch.load
    torch.load = functools.partial(_orig, weights_only=False)

    from src.tracker import load_botsort, load_strongsort, load_reid_model
    from src.reid_gallery import ReIDGallery
    from src.counter import process_video
    from ultralytics import YOLO

    print("Loading models ...")
    model      = YOLO(cfg.YOLO_MODEL)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    reid_model = load_reid_model()

    # ── Side-view ─────────────────────────────────────────────────────
    tracker_side = load_botsort(cfg.LOST_TRACK_BUFFER, cfg.MIN_TRACK_FRAMES)
    gallery_side = ReIDGallery(threshold=cfg.REID_THRESHOLD,
                               timeout_sec=cfg.GALLERY_TIMEOUT)
    result_side  = process_video(
        video_path=cfg.VIDEO_PATH_SIDE,
        output_path=cfg.OUTPUT_PATH_SIDE,
        model=model, tracker=tracker_side,
        reid_model=reid_model, gallery=gallery_side,
        cfg=cfg, is_top_view=False,
    )

    # ── Top-view ──────────────────────────────────────────────────────
    tracker_top = load_strongsort(cfg.LOST_TRACK_BUFFER, cfg.MIN_TRACK_FRAMES)
    gallery_top = ReIDGallery(threshold=cfg.REID_THRESHOLD,
                              timeout_sec=cfg.GALLERY_TIMEOUT)
    result_top  = process_video(
        video_path=cfg.VIDEO_PATH_TOP,
        output_path=cfg.OUTPUT_PATH_TOP,
        model=model, tracker=tracker_top,
        reid_model=reid_model, gallery=gallery_top,
        cfg=cfg, is_top_view=True,
    )

    # ── Merge ─────────────────────────────────────────────────────────
    cap_tmp = cv2.VideoCapture(cfg.VIDEO_PATH_SIDE)
    vid_fps = cap_tmp.get(cv2.CAP_PROP_FPS)
    cap_tmp.release()

    merge_side_by_side(cfg.OUTPUT_PATH_SIDE, cfg.OUTPUT_PATH_TOP,
                       cfg.OUTPUT_PATH_DUAL, vid_fps)

    # ── Re-encode all three to H.264 ──────────────────────────────────
    for src, dst in [
        (cfg.OUTPUT_PATH_SIDE, cfg.PLAYABLE_SIDE),
        (cfg.OUTPUT_PATH_TOP,  cfg.PLAYABLE_TOP),
        (cfg.OUTPUT_PATH_DUAL, cfg.PLAYABLE_DUAL),
    ]:
        os.system(f"ffmpeg -y -i {src} -vcodec libx264 -crf 23 {dst} -loglevel quiet")
    print("✅ All 3 videos encoded to H.264:")
    print(f"   Side-view → {cfg.PLAYABLE_SIDE}")
    print(f"   Top-view  → {cfg.PLAYABLE_TOP}")
    print(f"   Dual view → {cfg.PLAYABLE_DUAL}")


if __name__ == "__main__":
    main()
