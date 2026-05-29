"""
scripts/run_side.py
Process the side-view (door camera) video only.

Usage:
    python scripts/run_side.py
    python scripts/run_side.py --config config/default_config.py
"""

import sys
import torch
import functools
import importlib.util

# ── Config loading ─────────────────────────────────────────────────────
def load_config(path: str = "config/default_config.py"):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def main():
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/default_config.py"
    cfg = load_config(config_path)

    # PyTorch 2.6+ compatibility patch
    _orig = torch.load
    torch.load = functools.partial(_orig, weights_only=False)

    from src.tracker import load_botsort, load_reid_model
    from src.reid_gallery import ReIDGallery
    from src.counter import process_video
    from ultralytics import YOLO

    device = '0' if torch.cuda.is_available() else 'cpu'

    print("Loading models ...")
    model      = YOLO(cfg.YOLO_MODEL)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    tracker    = load_botsort(cfg.LOST_TRACK_BUFFER, cfg.MIN_TRACK_FRAMES)
    reid_model = load_reid_model()
    gallery    = ReIDGallery(threshold=cfg.REID_THRESHOLD,
                             timeout_sec=cfg.GALLERY_TIMEOUT)

    result = process_video(
        video_path=cfg.VIDEO_PATH_SIDE,
        output_path=cfg.OUTPUT_PATH_SIDE,
        model=model,
        tracker=tracker,
        reid_model=reid_model,
        gallery=gallery,
        cfg=cfg,
        is_top_view=False,
    )

    # Re-encode to H.264 for playback compatibility
    import os
    os.system(f"ffmpeg -y -i {cfg.OUTPUT_PATH_SIDE} "
              f"-vcodec libx264 -crf 23 {cfg.PLAYABLE_SIDE} -loglevel quiet")
    print(f"✅ H.264 output → {cfg.PLAYABLE_SIDE}")

    return result


if __name__ == "__main__":
    main()
