"""
src/tracker.py
Tracker initialisation helpers (BoTSORT for side-view, StrongSORT for top-view).

StrongSORT / BoTSORT both use appearance embeddings on every frame (unlike
ByteTrack which only uses IoU), making them significantly more stable through
occlusion and pose changes — critical for passengers navigating bus steps.
"""

import torch
from pathlib import Path


REID_WEIGHTS = "osnet_ain_x1_0_msmt17.pt"


def get_device() -> str:
    return '0' if torch.cuda.is_available() else 'cpu'


def load_botsort(lost_track_buffer: int, min_track_frames: int):
    """
    Load BoTSORT tracker — used as the primary side-view tracker.

    Args:
        lost_track_buffer:  Frames to hold a lost track before removing it.
                            Keep at 150 to survive stair-navigation disappearances.
        min_track_frames:   Frames before a new track is considered reliable.
    """
    from boxmot import BoTSORT
    tracker = BoTSORT(
        model_weights=Path(REID_WEIGHTS),
        device=get_device(),
        fp16=False,
        track_high_thresh=0.35,
        track_low_thresh=0.15,
        new_track_thresh=0.35,
        track_buffer=lost_track_buffer,
        match_thresh=0.9,
        proximity_thresh=0.7,
        appearance_thresh=0.20,
    )
    print(f"✅ BoTSORT loaded")
    print(f"   max_age (lost buffer) : {lost_track_buffer} frames")
    print(f"   n_init (min frames)   : {min_track_frames} frames")
    return tracker


def load_strongsort(lost_track_buffer: int, min_track_frames: int):
    """
    Load StrongSORT tracker — used for the top-view camera.

    StrongSORT uses appearance embeddings on every frame (not just on track
    loss like BoT-SORT), making it more stable through occlusion.

    Args:
        lost_track_buffer:  Frames to hold a lost track.
        min_track_frames:   Frames before a new track is trusted.
    """
    from boxmot import StrongSort
    tracker = StrongSort(
        model_weights=Path(REID_WEIGHTS),
        device=get_device(),
        fp16=False,
        max_age=lost_track_buffer,
        n_init=min_track_frames,
    )
    print(f"✅ StrongSORT loaded")
    return tracker


def load_reid_model():
    """Load OSNet Re-ID backend for embedding extraction."""
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    device = get_device()
    reid_model = ReidAutoBackend(
        weights=Path(REID_WEIGHTS),
        device=device,
        half=False,
    )
    print(f"✅ OSNet Re-ID loaded  (device={device})")
    return reid_model
