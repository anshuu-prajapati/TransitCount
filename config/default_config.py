"""
config/default_config.py
All tunable parameters for the Bus Passenger Counter.
Edit this file to match your camera setup and video paths.
"""

import numpy as np

# ── Video paths ───────────────────────────────────────────────────────
VIDEO_PATH_SIDE = "/content/1.mp4"   # side-view door camera
VIDEO_PATH_TOP  = "/content/2.mp4"   # top-view camera
VIDEO_PATH      = VIDEO_PATH_SIDE    # primary video for single-cam runs

# ── Output paths ──────────────────────────────────────────────────────
OUTPUT_PATH_SIDE = "/content/output_side_counted.mp4"
OUTPUT_PATH_TOP  = "/content/output_top_counted.mp4"
OUTPUT_PATH_DUAL = "/content/output_dual_sidebyside.mp4"
OUTPUT_PATH      = OUTPUT_PATH_SIDE

# Playable H.264 re-encoded outputs
PLAYABLE_SIDE = "/content/output_side_h264.mp4"
PLAYABLE_TOP  = "/content/output_top_h264.mp4"
PLAYABLE_DUAL = "/content/output_dual_h264.mp4"

# ── YOLO model ────────────────────────────────────────────────────────
YOLO_MODEL  = "yolov8l.pt"   # yolov8n=fastest / yolov8l=best accuracy
CONF_THRESH = 0.40
IOU_THRESH  = 0.45

# ── Baseline (vertical counting line) ────────────────────────────────
# Run scripts/visualise_config.py first to find the correct pixel X.
# LEFT  of baseline = inside bus  → crossing right-to-left = ENTRY
# RIGHT of baseline = outside/steps → crossing left-to-right = EXIT
BASELINE_X     = 620    # ← adjust after visualise_config.py preview
BASELINE_X_TOP = 380

# ── ROI polygon (counting zone) ──────────────────────────────────────
# Only persons whose centre point falls inside this polygon are tracked.
# Ignores pedestrians outside the bus door area.
ROI_POLYGON = np.array([
    [0,   200],
    [900, 200],
    [900, 720],
    [0,   720],
], dtype=np.int32)

ROI_POLYGON_TOP = np.array(
    [[0, 0], [750, 0], [750, 720], [0, 720]], dtype=np.int32
)

# ── Re-ID settings ────────────────────────────────────────────────────
REID_THRESHOLD  = 0.45   # cosine similarity. Lower = more forgiving match
GALLERY_TIMEOUT = 600    # seconds to remember an exited person in the gallery

# ── Stable-ID fixes ───────────────────────────────────────────────────
MIN_BOX_HEIGHT    = 100   # ignore detections shorter than this (px) — removes ghost tracks
NMS_MERGE_IOU     = 0.30  # merge overlapping boxes before tracker (kills pillar-split)
LOST_TRACK_BUFFER = 150   # frames tracker holds a lost ID (150 @ 15fps ≈ 10 sec)
MIN_TRACK_FRAMES  = 8     # frames before a new track is trusted
COOLDOWN_FRAMES   = 90    # min frames between same person crossing twice
BASELINE_BUFFER   = 80    # dead-band px around baseline (stops straddling jitter)

# ── Processing ────────────────────────────────────────────────────────
PROCESS_EVERY_N = 2      # process every 2nd frame → stable ~15fps for Kalman filter
MAX_FRAMES      = None   # None = process full video; set e.g. 900 for quick tests

# ── Display ───────────────────────────────────────────────────────────
SHOW_ROI_OVERLAY = True
SHOW_TRACK_TRAIL = True
TRAIL_LENGTH     = 40
