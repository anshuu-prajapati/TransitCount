# 🚌 Bus Passenger Counter — Stable ID Edition

A robust, GPU-accelerated system for counting bus passengers entering and exiting using dual-camera video feeds (side-view door cam + top-view cam).

## ✨ Key Features

- **StrongSORT / BoTSORT tracker** — superior occlusion recovery over BoT-SORT/ByteTrack
- **OSNet Re-ID** — appearance-based identity matching prevents double-counting returning passengers
- **NMS pre-merge** — eliminates split detections caused by door pillars
- **Minimum box height filter** — removes ghost/partial-body tracks
- **EMA embedding updates** — running-average embeddings improve ReID robustness over time
- **Dual-camera support** — side-view + top-view, merged into side-by-side output video
- **Correct TOTAL logic** — `TOTAL = ENTRY + EXIT`, same person never double-counted via ReID gallery

---

## 📁 Project Structure

```
bus-passenger-counter/
├── README.md
├── requirements.txt
├── config/
│   └── default_config.py       # All tunable parameters in one place
├── src/
│   ├── __init__.py
│   ├── detector.py             # YOLOv8 detection + NMS merge + ROI filter
│   ├── tracker.py              # StrongSORT/BoTSORT tracker setup
│   ├── reid_gallery.py         # ReID gallery with EMA embedding updates
│   ├── annotator.py            # Drawing helpers (HUD, ROI overlay, trails)
│   └── counter.py              # Main counting loop logic
├── scripts/
│   ├── run_side.py             # Process side-view video only
│   ├── run_top.py              # Process top-view video only
│   ├── run_dual.py             # Process both and merge side-by-side
│   └── visualise_config.py     # Preview baseline & ROI on a frame
├── notebooks/
│   └── bus_passenger_counter.ipynb   # Original Colab notebook
└── docs/
    └── tuning_guide.md         # Parameter tuning reference
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Requires a CUDA-capable GPU.** On Google Colab, go to Runtime → Change runtime type → T4 GPU.

### 2. Configure

Edit `config/default_config.py` to set your video paths and tune the baseline/ROI for your specific camera setup.

```python
VIDEO_PATH_SIDE = "/path/to/1.mp4"
VIDEO_PATH_TOP  = "/path/to/2.mp4"
BASELINE_X      = 620   # adjust after running visualise_config.py
```

### 3. Preview Baseline & ROI

```bash
python scripts/visualise_config.py
```

This shows a frame with the counting line and ROI polygon overlaid. Adjust `BASELINE_X` and `ROI_POLYGON` until they match your door frame.

### 4. Run

```bash
# Side-view only
python scripts/run_side.py

# Both cameras, merged side-by-side output
python scripts/run_dual.py
```

---

## ⚙️ Parameter Reference

| Parameter | Default | Effect |
|---|---|---|
| `BASELINE_X` | `620` | X-pixel of the counting line. Run `visualise_config.py` to find correct value. |
| `ROI_POLYGON` | see config | Only count people inside this zone — ignores pedestrians outside. |
| `REID_THRESHOLD` | `0.45` | Cosine similarity for ReID match. Lower = more forgiving. |
| `MIN_BOX_HEIGHT` | `100` | Ignore detections shorter than this (px) — removes ghost tracks. |
| `NMS_MERGE_IOU` | `0.30` | Merge overlapping boxes before tracker — fixes pillar-split problem. |
| `LOST_TRACK_BUFFER` | `150` | Frames tracker holds a lost ID. Keep at 150 to survive stair navigation. |
| `PROCESS_EVERY_N` | `2` | Process every Nth frame → stable ~15fps for Kalman filter. |
| `COOLDOWN_FRAMES` | `90` | Min frames between the same person crossing twice (prevents jitter counts). |
| `BASELINE_BUFFER` | `80` | Dead-band px around baseline — stops straddling jitter. |

See `docs/tuning_guide.md` for detailed tuning advice.

---

## 🧠 Counting Logic

```
LEFT  of baseline = inside bus  → crossing right-to-left = ENTRY
RIGHT of baseline = outside/steps → crossing left-to-right = EXIT

TOTAL = ENTRY + EXIT
```

Each unique person gets a `person_id` from the ReID gallery. If the same person re-enters, their `person_id` is matched from the gallery → the re-entry is **blocked from being counted again**.

---

## 📦 Requirements

See `requirements.txt`. Key packages:

- `ultralytics==8.3.0` (YOLOv8)
- `supervision==0.21.0`
- `boxmot==10.0.84` (StrongSORT/BoTSORT + OSNet)
- `torch>=2.3.0` + `torchvision`
- `opencv-python-headless`
- `scipy`, `numpy==1.26.4`

---

## 📝 License

MIT
