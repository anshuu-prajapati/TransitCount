"""
src/counter.py
Core counting loop: processes one video with a given tracker + ReID gallery,
writes an annotated output video, and returns final entry/exit counts.

Counting logic (TOTAL = ENTRY + EXIT, no double-count):
  - LEFT  of baseline = inside bus  → right→left crossing = ENTRY
  - RIGHT of baseline = outside/steps → left→right crossing = EXIT
  - Each person_id can appear in entry_person_ids OR exit_person_ids (or both)
    but is only counted ONCE per direction.
  - A re-entering person is matched by ReID → same person_id → blocked.
"""

import cv2
import time
import collections
import numpy as np
import supervision as sv
from collections import defaultdict
from typing import Optional

from src.detector import run_detection_pipeline
from src.reid_gallery import ReIDGallery
from src.annotator import (build_annotators, draw_hud, draw_roi,
                            draw_baseline, annotate_detections)


def process_video(
    video_path: str,
    output_path: str,
    model,
    tracker,
    reid_model,
    gallery: ReIDGallery,
    cfg,                  # config module or object with all parameters
    is_top_view: bool = False,
) -> dict:
    """
    Main processing loop for a single video.

    Args:
        video_path:   Path to input video.
        output_path:  Path for annotated output (.mp4).
        model:        Loaded YOLO model.
        tracker:      Loaded BoTSORT or StrongSORT tracker.
        reid_model:   Loaded OSNet Re-ID backend.
        gallery:      ReIDGallery instance (shared across runs for dual-cam).
        cfg:          Config object/module (see config/default_config.py).
        is_top_view:  If True, reverses ENTRY/EXIT direction labels.

    Returns:
        dict with keys: entry_count, exit_count, total_count, elapsed_sec
    """
    baseline_x  = cfg.BASELINE_X_TOP if is_top_view else cfg.BASELINE_X
    roi_polygon = cfg.ROI_POLYGON_TOP if is_top_view else cfg.ROI_POLYGON

    # ── Video I/O ─────────────────────────────────────────────────────
    cap   = cv2.VideoCapture(video_path)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cfg.MAX_FRAMES:
        total = min(total, cfg.MAX_FRAMES)

    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # ── Annotators ────────────────────────────────────────────────────
    box_ann, lbl_ann, trc_ann = build_annotators(
        cfg.TRAIL_LENGTH, cfg.SHOW_TRACK_TRAIL
    )

    # ── Per-track state ───────────────────────────────────────────────
    track_frames     = defaultdict(int)
    track_person_ids: dict = {}
    track_last_side: dict  = {}
    track_labels: dict     = {}

    # ── Per-person state ──────────────────────────────────────────────
    entry_count     = 0
    exit_count      = 0
    entry_person_ids: set = set()
    exit_person_ids: set  = set()
    crossing_cooldown: dict = {}

    frame_idx = 0
    t_start   = time.time()
    fps_hist  = collections.deque(maxlen=30)

    print(f"▶  Processing {total} frames ({total/fps:.1f}s) → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret or (cfg.MAX_FRAMES and frame_idx >= cfg.MAX_FRAMES):
            break

        frame_idx += 1
        t0 = time.time()

        if cfg.PROCESS_EVERY_N > 1 and frame_idx % cfg.PROCESS_EVERY_N != 0:
            continue  # skip frame — do NOT write raw frame to output

        # ── Detection pipeline ────────────────────────────────────────
        detections = run_detection_pipeline(
            model, frame, roi_polygon,
            cfg.CONF_THRESH, cfg.IOU_THRESH,
            cfg.MIN_BOX_HEIGHT, cfg.NMS_MERGE_IOU,
        )

        # ── Tracker update ────────────────────────────────────────────
        if len(detections) > 0:
            dets = np.column_stack([
                detections.xyxy,
                detections.confidence,
                detections.class_id.astype(float),
            ])
        else:
            dets = np.empty((0, 6))

        tracks = tracker.update(dets, frame)

        if frame_idx % 100 == 0:
            n_tracks = 0 if tracks is None else len(tracks)
            print(f"  frame={frame_idx}  dets={len(dets)}  tracks={n_tracks}")

        # ── Draw when no tracks ───────────────────────────────────────
        if tracks is None or len(tracks) == 0:
            if cfg.SHOW_ROI_OVERLAY:
                frame = draw_roi(frame, roi_polygon)
            frame = draw_baseline(frame, baseline_x, is_top_view)
            fps_hist.append(1.0 / max(time.time() - t0, 1e-6))
            frame = draw_hud(frame, entry_count, exit_count,
                             entry_count + exit_count, float(np.mean(fps_hist)))
            out_vid.write(frame)
            continue

        # ── Rebuild sv.Detections from tracker output ─────────────────
        tracks    = np.atleast_2d(tracks)
        xyxy      = tracks[:, 0:4].astype(float)
        track_ids = tracks[:, 4].astype(int)
        confs     = tracks[:, 5].astype(float) if tracks.shape[1] > 5 else np.ones(len(tracks))
        cls_ids   = tracks[:, 6].astype(int)   if tracks.shape[1] > 6 else np.zeros(len(tracks), dtype=int)
        detections = sv.Detections(
            xyxy=xyxy, confidence=confs,
            class_id=cls_ids, tracker_id=track_ids,
        )

        # ── Re-ID embedding extraction + EMA update ───────────────────
        for i, track_id in enumerate(track_ids):
            track_frames[track_id] += 1
            emb = gallery.get_embedding(reid_model, frame, detections.xyxy[i])
            if emb is not None:
                gallery.update_embedding_ema(track_id, emb)
            elif track_frames[track_id] <= 5:
                print(f"  [EMB FAIL] track={track_id} "
                      f"frame={frame_idx} bbox={detections.xyxy[i].astype(int)}")

        # ── Person ID assignment + crossing detection ──────────────────
        labels = []
        for i, track_id in enumerate(track_ids):
            bbox   = detections.xyxy[i]
            cx_det = (bbox[0] + bbox[2]) / 2

            # Side determination with dead-band buffer
            if cx_det < baseline_x - cfg.BASELINE_BUFFER:
                side = 'left'
            elif cx_det > baseline_x + cfg.BASELINE_BUFFER:
                side = 'right'
            else:
                side = track_last_side.get(track_id, 'right')

            # Assign person_id once the track is old enough to trust
            if track_id not in track_person_ids:
                if track_frames[track_id] >= cfg.MIN_TRACK_FRAMES:
                    pid, returning = gallery.match_or_new(track_id)
                    track_person_ids[track_id] = pid
                    if returning:
                        track_labels[track_id] = f"ID:{pid}(return)"
                    else:
                        track_labels[track_id] = f"ID:{pid}"
                else:
                    track_labels[track_id] = "ID:?"
                    track_last_side[track_id] = side
                    labels.append(track_labels[track_id])
                    continue

            pid       = track_person_ids[track_id]
            prev_side = track_last_side.get(track_id)
            track_last_side[track_id] = side

            # ── Crossing event detection ──────────────────────────────
            if prev_side is not None and prev_side != side:
                last_cross  = crossing_cooldown.get(pid, -9999)
                on_cooldown = (frame_idx - last_cross) < cfg.COOLDOWN_FRAMES

                if not on_cooldown:
                    if prev_side == 'right' and side == 'left':
                        # ENTRY: outside → inside
                        if pid not in entry_person_ids:
                            entry_count += 1
                            entry_person_ids.add(pid)
                            crossing_cooldown[pid] = frame_idx
                            track_labels[track_id] = f"ID:{pid} ENTRY"
                            print(f"  [ENTRY]  pid={pid} track={track_id} "
                                  f"frame={frame_idx} "
                                  f"ENTRY={entry_count} EXIT={exit_count} "
                                  f"TOTAL={entry_count+exit_count}")
                        else:
                            crossing_cooldown[pid] = frame_idx
                            track_labels[track_id] = f"ID:{pid}(re-entry)"
                            print(f"  [RE-ENTRY blocked]  pid={pid} frame={frame_idx}")

                    elif prev_side == 'left' and side == 'right':
                        # EXIT: inside → outside
                        crossing_cooldown[pid] = frame_idx
                        gallery.mark_exited(track_id)
                        track_person_ids.pop(track_id, None)
                        if pid not in exit_person_ids:
                            exit_count += 1
                            exit_person_ids.add(pid)
                            track_labels[track_id] = f"ID:{pid} EXIT"
                            print(f"  [EXIT]   pid={pid} track={track_id} "
                                  f"frame={frame_idx} "
                                  f"ENTRY={entry_count} EXIT={exit_count} "
                                  f"TOTAL={entry_count+exit_count}")
                        else:
                            track_labels[track_id] = f"ID:{pid}(re-exit blocked)"
                else:
                    track_labels[track_id] = f"ID:{pid}(cooldown)"

            labels.append(track_labels.get(track_id, f"ID:{track_id}"))

        # ── Draw ──────────────────────────────────────────────────────
        if cfg.SHOW_ROI_OVERLAY:
            frame = draw_roi(frame, roi_polygon)
        frame = draw_baseline(frame, baseline_x, is_top_view)
        frame = annotate_detections(frame, detections, labels,
                                    box_ann, lbl_ann, trc_ann)
        fps_hist.append(1.0 / max(time.time() - t0, 1e-6))
        frame = draw_hud(frame, entry_count, exit_count,
                         entry_count + exit_count, float(np.mean(fps_hist)))
        out_vid.write(frame)

        if frame_idx % 150 == 0:
            elapsed = time.time() - t_start
            pct     = 100 * frame_idx / total
            eta     = elapsed / frame_idx * (total - frame_idx)
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total}  "
                  f"Entry={entry_count} Exit={exit_count} "
                  f"Total={entry_count+exit_count}  ETA {eta:.0f}s")

    cap.release()
    out_vid.release()

    elapsed_total = time.time() - t_start
    result = {
        'entry_count': entry_count,
        'exit_count':  exit_count,
        'total_count': entry_count + exit_count,
        'elapsed_sec': elapsed_total,
    }

    print()
    print("=" * 55)
    print(f"  Done in {elapsed_total:.1f}s")
    print(f"   ENTRY  : {entry_count}")
    print(f"   EXIT   : {exit_count}")
    print(f"   TOTAL  : {entry_count + exit_count}  (= ENTRY + EXIT, no double-count)")
    print(f"   Output : {output_path}")
    print("=" * 55)

    return result
