# Parameter Tuning Guide

## 1. Find your BASELINE_X

Run `python scripts/visualise_config.py`. A yellow dashed vertical line will appear on a sample frame.

- The line should sit exactly on the **edge of the door frame** — the boundary between "inside bus" (left) and "outside/steps" (right).
- Adjust `BASELINE_X` in `config/default_config.py` and re-run until it's correct.

---

## 2. Set your ROI_POLYGON

The ROI polygon restricts counting to a specific zone, preventing pedestrians outside the door from being tracked.

- In the visualise script output, the cyan polygon shows the ROI.
- Points are `[x, y]` pixel coordinates. The polygon must fully enclose the door area.
- Common mistake: making the ROI too wide and including street pedestrians.

---

## 3. Tune REID_THRESHOLD

Controls how similar two embeddings must be to be considered the same person.

| Value | Effect |
|---|---|
| `0.35` – `0.45` | Loose — same person matched more easily, but risk of false matches |
| `0.50` – `0.60` | Moderate — good default range |
| `0.65` – `0.75` | Strict — reduces false matches, but may not recognise same person on re-entry |

Start at `0.45`. If you see `[REID FAIL]` logs for the same person re-entering, lower it. If you see false `[REID MATCH]` logs for different people, raise it.

---

## 4. MIN_BOX_HEIGHT

Set this to the minimum plausible pixel height of a person in your video.

- Too low → ghost tracks from reflections, luggage, partial bodies.
- Too high → real persons entering from far away get filtered out.

A good starting point is `height_of_frame * 0.12` (e.g., for 720p: `720 * 0.12 = 86 → round to 100`).

---

## 5. NMS_MERGE_IOU

Only relevant if YOLO splits one person into two boxes (pillar-split problem).

- `0.25` – `0.35` is typically sufficient.
- If you see two bounding boxes on the same person, lower this value.
- If separate persons standing close together get merged, raise it.

---

## 6. LOST_TRACK_BUFFER

How many frames the tracker holds a lost identity before deleting it.

- At `PROCESS_EVERY_N=2` and ~30fps, effective rate is ~15fps.
- `150 frames / 15fps = 10 seconds` — enough to survive a passenger navigating bus steps.
- Reduce this if you see ghost tracks persisting long after a person has left.

---

## 7. COOLDOWN_FRAMES

Minimum frames between the same person counting twice (in the same direction).

- Prevents jitter-crossing: a person straddling the baseline from oscillating the counter.
- At 15fps effective, `90 frames = 6 seconds`. Suitable for most scenarios.

---

## 8. BASELINE_BUFFER (dead-band)

A dead-band zone of ±N pixels around the baseline where side assignment is frozen.

- While a person's centre is within `[BASELINE_X - BUFFER, BASELINE_X + BUFFER]`, they keep their last known side.
- Prevents rapid left/right/left oscillation from triggering spurious crossings.
- Default `80px` is suitable for most door widths. Increase for wider doors or shaky footage.

---

## 9. PROCESS_EVERY_N

Set to `2` to process every other frame (~15fps effective on 30fps video).

- This gives the Kalman filter a **consistent time step**, making predictions more accurate.
- Do not set to `1` on 30fps video unless you have a high-end GPU — the Kalman filter performance degrades with variable frame timings.
- Never set above `3` — too few frames per second for reliable track continuity.
