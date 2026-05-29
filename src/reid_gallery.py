"""
src/reid_gallery.py
Re-ID gallery that prevents double-counting returning passengers.

Key ideas:
- Every person who crosses the baseline gets a unique `person_id`.
- Their OSNet appearance embedding is stored in the gallery when they exit.
- If the same person re-enters, their embedding matches a gallery entry
  → they receive the same `person_id` → the re-entry is blocked from counting.
- Embeddings are updated via EMA (exponential moving average) for stability.
"""

import time
import numpy as np
from scipy.spatial.distance import cosine


class ReIDGallery:
    """
    Tracks unique persons across entry/exit events using appearance embeddings.

    Attributes:
        threshold:    Cosine similarity required for a gallery match.
        timeout_sec:  Seconds before a gallery entry expires.
        gallery:      person_id → {embedding, exit_time}
        active:       track_id  → person_id
        embeddings:   track_id  → current EMA embedding
        next_pid:     Counter for assigning new person_ids.
    """

    def __init__(self, threshold: float = 0.58, timeout_sec: int = 600):
        self.threshold   = threshold
        self.timeout_sec = timeout_sec
        self.gallery: dict   = {}
        self.active: dict    = {}
        self.embeddings: dict = {}
        self.next_pid = 1

    # ── Embedding extraction ──────────────────────────────────────────

    def get_embedding(self, reid_model, frame: np.ndarray,
                      bbox_xyxy: np.ndarray):
        """
        Extract an L2-normalised OSNet embedding for the given bounding box.

        Args:
            reid_model:  boxmot ReidAutoBackend instance.
            frame:       Full BGR frame.
            bbox_xyxy:   [x1, y1, x2, y2] bounding box.

        Returns:
            Normalised embedding array, or None on failure.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 40:
            return None
        try:
            boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            emb   = reid_model.model.get_features(boxes, frame)[0]
            norm  = np.linalg.norm(emb)
            return emb / (norm + 1e-6)
        except Exception as e:
            print(f"  [EMB ERROR] {type(e).__name__}: {e}")
            return None

    # ── EMA update ────────────────────────────────────────────────────

    def update_embedding_ema(self, track_id: int, new_emb: np.ndarray,
                              alpha: float = 0.85) -> None:
        """
        Update the stored embedding for `track_id` using EMA.

        alpha=0.85 means 85% old + 15% new each update, making the
        embedding progressively more representative over time.
        """
        if track_id not in self.embeddings or self.embeddings[track_id] is None:
            self.embeddings[track_id] = new_emb
        else:
            ema  = alpha * self.embeddings[track_id] + (1 - alpha) * new_emb
            norm = np.linalg.norm(ema)
            self.embeddings[track_id] = ema / (norm + 1e-6)

    # ── Identity assignment ───────────────────────────────────────────

    def match_or_new(self, track_id: int) -> tuple[int, bool]:
        """
        Assign a person_id to a track.

        Resolution order:
          1. Track already active → return existing person_id.
          2. Embedding matches a gallery entry → restore that person_id
             (returning passenger, not double-counted).
          3. No match → assign a brand-new person_id.

        Returns:
            (person_id, is_returning)
        """
        if track_id in self.active:
            return self.active[track_id], False

        # Expire old gallery entries
        now     = time.time()
        expired = [pid for pid, d in self.gallery.items()
                   if now - d['exit_time'] > self.timeout_sec]
        for pid in expired:
            del self.gallery[pid]

        emb = self.embeddings.get(track_id)
        best_pid, best_sim = None, -1.0

        if emb is not None:
            for pid, data in self.gallery.items():
                if data['embedding'] is None:
                    continue
                sim = 1 - cosine(emb, data['embedding'])
                if sim > best_sim:
                    best_sim, best_pid = sim, pid

        if best_pid is not None and best_sim >= self.threshold:
            self.active[track_id] = best_pid
            del self.gallery[best_pid]
            print(f"  [REID MATCH] track={track_id} matched "
                  f"person_id={best_pid}  sim={best_sim:.3f}")
            return best_pid, True
        else:
            pid = self.next_pid
            self.next_pid += 1
            self.active[track_id] = pid
            if best_pid is not None:
                print(f"  [REID FAIL]  track={track_id} "
                      f"best_sim={best_sim:.3f} < threshold={self.threshold}  "
                      f"new pid={pid}")
            else:
                print(f"  [REID NEW]   track={track_id} "
                      f"gallery empty → new pid={pid}")
            return pid, False

    # ── Exit handling ─────────────────────────────────────────────────

    def mark_exited(self, track_id: int) -> None:
        """Move a track from active → gallery when it crosses the EXIT line."""
        if track_id not in self.active:
            return
        pid = self.active.pop(track_id)
        emb = self.embeddings.get(track_id)
        if emb is None:
            print(f"  [WARN] person_id={pid} exited with no embedding "
                  f"— ReID won't work for re-entry")
        self.gallery[pid] = {'embedding': emb, 'exit_time': time.time()}
        self.embeddings.pop(track_id, None)
        print(f"  [GALLERY] person_id={pid} stored  "
              f"gallery_size={len(self.gallery)}")

    def remove_active(self, track_id: int) -> None:
        """Remove a track that was lost without crossing the baseline."""
        self.active.pop(track_id, None)
        self.embeddings.pop(track_id, None)
