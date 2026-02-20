"""
tracker.py
----------
Wraps OpenCV's CSRT tracker.  If the tracker loses the target (low IoU /
failure) we signal the caller to trigger a fresh YOLOv8 detection round.
"""

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SubjectTracker:
    """
    Single-subject tracker built on cv2.TrackerCSRT.

    Lifecycle
    ---------
    1. Call ``init(frame, bbox)`` to lock onto a bounding box.
    2. Call ``update(frame)`` every frame — returns (ok, bbox, conf_proxy).
    3. If ``update`` returns ok=False, caller should re-detect.
    """

    # Minimum intersection-over-union with previous box to accept update
    MIN_IOU = 0.10
    # Shrink factor so the tracker's inner crop is slightly inside the bbox
    PADDING = 0.05

    def __init__(self):
        self._tracker: cv2.Tracker | None = None
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._lost_count: int = 0
        self._max_lost: int = 5          # frames before we declare failure
        self._label: str = ""
        self._conf: float = 0.0
        self._init_time: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def init(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str = "",
        conf: float = 0.0,
    ) -> None:
        """Initialise (or re-initialise) tracker on *bbox* = (x, y, w, h)."""
        self._tracker = cv2.TrackerCSRT_create()
        self._tracker.init(frame, bbox)
        self._last_bbox = bbox
        self._lost_count = 0
        self._label = label
        self._conf = conf
        self._init_time = time.time()
        logger.info("Tracker initialised: label=%s bbox=%s", label, bbox)

    def update(self, frame: np.ndarray) -> tuple[bool, tuple | None, float]:
        """
        Update tracker with the new *frame*.

        Returns
        -------
        ok     : bool   — False means the target is lost
        bbox   : (x, y, w, h) or None
        conf   : float  — proxy confidence (1.0 while tracking well)
        """
        if self._tracker is None or self._last_bbox is None:
            return False, None, 0.0

        ok, new_bbox = self._tracker.update(frame)
        new_bbox = tuple(int(v) for v in new_bbox) if ok else None

        if not ok:
            self._lost_count += 1
            logger.debug("Tracker update failed (lost_count=%d)", self._lost_count)
            if self._lost_count >= self._max_lost:
                logger.warning("Tracker lost target — re-detection required.")
                return False, None, 0.0
            # Return last known bbox while we give it a few more chances
            return True, self._last_bbox, max(0.0, 0.4 - 0.1 * self._lost_count)

        # Sanity-check via IoU with previous box
        iou = self._compute_iou(self._last_bbox, new_bbox)
        if iou < self.MIN_IOU and self._lost_count == 0:
            # Sudden large jump — might be drift
            logger.debug("Low IoU (%.3f) — possible drift, incrementing lost count", iou)
            self._lost_count += 1

        self._lost_count = max(0, self._lost_count - 1)
        self._last_bbox = new_bbox

        # Confidence proxy based on how long we've been stable
        elapsed = time.time() - self._init_time
        conf_proxy = min(1.0, 0.6 + 0.4 * min(elapsed, 1.0))
        return True, new_bbox, conf_proxy

    @property
    def is_active(self) -> bool:
        return self._tracker is not None and self._last_bbox is not None

    @property
    def last_bbox(self) -> tuple | None:
        return self._last_bbox

    @property
    def label(self) -> str:
        return self._label

    @property
    def conf(self) -> float:
        return self._conf

    def reset(self) -> None:
        """Discard current tracking state."""
        self._tracker = None
        self._last_bbox = None
        self._lost_count = 0
        logger.debug("Tracker reset.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(
        b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]
    ) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        # Intersection
        ix = max(x1, x2)
        iy = max(y1, y2)
        iw = min(x1 + w1, x2 + w2) - ix
        ih = min(y1 + h1, y2 + h2) - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0
