

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def expand_bbox(
    bbox: Tuple[int, int, int, int],
    factor: float = 0.20,        
    frame_w: int = 9999,
    frame_h: int = 9999,
) -> Tuple[int, int, int, int]:

    x, y, w, h = bbox
    dx = int(w * factor)
    dy = int(h * factor)

    nx = max(0, x - dx)
    ny = max(0, y - dy)
    nw = min(w + 2 * dx, frame_w - nx)
    nh = min(h + 2 * dy, frame_h - ny)

    return nx, ny, nw, nh


def shrink_bbox(
    bbox: Tuple[int, int, int, int],
    factor: float = 0.0,
    frame_w: int = 9999,
    frame_h: int = 9999,
) -> Tuple[int, int, int, int]:

    x, y, w, h = bbox
    dx = int(w * factor)
    dy = int(h * factor)

    nx = x + dx
    ny = y + dy
    nw = max(10, w - 2 * dx)
    nh = max(10, h - 2 * dy)

    nx = max(0, min(nx, frame_w - 1))
    ny = max(0, min(ny, frame_h - 1))
    nw = min(nw, frame_w - nx)
    nh = min(nh, frame_h - ny)

    return nx, ny, nw, nh


class SubjectTracker:


    MIN_IOU   = 0.10
    MAX_LOST  = 5

    def __init__(self, shrink_factor: float = 0.08):
        self.shrink_factor  = shrink_factor
        self._tracker       = None
        self._last_bbox     = None
        self._lost_count    = 0
        self._label         = ""
        self._conf          = 0.0
        self._init_time     = 0.0


    def init(
        self,
        frame: np.ndarray,
        bbox:  Tuple[int, int, int, int],
        label: str   = "",
        conf:  float = 0.0,
    ) -> None:

        H, W = frame.shape[:2]

        tight_bbox = shrink_bbox(bbox, factor=self.shrink_factor,
                                 frame_w=W, frame_h=H)

        self._tracker   = cv2.TrackerCSRT_create()
        self._tracker.init(frame, tight_bbox)
        self._last_bbox = tight_bbox
        self._lost_count = 0
        self._label      = label
        self._conf       = conf
        self._init_time  = time.time()
        logger.info(
            "Tracker init: label=%s  raw=%s  tight=%s  shrink=%.0f%%",
            label, bbox, tight_bbox, self.shrink_factor * 100,
        )

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple], float]:

        if self._tracker is None or self._last_bbox is None:
            return False, None, 0.0

        ok, new_bbox = self._tracker.update(frame)
        new_bbox = tuple(int(v) for v in new_bbox) if ok else None

        if not ok:
            self._lost_count += 1
            logger.debug("Tracker failed (lost=%d)", self._lost_count)
            if self._lost_count >= self.MAX_LOST:
                logger.warning("Tracker lost — re-detection required.")
                return False, None, 0.0
            return True, self._last_bbox, max(0.0, 0.4 - 0.1 * self._lost_count)

        iou = _iou(self._last_bbox, new_bbox)
        if iou < self.MIN_IOU:
            self._lost_count += 1
            logger.debug("Low IoU %.3f — possible drift", iou)
        else:
            self._lost_count = max(0, self._lost_count - 1)

        self._last_bbox = new_bbox
        elapsed     = time.time() - self._init_time
        conf_proxy  = min(1.0, 0.6 + 0.4 * min(elapsed, 1.0))
        return True, new_bbox, conf_proxy

    @property
    def is_active(self) -> bool:
        return self._tracker is not None and self._last_bbox is not None

    @property
    def last_bbox(self) -> Optional[Tuple]:
        return self._last_bbox

    @property
    def label(self) -> str:
        return self._label

    @property
    def conf(self) -> float:
        return self._conf

    def reset(self) -> None:
        self._tracker    = None
        self._last_bbox  = None
        self._lost_count = 0
        logger.debug("Tracker reset.")



def _iou(b1: tuple, b2: tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(x1, x2);  iy = max(y1, y2)
    iw = min(x1+w1, x2+w2) - ix
    ih = min(y1+h1, y2+h2) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0
