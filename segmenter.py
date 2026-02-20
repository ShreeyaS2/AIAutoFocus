"""
segmenter.py
------------
Generates a foreground mask for the tracked subject.

Strategy
--------
We use a two-level approach:
  1. MediaPipe Selfie Segmentation  — best for *people* (fast, accurate edges).
  2. Bounding-box GrabCut fallback  — for any object class (car, dog, cup…).

The caller gets a single binary mask (uint8, 0/255) the same size as the
input frame.  Mask edges are refined with a Gaussian blur so the composite
blending looks smooth.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import MediaPipe — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp

    _MP_AVAILABLE = True
    logger.info("MediaPipe available — will use Selfie Segmentation for persons.")
except ImportError:
    _MP_AVAILABLE = False
    logger.warning("MediaPipe not found — using GrabCut for all classes.")


class Segmenter:
    """
    Produce a refined binary foreground mask.

    Parameters
    ----------
    edge_blur_ksize : int   Gaussian kernel size for mask-edge softening.
    grabcut_iters   : int   GrabCut iterations (higher = slower / better).
    """

    PERSON_CLASSES = {"person"}          # YOLO class names that trigger MP

    def __init__(self, edge_blur_ksize: int = 21, grabcut_iters: int = 3):
        self.edge_blur_ksize = edge_blur_ksize | 1   # ensure odd
        self.grabcut_iters = grabcut_iters
        self._mp_seg = None

        if _MP_AVAILABLE:
            self._mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1   # 1 = landscape model (more accurate)
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_mask(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str = "",
    ) -> np.ndarray:
        """
        Return a soft binary mask (uint8, 0–255) the same H×W as *frame*.

        Parameters
        ----------
        frame : BGR uint8 image
        bbox  : (x, y, w, h) of the tracked subject
        label : YOLO class name (used to pick the best segmentation method)
        """
        h, w = frame.shape[:2]
        x, y, bw, bh = _clamp_bbox(bbox, w, h)

        if bw <= 0 or bh <= 0:
            return np.zeros((h, w), dtype=np.uint8)

        # ---- choose method ------------------------------------------------
        if _MP_AVAILABLE and label in self.PERSON_CLASSES:
            mask = self._mediapipe_mask(frame, x, y, bw, bh)
        else:
            mask = self._grabcut_mask(frame, x, y, bw, bh)

        # ---- constrain mask to a slightly expanded bounding box -----------
        expanded = _expand_bbox((x, y, bw, bh), w, h, factor=0.15)
        ex, ey, ew, eh = expanded
        region_mask = np.zeros((h, w), dtype=np.uint8)
        region_mask[ey : ey + eh, ex : ex + ew] = 255
        mask = cv2.bitwise_and(mask, region_mask)

        # ---- soft edges ---------------------------------------------------
        mask = self._soften_edges(mask)
        return mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mediapipe_mask(
        self, frame: np.ndarray, x: int, y: int, bw: int, bh: int
    ) -> np.ndarray:
        """Run MediaPipe Selfie Segmentation; threshold to binary mask."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_seg.process(rgb)
        if result.segmentation_mask is None:
            return self._fallback_bbox_mask(frame, x, y, bw, bh)
        mp_mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255
        return mp_mask

    def _grabcut_mask(
        self, frame: np.ndarray, x: int, y: int, bw: int, bh: int
    ) -> np.ndarray:
        """GrabCut on bounding-box ROI → binary mask."""
        h, w = frame.shape[:2]
        # GrabCut needs at least 2-pixel margins inside the image
        if x < 1 or y < 1 or x + bw > w - 1 or y + bh > h - 1:
            return self._fallback_bbox_mask(frame, x, y, bw, bh)
        if bw < 10 or bh < 10:
            return self._fallback_bbox_mask(frame, x, y, bw, bh)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        gc_mask = np.zeros((h, w), np.uint8)
        rect = (x, y, bw, bh)

        try:
            cv2.grabCut(
                frame, gc_mask, rect, bgd_model, fgd_model,
                self.grabcut_iters, cv2.GC_INIT_WITH_RECT
            )
        except cv2.error as exc:
            logger.debug("GrabCut failed: %s — using bbox fallback", exc)
            return self._fallback_bbox_mask(frame, x, y, bw, bh)

        fg_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return fg_mask

    @staticmethod
    def _fallback_bbox_mask(
        frame: np.ndarray, x: int, y: int, bw: int, bh: int
    ) -> np.ndarray:
        """Solid rectangle mask — last resort."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y : y + bh, x : x + bw] = 255
        return mask

    def _soften_edges(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological dilation then Gaussian blur to soften edges."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (self.edge_blur_ksize, self.edge_blur_ksize), 0)
        return mask


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clamp_bbox(
    bbox: tuple[int, int, int, int], W: int, H: int
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _expand_bbox(
    bbox: tuple[int, int, int, int], W: int, H: int, factor: float = 0.1
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx = int(w * factor)
    dy = int(h * factor)
    x2 = max(0, x - dx)
    y2 = max(0, y - dy)
    w2 = min(W - x2, w + 2 * dx)
    h2 = min(H - y2, h + 2 * dy)
    return x2, y2, w2, h2
