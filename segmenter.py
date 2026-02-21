

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MP_AVAILABLE = False
_mp_selfie_class = None

try:
    import mediapipe as mp

    if hasattr(mp.solutions, "selfie_segmentation"):
        _mp_selfie_class = mp.solutions.selfie_segmentation.SelfieSegmentation
        _MP_AVAILABLE = True
        logger.info("MediaPipe: using legacy SelfieSegmentation API")
    else:
        try:
            from mediapipe.tasks.python import vision as _mp_vision
            from mediapipe.tasks.python.core import base_options as _mp_base
            _MP_AVAILABLE = True
            logger.info("MediaPipe: using Tasks ImageSegmenter API (>= 0.10)")
        except Exception as _e:
            logger.warning(
                "MediaPipe installed but neither SelfieSegmentation nor Tasks API "
                "is available (%s). Falling back to ellipse mask.", _e
            )
except ImportError:
    logger.warning("MediaPipe not installed — using ellipse mask for all classes.")


class Segmenter:

    PERSON_CLASSES = {"person"}

    def __init__(
        self,
        edge_blur_ksize: int = 15,
        cpu_mode:        bool = True,
        mp_model:        int  = 0,
    ):
        self.edge_blur_ksize = edge_blur_ksize | 1
        self.cpu_mode        = cpu_mode
        self._mp_seg         = None
        self._use_legacy_mp  = False

        if _MP_AVAILABLE:
            if _mp_selfie_class is not None:
                self._mp_seg = _mp_selfie_class(model_selection=mp_model)
                self._use_legacy_mp = True
                logger.info("Segmenter ready — MediaPipe legacy (model=%d)", mp_model)
            else:
                self._use_legacy_mp = False
                logger.info("Segmenter ready — MediaPipe Tasks API")
        else:
            logger.info("Segmenter ready — ellipse masks only")


    def get_mask(
        self,
        frame: np.ndarray,
        bbox:  Tuple[int, int, int, int],
        label: str = "",
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y, bw, bh = _clamp_bbox(bbox, w, h)
        if bw <= 0 or bh <= 0:
            return np.zeros((h, w), dtype=np.uint8)

        if _MP_AVAILABLE and label in self.PERSON_CLASSES:
            mask = self._mediapipe_mask(frame, x, y, bw, bh)
        elif not self.cpu_mode:
            mask = self._grabcut_mask(frame, x, y, bw, bh)
        else:
            mask = self._ellipse_mask(h, w, x, y, bw, bh)


        ex, ey, ew, eh = _expand_bbox((x, y, bw, bh), w, h, factor=0.10)
        region = np.zeros((h, w), dtype=np.uint8)
        region[ey:ey+eh, ex:ex+ew] = 255
        mask = cv2.bitwise_and(mask, region)

        return self._feather(mask)

 
    def _ellipse_mask(self, H, W, x, y, bw, bh) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (x + bw // 2, y + bh // 2),
            (max(1, bw // 2), max(1, bh // 2)),
            0, 0, 360, 255, -1,
        )
        return mask

    def _mediapipe_mask(self, frame, x, y, bw, bh) -> np.ndarray:

        H, W = frame.shape[:2]
        pad  = max(bw, bh) // 4
        rx   = max(0, x - pad);  ry = max(0, y - pad)
        rx2  = min(W, x + bw + pad); ry2 = min(H, y + bh + pad)
        crop = frame[ry:ry2, rx:rx2]

        scale = min(1.0, 256 / max(crop.shape[0], crop.shape[1]))
        if scale < 1.0:
            small = cv2.resize(crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = crop

        try:
            if self._use_legacy_mp and self._mp_seg is not None:
                result = self._mp_seg.process(
                    cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                )
                if result.segmentation_mask is None:
                    return self._ellipse_mask(H, W, x, y, bw, bh)
                mp_small = (result.segmentation_mask > 0.45).astype(np.uint8) * 255
            else:

                return self._ellipse_mask(H, W, x, y, bw, bh)
        except Exception as e:
            logger.debug("MediaPipe segmentation failed: %s", e)
            return self._ellipse_mask(H, W, x, y, bw, bh)

        if scale < 1.0:
            mp_crop = cv2.resize(mp_small, (rx2 - rx, ry2 - ry),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            mp_crop = mp_small

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[ry:ry2, rx:rx2] = mp_crop
        return mask

    def _grabcut_mask(self, frame, x, y, bw, bh) -> np.ndarray:
        H, W = frame.shape[:2]
        if x < 1 or y < 1 or x+bw > W-1 or y+bh > H-1 or bw < 10 or bh < 10:
            return self._ellipse_mask(H, W, x, y, bw, bh)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        gc  = np.zeros((H, W),  np.uint8)
        try:
            cv2.grabCut(frame, gc, (x, y, bw, bh), bgd, fgd, 2,
                        cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return self._ellipse_mask(H, W, x, y, bw, bh)
        return np.where(
            (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
            np.uint8(255), np.uint8(0),
        )

    def _feather(self, mask: np.ndarray) -> np.ndarray:
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.dilate(mask, k, iterations=1)
        return cv2.GaussianBlur(
            out, (self.edge_blur_ksize, self.edge_blur_ksize), 0
        )



def _clamp_bbox(bbox, W, H):
    x, y, w, h = bbox
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    return x, y, w, h

def _expand_bbox(bbox, W, H, factor=0.1):
    x, y, w, h = bbox
    dx = int(w * factor); dy = int(h * factor)
    x2 = max(0, x-dx);   y2 = max(0, y-dy)
    return x2, y2, min(W-x2, w+2*dx), min(H-y2, h+2*dy)
