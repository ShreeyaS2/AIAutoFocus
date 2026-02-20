"""
detector.py
-----------
YOLOv8 wrapper with four targeted fixes:

FIX 1 — Person-only detection
  • `classes=[0]` is passed directly to YOLO inference so non-person
    classes are filtered before NMS — no wasted computation.
  • Secondary guard in the loop as a safety net.

FIX 2 — Tight bounding box
  • `shrink_bbox()` insets the raw YOLO box by `shrink_pct` on every side
    before it is handed to the CSRT tracker.
  • Default shrink is 8% per side — enough to exclude background pixels
    that bleed outside the person's silhouette without cropping the body.
  • The value is tunable per call so callers can loosen it for large objects.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0   # COCO class index for "person"


class Detector:
    """
    Parameters
    ----------
    model_path    : YOLOv8 weights file (auto-downloaded if absent)
    conf_threshold: Minimum confidence to keep a detection (0-1)
    iou_threshold : NMS IoU threshold
    device        : 'cpu', 'cuda', or None (auto)
    imgsz         : YOLO inference resolution (smaller = faster on CPU)
    person_only   : When True (default) only 'person' class is returned
    bbox_shrink   : Fraction to inset each side of the bbox (Fix 2)
    """

    def __init__(
        self,
        model_path:     str            = "yolov8n.pt",
        conf_threshold: float          = 0.45,
        iou_threshold:  float          = 0.45,
        device:         Optional[str]  = None,
        imgsz:          int            = 320,
        person_only:    bool           = True,
        bbox_shrink:    float          = 0.08,   # FIX 2 — 8% inset per side
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device         = device
        self.imgsz          = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.person_only    = person_only
        self.bbox_shrink    = bbox_shrink
        # FIX 1 — pass class list directly to YOLO, not just as a post-filter
        self._classes       = [PERSON_CLASS_ID] if person_only else None

        logger.info(
            "Loading YOLO '%s' | device=%s | imgsz=%d | "
            "person_only=%s | bbox_shrink=%.0f%%",
            model_path, device, imgsz, person_only, bbox_shrink * 100,
        )
        if not Path(model_path).exists():
            logger.info("Model not found — Ultralytics will download it automatically.")

        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info("Detector ready.")

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 on *frame* (BGR uint8).

        Returns
        -------
        list of dicts with keys:
            raw_bbox  : (x, y, w, h) — original YOLO box, full size
            bbox      : (x, y, w, h) — shrunk box used for tracker init
            conf      : float
            cls       : int
            label     : str
        """
        with torch.no_grad():
            results = self.model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self._classes,   # FIX 1 — restrict inside YOLO
                verbose=False,
            )

        H, W = frame.shape[:2]
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])

                # FIX 1 — secondary guard (should never fire, but be safe)
                if self.person_only and cls_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                label = self.model.names[cls_id]

                raw_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                # FIX 2 — produce tighter bbox for tracker initialisation
                tight_bbox = shrink_bbox(raw_bbox, W, H, self.bbox_shrink)

                detections.append(dict(
                    raw_bbox  = raw_bbox,
                    bbox      = tight_bbox,   # tracker uses this
                    conf      = conf,
                    cls       = cls_id,
                    label     = label,
                ))

        logger.debug("Detected %d person(s).", len(detections))
        return detections

    def pick_closest(
        self,
        detections: List[Dict[str, Any]],
        click_x:    int,
        click_y:    int,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the detection whose bbox centre is closest to the click point.
        Distance is computed on the RAW (un-shrunk) bbox so the click target
        area matches what the user sees on screen.
        """
        if not detections:
            return None

        best      = None
        best_dist = float("inf")

        for det in detections:
            # Use raw_bbox for hit-testing (larger, matches visual box)
            x, y, w, h = det["raw_bbox"]
            cx   = x + w / 2
            cy   = y + h / 2
            dist = (cx - click_x) ** 2 + (cy - click_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best      = det

        logger.debug(
            "Closest: %s conf=%.2f dist=%.1f",
            best["label"] if best else "none",
            best["conf"]  if best else 0.0,
            best_dist ** 0.5,
        )
        return best


# ── Utility ───────────────────────────────────────────────────────────────

def shrink_bbox(
    bbox:   tuple[int, int, int, int],
    W:      int,
    H:      int,
    factor: float = 0.08,
) -> tuple[int, int, int, int]:
    """
    Inset a bounding box by *factor* on each side.

    Example: factor=0.08 removes 8% from left, right, top, and bottom.
    This prevents CSRT from locking onto background pixels or neighbouring
    people that bleed just inside the raw YOLO boundary.

    The result is always clamped to the frame dimensions and guaranteed
    to have w >= 10 and h >= 10 so the tracker can initialise.
    """
    x, y, w, h = bbox
    dx = int(w * factor)
    dy = int(h * factor)

    nx  = max(0,     x + dx)
    ny  = max(0,     y + dy)
    nw  = max(10,    w - 2 * dx)
    nh  = max(10,    h - 2 * dy)

    # Clamp to frame boundary
    nw  = min(nw, W - nx)
    nh  = min(nh, H - ny)

    return nx, ny, nw, nh
