"""
detector.py
-----------
YOLOv8-based object detector. Runs inference on a frame and returns
bounding boxes.  On a mouse-click the caller asks us to find the box
that is closest to the click point.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Detector:
    """Wraps YOLOv8 for object detection."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Loading YOLO model from '%s' on device '%s'", model_path, device)

        if not Path(model_path).exists():
            logger.info("Model file not found locally — Ultralytics will auto-download it.")

        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info("Detector ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 on *frame* (BGR uint8).

        Returns
        -------
        list of dicts, each with keys:
            bbox  : (x, y, w, h) in pixel coords
            conf  : float confidence
            cls   : int class id
            label : str class name
        """
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                detections.append(
                    dict(bbox=(x, y, w, h), conf=conf, cls=cls, label=label)
                )

        logger.debug("Detected %d objects.", len(detections))
        return detections

    # ------------------------------------------------------------------

    def pick_closest(
        self, detections: list[dict], click_x: int, click_y: int
    ) -> dict | None:
        """
        Return the detection whose bounding-box centre is closest to the
        click point, or *None* if the list is empty.
        """
        if not detections:
            return None

        best = None
        best_dist = float("inf")
        for det in detections:
            x, y, w, h = det["bbox"]
            cx = x + w / 2
            cy = y + h / 2
            dist = (cx - click_x) ** 2 + (cy - click_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best = det

        logger.debug(
            "Closest detection: label=%s conf=%.2f dist=%.1f",
            best["label"] if best else "none",
            best["conf"] if best else 0.0,
            best_dist ** 0.5,
        )
        return best
