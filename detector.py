

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  


class Detector:

    def __init__(
        self,
        model_path:     str            = "yolov8n.pt",
        conf_threshold: float          = 0.45,
        iou_threshold:  float          = 0.45,
        device:         Optional[str]  = None,
        imgsz:          int            = 320,
        person_only:    bool           = True,
        bbox_shrink:    float          = 0.08,   
        
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device         = device
        self.imgsz          = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.person_only    = person_only
        self.bbox_shrink    = bbox_shrink
        
       
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


    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        with torch.no_grad():
            results = self.model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self._classes,   
                verbose=False,
            )

        H, W = frame.shape[:2]
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])

                
                if self.person_only and cls_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                label = self.model.names[cls_id]

                raw_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
             
                tight_bbox = shrink_bbox(raw_bbox, W, H, self.bbox_shrink)

                detections.append(dict(
                    raw_bbox  = raw_bbox,
                    bbox      = tight_bbox,   
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

        if not detections:
            return None

        best      = None
        best_dist = float("inf")

        for det in detections:
          
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




def shrink_bbox(
    bbox:   tuple[int, int, int, int],
    W:      int,
    H:      int,
    factor: float = 0.08,
) -> tuple[int, int, int, int]:

    x, y, w, h = bbox
    dx = int(w * factor)
    dy = int(h * factor)

    nx  = max(0,     x + dx)
    ny  = max(0,     y + dy)
    nw  = max(10,    w - 2 * dx)
    nh  = max(10,    h - 2 * dy)

    
    nw  = min(nw, W - nx)
    nh  = min(nh, H - ny)

    return nx, ny, nw, nh
