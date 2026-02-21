

import logging
import math
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Renderer:
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.52
    FONT_TH    = 1

    COLOR_LOCK  = (0,  255, 120)
    COLOR_HOVER = (0,  200, 255)

    def __init__(
        self,
        blur_ksize:  int   = 31,
        blur_sigma:  float = 12.0,
        low_light:   bool  = False,
        animate_hz:  float = 1.5,
        show_box:    bool  = False,
    ):
        self.blur_ksize  = blur_ksize | 1
        self.blur_sigma  = blur_sigma
        self.low_light   = low_light
        self.animate_hz  = animate_hz
        self.show_box    = show_box

        self._t0         = time.time()
        self._clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._fps_times  = []
        self._fps_window = 30



    def render(
        self,
        frame,
        mask        = None,
        bbox        = None,
        label:  str   = "",
        conf:   float = 0.0,
        fps:    float = 0.0,
        hover_box    = None,
        hover_label: str = "",
        all_dets     = (),
        **kwargs,
    ) -> np.ndarray:

        if self.low_light:
            frame = self._enhance_low_light(frame)

      
        if mask is None or bbox is None:
            out = frame.copy()

            if hover_box is not None:
                self._draw_hover_box(out, hover_box, hover_label)
            out = self._draw_hud(out, fps, locked=False)
            return out


        blurred = cv2.GaussianBlur(
            frame,
            (self.blur_ksize, self.blur_ksize),
            self.blur_sigma,
        )
        alpha = mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        out   = (alpha * frame + (1.0 - alpha) * blurred).astype(np.uint8)

        if self.show_box:
            out = self._draw_lock_box(out, bbox, label, conf)

        out = self._draw_hud(out, fps, locked=True)
        return out


    def tick_fps(self) -> float:
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) > self._fps_window:
            self._fps_times.pop(0)
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0


    def _draw_lock_box(self, frame, bbox, label, conf):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x2 = min(x + w, W - 1);  y2 = min(y + h, H - 1)
        x  = max(x, 0);           y  = max(y, 0)
        t     = time.time() - self._t0
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * self.animate_hz * t)
        th    = max(1, int(1 + pulse * 2))
        color = tuple(min(255, int(c * (0.65 + 0.35 * pulse))) for c in self.COLOR_LOCK)
        cv2.rectangle(frame, (x, y), (x2, y2), color, th)
        self._corner_accents(frame, x, y, x2, y2, w, h, color, th + 1)
        self._pill_label(frame, f"  {label}  {conf:.0%}  ", x, y, color, text_color=(15, 15, 15))
        return frame

    def _draw_hover_box(self, frame, bbox, label):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x2 = min(x + w, W - 1)
        y2 = min(y + h, H - 1)
        x  = max(x, 0)
        y  = max(y, 0)

        
    def _draw_hud(self, frame, fps, locked):
        lines  = [f"FPS  {fps:.1f}", "● TRACKING" if locked else "○ click a person"]
        colors = [(0, 255, 80), (0, 220, 255) if locked else (160, 160, 160)]

        return frame

    def _pill_label(self, frame, text, bx, by, bg_color, text_color=(255, 255, 255)):
        (tw, th), bl = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.FONT_TH)
        pad = 4
        tx  = max(0, bx)
        ty  = max(th + pad * 2, by - 2)
        cv2.rectangle(frame, (tx, ty - th - pad), (tx + tw + pad, ty + bl), bg_color, -1)
        cv2.putText(frame, text, (tx + pad // 2, ty), self.FONT, self.FONT_SCALE,
                    text_color, self.FONT_TH, cv2.LINE_AA)

    def _corner_accents(self, frame, x, y, x2, y2, w, h, color, th):
        clen = max(8, min(20, w // 6, h // 6))
        for cx, cy, sx, sy in [(x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (cx, cy), (cx + sx * clen, cy), color, th)
            cv2.line(frame, (cx, cy), (cx, cy + sy * clen), color, th)

    def _enhance_low_light(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab2 = cv2.merge([self._clahe.apply(l), a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)