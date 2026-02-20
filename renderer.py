"""
renderer.py
-----------
Compositing pipeline.

FIX 3 — Tracking rectangle hidden by default
  • `show_box=False` in __init__ (can be flipped to True for debugging).
  • The lock-box drawing block is completely skipped when show_box=False.
  • Hover preview box is still shown (it disappears once locked — 
    the user gets visual confirmation of the click, then nothing).
  • The HUD status line still says "● TRACKING" so the user knows it's
    active even without a visible rectangle.

FIX 4 — Video stays sharp until a person is selected
  • `tracking_active = (bbox is not None and mask is not None)`
  • When False  → return `frame.copy()` — original, unprocessed pixels.
  • When True   → compute blur THEN composite — bokeh only after selection.
  • This means GaussianBlur is never called on idle frames, which also
    improves idle FPS.
"""

import logging
import math
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Renderer:
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.52
    FONT_TH    = 1

    COLOR_LOCK  = (0,  255, 120)   # green  — locked (only used when show_box=True)
    COLOR_HOVER = (0,  200, 255)   # amber  — hover preview before lock

    def __init__(
        self,
        blur_ksize:  int   = 31,
        blur_sigma:  float = 12.0,
        low_light:   bool  = False,
        animate_hz:  float = 1.5,
        show_box:    bool  = False,   # FIX 3 — False = invisible tracking
    ):
        self.blur_ksize  = blur_ksize | 1   # must be odd
        self.blur_sigma  = blur_sigma
        self.low_light   = low_light
        self.animate_hz  = animate_hz
        self.show_box    = show_box           # FIX 3

        self._t0         = time.time()
        self._clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._fps_times: list[float] = []
        self._fps_window = 30

    # ── Main entry point ──────────────────────────────────────────────────

    def render(
        self,
        frame:       np.ndarray,
        mask:        Optional[np.ndarray],
        bbox:        Optional[tuple],
        label:       str   = "",
        conf:        float = 0.0,
        fps:         float = 0.0,
        hover_box:   Optional[tuple] = None,
        hover_label: str   = "",
        all_dets:    list  = (),
    ) -> np.ndarray:
        """
        Produce the final composited output frame.

        idle  (bbox=None) → sharp original frame, hover previews shown
        locked (bbox set) → background blurred, subject sharp, box optional
        """

        # Optional low-light boost (applied before any compositing)
        if self.low_light:
            frame = self._enhance_low_light(frame)

        tracking_active = (bbox is not None and mask is not None)

        # ── FIX 4 ── Sharp-by-default ─────────────────────────────────────
        if tracking_active:
            # Only now do we pay the cost of GaussianBlur
            blurred = cv2.GaussianBlur(
                frame,
                (self.blur_ksize, self.blur_ksize),
                self.blur_sigma,
            )
            out = self._composite(frame, blurred, mask)
        else:
            # No person selected — return original untouched frame
            out = frame.copy()

        # ── Hover box — shown while scanning, hidden once locked ──────────
        # (When tracking_active, bbox==hover_box, so the condition below
        #  keeps it hidden automatically after lock.)
        if hover_box is not None and hover_box != bbox:
            self._draw_hover_box(out, hover_box, hover_label)

        # ── FIX 3 — Lock box: only drawn when show_box=True ───────────────
        if tracking_active and self.show_box:
            out = self._draw_lock_box(out, bbox, label, conf)

        # ── HUD — always drawn (FPS + tracking state text) ────────────────
        out = self._draw_hud(out, fps, locked=tracking_active)
        return out

    # ── FPS helper ────────────────────────────────────────────────────────

    def tick_fps(self) -> float:
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) > self._fps_window:
            self._fps_times.pop(0)
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    # ── Private helpers ───────────────────────────────────────────────────

    def _composite(
        self,
        sharp:   np.ndarray,
        blurred: np.ndarray,
        mask:    np.ndarray,
    ) -> np.ndarray:
        """Per-pixel alpha blend: sharp subject over blurred background."""
        alpha = (mask.astype(np.float32) / 255.0)[..., np.newaxis]  # H×W×1
        return (alpha * sharp + (1.0 - alpha) * blurred).astype(np.uint8)

    def _draw_lock_box(self, frame, bbox, label, conf):
        """Animated corner-accent rectangle — only called when show_box=True."""
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x2 = min(x + w, W - 1);  y2 = min(y + h, H - 1)
        x  = max(x, 0);           y  = max(y, 0)

        t     = time.time() - self._t0
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * self.animate_hz * t)
        th    = max(1, int(1 + pulse * 2))
        color = tuple(min(255, int(c * (0.65 + 0.35 * pulse)))
                      for c in self.COLOR_LOCK)

        cv2.rectangle(frame, (x, y), (x2, y2), color, th)
        self._corner_accents(frame, x, y, x2, y2, w, h, color, th + 1)
        self._pill_label(
            frame, f"  {label}  {conf:.0%}  ", x, y, color,
            text_color=(15, 15, 15),
        )
        return frame

    def _draw_hover_box(self, frame, bbox, label):
        """Amber preview box shown while cursor hovers over a detected person."""
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x2 = min(x + w, W - 1);  y2 = min(y + h, H - 1)
        x  = max(x, 0);           y  = max(y, 0)
        cv2.rectangle(frame, (x, y), (x2, y2), self.COLOR_HOVER, 2)
        self._pill_label(
            frame, f"  {label}  ", x, y, self.COLOR_HOVER,
            text_color=(15, 15, 15),
        )

    def _draw_hud(self, frame, fps, locked):
        lines  = [
            f"FPS  {fps:.1f}",
            "● TRACKING" if locked else "○ click a person",
        ]
        colors = [
            (0, 255, 80),
            (0, 220, 255) if locked else (160, 160, 160),
        ]
        for i, (line, color) in enumerate(zip(lines, colors)):
            yy = 24 + i * 22
            cv2.putText(frame, line, (11, yy + 1),
                        self.FONT, self.FONT_SCALE, (0, 0, 0),
                        self.FONT_TH + 1, cv2.LINE_AA)
            cv2.putText(frame, line, (10, yy),
                        self.FONT, self.FONT_SCALE, color,
                        self.FONT_TH, cv2.LINE_AA)
        return frame

    def _pill_label(self, frame, text, bx, by, bg_color,
                    text_color=(255, 255, 255)):
        (tw, th), bl = cv2.getTextSize(
            text, self.FONT, self.FONT_SCALE, self.FONT_TH)
        pad = 4
        tx  = max(0, bx)
        ty  = max(th + pad * 2, by - 2)
        cv2.rectangle(frame,
                      (tx, ty - th - pad),
                      (tx + tw + pad, ty + bl),
                      bg_color, -1)
        cv2.putText(frame, text, (tx + pad // 2, ty),
                    self.FONT, self.FONT_SCALE, text_color,
                    self.FONT_TH, cv2.LINE_AA)

    def _corner_accents(self, frame, x, y, x2, y2, w, h, color, th):
        clen = max(8, min(20, w // 6, h // 6))
        for cx, cy, sx, sy in [
            (x,  y,   1,  1), (x2, y,  -1,  1),
            (x,  y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + sx * clen, cy), color, th)
            cv2.line(frame, (cx, cy), (cx, cy + sy * clen), color, th)

    def _enhance_low_light(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab2 = cv2.merge([self._clahe.apply(l), a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
