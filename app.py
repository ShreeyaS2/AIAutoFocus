"""
app.py  —  FOCUSR backend  (CPU-optimised build)
=================================================
CPU FPS improvements applied:
  1. asyncio.run_in_executor  — CV/AI work runs in a thread pool,
                                never blocks uvicorn's event loop
  2. Frame-drop guard         — if the pipeline is still busy with the
                                previous frame, incoming frames are
                                silently dropped (browser keeps sending,
                                server catches up)
  3. YOLO at 320px input      — half the pixels YOLO sees → ~2× faster
  4. Segmentation every 6 frames (not 3)
  5. GrabCut disabled         — replaced by feathered ellipse (<1 ms)
  6. Bokeh blur capped at 51  — bigger kernels give diminishing returns
                                but cost linearly
  7. Models loaded once at startup (not per-connection)
  8. CORS + health endpoint retained
"""

import asyncio
import base64
import concurrent.futures
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import base64
# ... rest of your imports ...

# ── ADD THIS RIGHT HERE — before the Session class ───────────────────────
def _best_iou_match(
    dets: list,
    last_bbox: tuple,
    min_iou: float = 0.15,
) -> dict | None:
    """
    Return the detection with the highest IoU overlap with last_bbox.
    Returns None if no detection clears min_iou — prevents switching
    to a completely different person.
    """
    lx, ly, lw, lh = last_bbox
    best = None
    best_iou = min_iou

    for det in dets:
        x, y, w, h = det["raw_bbox"]

        ix = max(lx, x)
        iy = max(ly, y)
        iw = min(lx + lw, x + w) - ix
        ih = min(ly + lh, y + h) - iy

        if iw <= 0 or ih <= 0:
            continue

        inter = iw * ih
        union = lw * lh + w * h - inter
        iou   = inter / union if union > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best     = det

    return best



# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Import pipeline modules ──────────────────────────────────────────────
try:
    from detector import Detector
    from tracker import SubjectTracker
    from segmenter import Segmenter
    from renderer import Renderer
except ImportError as e:
    logger.error("Import error: %s", e)
    logger.error("Run:  python install.py   to fix missing packages")
    sys.exit(1)

# Thread pool for running blocking CV/AI code without stalling uvicorn
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# ── Shared model (loaded once at startup) ────────────────────────────────
_shared_detector: "Detector | None" = None
_startup_error:   str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _shared_detector, _startup_error
    try:
        logger.info("Loading YOLOv8n …")
        loop = asyncio.get_event_loop()
        _shared_detector = await loop.run_in_executor(
            _executor,
            lambda: Detector(model_path="yolov8n.pt", person_only=True)
        )
        logger.info("✓ YOLOv8n ready (imgsz=320, CPU mode)")
    except Exception as exc:
        _startup_error = str(exc)
        logger.error("Model load failed: %s", exc, exc_info=True)
    yield
    # Shutdown: nothing to clean up for YOLO/torch


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(title="FOCUSR — Smart AutoFocus (CPU build)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Per-connection session ───────────────────────────────────────────────
class Session:
    # CPU-tuned defaults — defined directly here, no external dict needed
    DETECT_EVERY  = 20    # run YOLO every N frames
    MASK_EVERY    = 6     # regenerate mask every N frames
    MAX_FRAME_W   = 480   # resize input before all processing
    BLUR_KSIZE    = 31    # bokeh kernel size

    def __init__(self):
        if _shared_detector is None:
            raise RuntimeError(_startup_error or "Models still loading - retry in a moment")
        self.detector  = _shared_detector
        # shrink_factor=0.0 because detector.detect() already produces
        # a shrunk bbox as det["bbox"].  Shrinking again in tracker.init()
        # would apply the reduction twice.  (Bug 2 fix)
        self.tracker   = SubjectTracker(shrink_factor=0.0)
        self.segmenter = Segmenter(edge_blur_ksize=15, cpu_mode=True, mp_model=0)
        self.renderer  = Renderer(
            blur_ksize=self.BLUR_KSIZE,
            blur_sigma=12.0,
            show_box=False,
        )
        self.tracking   = False
        self.bbox       = None
        self.mask       = None
        self.label      = ""
        self.conf       = 0.0
        self.frame_id   = 0
        self.last_dets: list[dict] = []
        self.detect_every = self.DETECT_EVERY
        self.mask_every   = self.MASK_EVERY

    # Called from a thread pool — may use blocking CV/numpy freely
    def process_blocking(self, msg: dict) -> tuple[np.ndarray | None, dict]:
        # ── Decode ──────────────────────────────────────────────────────
        frame_b64 = msg.get("frame", "")
        if not frame_b64:
            return None, {}
        try:
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(frame_b64), np.uint8),
                cv2.IMREAD_COLOR,
            )
        except Exception:
            return None, {}
        if frame is None:
            return None, {}

        # ── Resize down before ALL processing ───────────────────────────
        h, w = frame.shape[:2]
        scale_to_client = 1.0   # ratio of original w to resized w
        if w > self.MAX_FRAME_W:
            scale_to_client = self.MAX_FRAME_W / w
            frame = cv2.resize(frame, (self.MAX_FRAME_W, int(h * scale_to_client)),
                               interpolation=cv2.INTER_LINEAR)

        # ── Live client params ───────────────────────────────────────────
        if "detect_every" in msg:
            self.detect_every = max(5, int(msg["detect_every"]))
        if "bokeh_k" in msg:
            # Cap at 51 on CPU — bigger buys almost nothing visually
            self.renderer.blur_ksize = min(51, int(msg["bokeh_k"]) | 1)
        self.renderer.low_light = bool(msg.get("low_light", False))
        # FIX 3 — show/hide tracking rectangle live from the browser toggle
        if "show_box" in msg:
            self.renderer.show_box = bool(msg["show_box"])

        # ── Release ──────────────────────────────────────────────────────
        if msg.get("release"):
            self.tracker.reset()
            self.tracking = False
            self.bbox       = None
            self.mask       = None
            self.render_bbox = None   # ← clears the blur region
            self.label      = ""
            self.conf       = 0.0
        cursor_x = msg.get("cursor_x")
        cursor_y = msg.get("cursor_y")
        clicked  = bool(msg.get("clicked", False))
        self.frame_id += 1

        # Scale cursor coords to match resized frame
        if cursor_x is not None and cursor_y is not None and scale_to_client < 1.0:
            cursor_x = int(cursor_x * scale_to_client)
            cursor_y = int(cursor_y * scale_to_client)

        # ── Periodic background detection ────────────────────────────────
        if self.frame_id % self.detect_every == 0 or (clicked and not self.last_dets):
            self.last_dets = self.detector.detect(frame)

        # ── Click → lock ─────────────────────────────────────────────────
        if clicked and cursor_x is not None:
            if not self.last_dets:
                self.last_dets = self.detector.detect(frame)
            chosen = self.detector.pick_closest(self.last_dets, cursor_x, cursor_y)
            if chosen:
                self.tracker.reset()
                self.tracker.init(frame, chosen["bbox"], chosen["label"], chosen["conf"])
                self.tracking = True
                self.label    = chosen["label"]
                self.conf     = chosen["conf"]
                self.mask     = None
                logger.info("Locked '%s' (%.2f)", chosen["label"], chosen["conf"])
            else:
                self.tracking = False
                self.bbox = self.mask = None

        # ── Hover preview ────────────────────────────────────────────────
        hover_box = None; hover_label = ""
        if not clicked and cursor_x is not None and self.last_dets:
            hov = self.detector.pick_closest(self.last_dets, cursor_x, cursor_y)
            if hov:
                hx, hy, hw, hh = hov["bbox"]
                if (abs(hx + hw / 2 - cursor_x) < hw * 0.85
                        and abs(hy + hh / 2 - cursor_y) < hh * 0.85):
                    hover_box   = hov["bbox"]
                    hover_label = hov["label"]

        # ── Update tracker ───────────────────────────────────────────────
        if self.tracking and self.tracker.is_active:
            ok, new_bbox, track_conf = self.tracker.update(frame)
            if ok and new_bbox is not None:
                self.bbox = new_bbox
                self.conf = float(track_conf)

                H, W = frame.shape[:2]
                x, y, w, h = new_bbox
                pad = 0.10
                dx, dy = int(w * pad), int(h * pad)

                x1 = max(0, x - dx)
                y1 = max(0, y - dy)
                x2 = min(W, x + w + dx)
                y2 = min(H, y + h + dy)

                self.render_bbox = (x1, y1, x2 - x1, y2 - y1)
            if self.frame_id % self.mask_every == 0 or self.mask is None:
                # Get actual person shape from MediaPipe
                raw_mask = self.segmenter.get_mask(frame, new_bbox, self.tracker.label)

                # Expand edges outward
                expand_px = 40
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1)
                )
                dilated = cv2.dilate(raw_mask, kernel, iterations=1)

                # ── Gradient fade ───────────────────────────────────────────
                # Distance transform: each pixel gets its distance from the edge
                dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)

                # Normalize so the furthest point from edge = 1.0
                if dist.max() > 0:
                    dist = dist / dist.max()

                # Apply a power curve — higher = sharper center, softer fade
                # 0.4 = very soft gradient   1.0 = linear   2.0 = sharp center
                gradient_mask = np.power(dist, 0.5).astype(np.float32)

                # Convert to uint8 (0-255) for renderer
                self.mask = (gradient_mask * 255).astype(np.uint8)
            else:
                logger.info("Tracker lost — re-detecting…")
                self.render_bbox = None
                dets = self.detector.detect(frame)
                self.last_dets = dets
                if dets and self.bbox:
                    # ── IoU match — only reacquire if bbox overlaps enough ──
                    chosen = _best_iou_match(dets, self.bbox, min_iou=0.15)
                    if chosen:
                        self.tracker.reset()
                        self.tracker.init(frame, chosen["bbox"], chosen["label"], chosen["conf"])
                        self.label = chosen["label"]
                        self.conf  = chosen["conf"]
                    else:
                        # Nobody overlaps enough — hold last position, don't switch
                        logger.info("No IoU match found — holding position.")
                else:
                    self.tracking = False
                    self.bbox = self.mask = None

        # ── Render ───────────────────────────────────────────────────────
        fps    = self.renderer.tick_fps()
        output = self.renderer.render(
            frame=frame,
            mask=self.mask        if self.tracking else None,
            bbox=self.render_bbox if self.tracking and self.render_bbox else self.bbox if self.tracking else None,
            label=self.label, conf=self.conf, fps=fps,
            hover_box=hover_box, hover_label=hover_label,
            all_dets=self.last_dets if not self.tracking else [],
        )

        return output, {
            "tracking":  self.tracking,
            "label":     self.label if self.tracking else "",
            "conf":      round(self.conf, 2) if self.tracking else 0.0,
            "fps":       round(fps, 1),
            "frame_id":  self.frame_id,
            "det_count": len(self.last_dets),
        }


# ── HTTP routes ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    p = STATIC_DIR / "index.html"
    if not p.exists():
        return HTMLResponse("<h2>static/index.html not found</h2>", status_code=404)
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return JSONResponse({
        "status":        "ok" if _shared_detector else "loading",
        "models_ready":  _shared_detector is not None,
        "startup_error": _startup_error,
        "opencv":        cv2.__version__,
        "mode":          "CPU",
    })


# ── WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected from %s", ws.client)

    if _shared_detector is None:
        err = _startup_error or "Models still loading — refresh in a few seconds"
        await ws.send_text(json.dumps({"error": err}))
        await ws.close(code=1011)
        return

    try:
        session = Session()
    except RuntimeError as exc:
        await ws.send_text(json.dumps({"error": str(exc)}))
        await ws.close(code=1011)
        return

    loop     = asyncio.get_event_loop()
    busy     = False          # frame-drop guard: True while pipeline is running

    try:
        while True:
            raw = await ws.receive_text()

            # ── Frame-drop: if we're still processing the last frame,
            #    parse only lightweight fields (release / low_light) and skip CV
            if busy:
                try:
                    quick = json.loads(raw)
                    if quick.get("release"):
                        session.tracker.reset()
                        session.tracking = False
                        session.bbox = session.mask = None
                except Exception:
                    pass
                continue   # drop this frame — server will catch up

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            busy = True
            try:
                # Run all blocking CV/AI work in the thread pool
                output, meta = await loop.run_in_executor(
                    _executor,
                    session.process_blocking,
                    msg,
                )
            except Exception as exc:
                logger.error("Pipeline error: %s", exc, exc_info=True)
                await ws.send_text(json.dumps({"error": str(exc)}))
                busy = False
                continue
            finally:
                busy = False

            if output is None:
                continue

            try:
                ok, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    await ws.send_text(json.dumps({
                        "frame": base64.b64encode(buf.tobytes()).decode(),
                        **meta,
                    }))
            except Exception as exc:
                logger.error("Send error: %s", exc)
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as exc:
        logger.error("WS error: %s", exc, exc_info=True)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  FOCUSR — CPU-Optimised Build")
    print("  Browser:      http://localhost:8000")
    print("  Health check: http://localhost:8000/health")
    print("═" * 58 + "\n")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
