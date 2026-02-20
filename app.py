"""
app.py  —  FastAPI WebSocket backend (fixed)
=============================================
Fixes applied:
  1. Models loaded ONCE at startup (not per connection) — prevents WS timeout
  2. CORS middleware added
  3. /health endpoint for diagnostics
  4. Per-frame errors are caught and reported back to client (not silent)
  5. WebSocket ping/pong keepalive
  6. Graceful startup failure with clear console message

Run:
    python app.py
Or:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import base64
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import AI modules with helpful error messages
# ---------------------------------------------------------------------------
try:
    from detector import Detector
    from tracker import SubjectTracker
    from segmenter import Segmenter
    from renderer import Renderer
except ImportError as e:
    logger.error("Failed to import pipeline module: %s", e)
    logger.error("Make sure all .py files are in the same folder as app.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="FOCUSR — Smart AutoFocus API")

# ── CORS — allow the browser to connect from any origin ──────────────────
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

# ---------------------------------------------------------------------------
# SHARED MODELS — loaded once at startup, reused by every session
# ---------------------------------------------------------------------------
_shared_detector  = None
_startup_error    = None


@app.on_event("startup")
async def load_models():
    """Load heavy models once. If this fails the server still starts but
    WebSocket sessions will return an error message to the client."""
    global _shared_detector, _startup_error
    try:
        logger.info("Loading YOLOv8 model (first run may download ~6 MB)…")
        _shared_detector = Detector(model_path="yolov8n.pt")
        logger.info("✓ Models ready.")
    except Exception as exc:
        _startup_error = str(exc)
        logger.error("Model load failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Per-connection session
# ---------------------------------------------------------------------------
class Session:
    """
    Pipeline state for one WebSocket client.
    Uses the shared detector (already loaded) and creates its own
    tracker / segmenter / renderer instances (these are lightweight).
    """

    def __init__(self):
        if _shared_detector is None:
            raise RuntimeError(
                "Models not ready yet — "
                + (_startup_error or "still loading, retry in a moment")
            )
        self.detector  = _shared_detector          # shared — read-only inference
        self.tracker   = SubjectTracker()          # per-client state
        self.segmenter = Segmenter(edge_blur_ksize=21)
        self.renderer  = Renderer(blur_ksize=51, blur_sigma=20.0)

        self.tracking   = False
        self.bbox       = None
        self.mask       = None
        self.label      = ""
        self.conf       = 0.0
        self.frame_id   = 0
        self.last_dets: list[dict] = []
        self.detect_every = 15
        self.mask_every   = 3

    def process(self, msg: dict) -> tuple[np.ndarray | None, dict]:
        # ── Decode frame ────────────────────────────────────────────────
        frame_b64 = msg.get("frame", "")
        if not frame_b64:
            return None, {}

        try:
            raw   = base64.b64decode(frame_b64)
            arr   = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Frame decode failed: %s", exc)
            return None, {}

        if frame is None:
            logger.warning("imdecode returned None — bad JPEG?")
            return None, {}

        # ── Live tunable params ──────────────────────────────────────────
        if "detect_every" in msg:
            self.detect_every = max(1, int(msg["detect_every"]))
        if "bokeh_k" in msg:
            self.renderer.blur_ksize = int(msg["bokeh_k"]) | 1
        self.renderer.low_light = bool(msg.get("low_light", False))

        # ── Release ──────────────────────────────────────────────────────
        if msg.get("release"):
            self.tracker.reset()
            self.tracking = False
            self.bbox = self.mask = None
            self.label = ""
            self.conf  = 0.0

        cursor_x = msg.get("cursor_x")
        cursor_y = msg.get("cursor_y")
        clicked  = bool(msg.get("clicked", False))
        self.frame_id += 1

        # ── Periodic detection ───────────────────────────────────────────
        run_detect = (self.frame_id % self.detect_every == 0
                      or (clicked and not self.last_dets))
        if run_detect:
            self.last_dets = self.detector.detect(frame)

        # ── Click → lock ─────────────────────────────────────────────────
        if clicked and cursor_x is not None and cursor_y is not None:
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
        hover_box = None
        hover_label = ""
        if not clicked and cursor_x is not None and self.last_dets:
            hov = self.detector.pick_closest(self.last_dets, cursor_x, cursor_y)
            if hov:
                hx, hy, hw, hh = hov["bbox"]
                near_x = abs(hx + hw / 2 - cursor_x) < hw * 0.85
                near_y = abs(hy + hh / 2 - cursor_y) < hh * 0.85
                if near_x and near_y:
                    hover_box   = hov["bbox"]
                    hover_label = hov["label"]

        # ── Update tracker ───────────────────────────────────────────────
        if self.tracking and self.tracker.is_active:
            ok, new_bbox, track_conf = self.tracker.update(frame)
            if ok and new_bbox is not None:
                self.bbox = new_bbox
                self.conf = float(track_conf)
                if self.frame_id % self.mask_every == 0 or self.mask is None:
                    self.mask = self.segmenter.get_mask(
                        frame, new_bbox, self.tracker.label)
            else:
                logger.info("Tracker lost — re-detecting…")
                dets = self.detector.detect(frame)
                self.last_dets = dets
                if dets and self.bbox:
                    lx, ly, lw, lh = self.bbox
                    chosen = self.detector.pick_closest(dets, lx + lw // 2, ly + lh // 2)
                    if chosen:
                        self.tracker.reset()
                        self.tracker.init(
                            frame, chosen["bbox"], chosen["label"], chosen["conf"])
                        self.label = chosen["label"]
                        self.conf  = chosen["conf"]
                    else:
                        self.tracking = False
                        self.bbox = self.mask = None
                else:
                    self.tracking = False
                    self.bbox = self.mask = None

        # ── Render ───────────────────────────────────────────────────────
        fps    = self.renderer.tick_fps()
        output = self.renderer.render(
            frame=frame,
            mask=self.mask  if self.tracking else None,
            bbox=self.bbox  if self.tracking else None,
            label=self.label,
            conf=self.conf,
            fps=fps,
            hover_box=hover_box,
            hover_label=hover_label,
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


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h2>index.html not found.</h2>"
            "<p>Make sure static/index.html exists next to app.py</p>",
            status_code=404,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Quick check — visit http://localhost:8000/health to confirm server is up."""
    return JSONResponse({
        "status": "ok" if _shared_detector is not None else "loading",
        "models_ready": _shared_detector is not None,
        "startup_error": _startup_error,
        "opencv": cv2.__version__,
    })


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected from %s", ws.client)

    # If models failed to load, tell the client immediately and close
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

    logger.info("Session ready.")

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Bad JSON from client, skipping.")
                continue

            try:
                output, meta = session.process(msg)
            except Exception as exc:
                logger.error("process() error: %s", exc, exc_info=True)
                await ws.send_text(json.dumps({"error": str(exc)}))
                continue

            if output is None:
                continue

            try:
                ok, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if not ok:
                    continue
                payload = json.dumps({
                    "frame": base64.b64encode(buf.tobytes()).decode(),
                    **meta,
                })
                await ws.send_text(payload)
            except Exception as exc:
                logger.error("Send error: %s", exc)
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected normally.")
    except Exception as exc:
        logger.error("Unexpected WS error: %s", exc, exc_info=True)
    finally:
        logger.info("Session cleaned up.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  FOCUSR — AI Smart AutoFocus Server")
    print("  Open in browser: http://localhost:8000")
    print("  Health check:    http://localhost:8000/health")
    print("═" * 60 + "\n")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_ping_interval=20,   # keepalive pings every 20 s
        ws_ping_timeout=30,
    )