# FOCUSR — AI Smart AutoFocus

Real-time subject tracking and background blur, running entirely in the browser.

The browser captures video from a webcam or uploaded file, streams frames to a Python backend over WebSocket, and displays the processed output live. Click any subject to lock focus — it stays sharp while everything else blurs. Click a different subject to switch instantly.

---

## Preview

![FOCUSR interface](docs/screenshot.png)

> To add a screenshot: create a `docs/` folder in the project root, save your image there as `screenshot.png`, and the above will render automatically on GitHub.

---

## How It Works

**Click-to-lock** — Tap any point on the video. A detection model identifies the object at that coordinate and begins tracking it, with no need to draw a box or label anything.

**Continuous tracking** — Full neural network detection runs every N frames. A lightweight CSRT tracker holds onto the subject between detections, keeping performance viable on CPU.

**Photographic bokeh** — The blur is gradual, not a hard edge. The further a pixel is from the subject, the heavier the blur, approximating real depth-of-field.

**On-device** — No cloud, no GPU requirement. The full pipeline runs within the compute budget of a local machine.

---

## Architecture

```
Browser                          Python Backend (FastAPI)
───────                          ────────────────────────
Capture frame via canvas
Encode to JPEG + base64
Send cursor position,
click flag, and params
        |
        |──── WebSocket (JSON) ──────►  Decode frame
                                        |
                                        ├─ Every N frames: YOLOv8 detection
                                        |
                                        ├─ On click: map cursor to detection,
                                        |  initialise CSRT tracker
                                        |
                                        ├─ Every frame: CSRT update
                                        |   └─ If lost: re-detect
                                        |
                                        ├─ Every 3 frames: segmentation
                                        |   ├─ Person: MediaPipe SelfieSegmentation
                                        |   └─ Other: GrabCut
                                        |
                                        └─ Render
                                            ├─ Gaussian blur full frame
                                            ├─ Alpha-blend sharp subject via mask
                                            ├─ Hover box (amber, dashed)
                                            ├─ Lock box (green, animated)
                                            └─ FPS and status overlay
                                        |
        Draw on canvas          ◄──── WebSocket (JSON) ──── JPEG base64 response
        Update sidebar stats
```

---

## Installation

**Requirements:** Python 3.11.9

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**Optional — GPU acceleration (NVIDIA CUDA 11.8+)**

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Running

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in Chrome, Edge, or Firefox.

---

## Usage

1. Choose a source — click **Webcam** to use your camera, **Upload Video** to select an MP4 or MOV file, or drag and drop a video onto the window.
2. Hover over the video to see object detection highlights.
3. Click any object to lock focus. It stays sharp while the background blurs.
4. Click a different object to switch focus instantly.
5. Use the sidebar to adjust low-light enhancement, blur intensity, and detection interval.

**Keyboard shortcuts**

| Key    | Action              |
|--------|---------------------|
| Space  | Pause / resume      |
| Escape | Release focus lock  |
| W      | Switch to webcam    |
| F      | Toggle fullscreen   |