# FOCUSR — AI Smart AutoFocus (Web Edition)

Real-time browser-based subject tracking and background bokeh.  
The browser captures video (webcam or uploaded file), streams frames to a
Python backend over WebSocket, and displays AI-processed output live.

---

## 📁 Project Structure

```
smart_autofocus_web/
├── app.py              # FastAPI server + WebSocket pipeline
├── detector.py         # YOLOv8 object detector
├── tracker.py          # OpenCV CSRT tracker
├── segmenter.py        # MediaPipe / GrabCut mask generation
├── renderer.py         # Bokeh compositing & HUD
├── static/
│   └── index.html      # Full browser frontend (zero dependencies)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Create & activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. (Optional) GPU support — NVIDIA CUDA 11.8+
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

The YOLOv8 model (`yolov8n.pt`, ~6 MB) downloads automatically on first run.

---

## 🚀 Running

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open your browser at **http://localhost:8000**

---

## 🖥️ How to Use

1. **Open** `http://localhost:8000` in Chrome / Edge / Firefox.
2. **Choose a source**:
   - Click **◉ Webcam** to use your camera.
   - Click **⬆ Upload Video** to select an `.mp4` / `.mov` file.
   - Or **drag & drop** a video file onto the window.
3. **Hover** the cursor over the video — objects are highlighted with a
   preview box as you move the mouse.
4. **Click** on any highlighted object to **lock focus** — it stays sharp
   while the background blurs.
5. **Click again** on a different object to instantly **switch focus**.
6. Use the **sidebar controls** to:
   - Toggle low-light enhancement.
   - Adjust bokeh blur intensity.
   - Change detection interval (speed vs. accuracy trade-off).
7. **Keyboard shortcuts**:
   - `Space` — pause / resume
   - `Escape` — release lock
   - `W` — switch to webcam

---

## 🏗️ Architecture

```
Browser                          Python Server (FastAPI)
────────                         ──────────────────────────────
video element (src: file/webcam)
    │
    │ capture frame via <canvas>
    │ encode → JPEG → base64
    │ + cursor (x, y)
    │ + clicked flag
    │ + tunable params
    │──── WebSocket (JSON) ──────►  Decode frame
                                    │
                                    ├─ every N frames: YOLOv8 detect
                                    │
                                    ├─ on click: pick_closest() → CSRT.init()
                                    │
                                    ├─ every frame: CSRT.update()
                                    │   └─ if lost → re-detect
                                    │
                                    ├─ every 3 frames: segmenter.get_mask()
                                    │   ├─ person → MediaPipe SelfieSegmentation
                                    │   └─ other  → GrabCut
                                    │
                                    └─ renderer.render()
                                        ├─ Gaussian blur whole frame
                                        ├─ alpha-blend sharp subject via mask
                                        ├─ hover box (amber, dashed)
                                        ├─ lock box (green, animated)
                                        └─ FPS / status HUD
                                    │
    Decode base64 JPEG             │
    draw on <canvas>    ◄─── WebSocket (JSON) ──── encode JPEG → base64
    update sidebar HUD
```

### WebSocket message format

**Client → Server**
```json
{
  "frame":         "<base64 JPEG>",
  "cursor_x":      320,
  "cursor_y":      240,
  "clicked":       true,
  "low_light":     false,
  "bokeh_k":       51,
  "detect_every":  15,
  "release":       false
}
```

**Server → Client**
```json
{
  "frame":     "<base64 JPEG>",
  "tracking":  true,
  "label":     "person",
  "conf":      0.87,
  "fps":       24.3,
  "frame_id":  412,
  "det_count": 3
}
```

---

## ⚡ Performance Tips

| Goal | Setting |
|------|---------|
| Faster on slow CPU | Detect interval → 30–60 |
| Better accuracy | Use `yolov8s.pt` (change in `app.py`) |
| Less network load | Lower JPEG quality in `app.py` (line `IMWRITE_JPEG_QUALITY, 82`) |
| Low-light scenes | Toggle "Low-light boost" in sidebar |

Expected FPS on modern hardware:
- CPU only: **18–26 FPS**
- GPU (RTX): **28–35 FPS**

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` in browser | Make sure `python app.py` is running |
| Webcam not working | Check browser camera permissions |
| Low FPS | Increase detect interval slider to 30+ |
| Mask bleeds outside subject | Reduce `edge_blur_ksize` in `segmenter.py` |
| Server crashes on video | Install `opencv-contrib-python` not plain `opencv-python` |
| No module `mediapipe` | `pip install mediapipe` |

---

## 📜 License
MIT
