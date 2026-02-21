# FOCUSR — AI Smart AutoFocus (Web Edition)

Real-time browser-based subject tracking and background bokeh.  
The browser captures video (webcam or uploaded file), streams frames to a
Python backend over WebSocket, and displays AI-processed output live.

---



## Installation

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


---

##  Running

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open your browser at **http://localhost:8000**

---

##  How to Use

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

##  Architecture

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




