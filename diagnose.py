

import sys
import importlib
import subprocess

REQUIRED = [
    ("fastapi",       "fastapi",            "pip install fastapi"),
    ("uvicorn",       "uvicorn",            "pip install 'uvicorn[standard]'"),
    ("cv2",           "opencv-contrib-python", "pip install opencv-contrib-python"),
    ("numpy",         "numpy",              "pip install numpy"),
    ("ultralytics",   "ultralytics",        "pip install ultralytics"),
    ("mediapipe",     "mediapipe",          "pip install mediapipe"),
]

print("\n" + "═" * 60)
print("  FOCUSR — Dependency Diagnostics")
print("═" * 60)

all_ok = True

for mod, pkg, install in REQUIRED:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓  {mod:<20}  v{ver}")
    except ImportError:
        print(f"  ✗  {mod:<20}  MISSING  →  {install}")
        all_ok = False

print()

from pathlib import Path
model_path = Path("yolov8n.pt")
if model_path.exists():
    size_mb = model_path.stat().st_size / 1_000_000
    print(f"  ✓  yolov8n.pt            present ({size_mb:.1f} MB)")
else:
    print("  ○  yolov8n.pt            not downloaded yet (will auto-download on first run)")

html_path = Path("static") / "index.html"
if html_path.exists():
    print(f"  ✓  static/index.html     present")
else:
    print(f"  ✗  static/index.html     MISSING — make sure it's in ./static/")
    all_ok = False

import socket
print()
try:
    with socket.create_connection(("127.0.0.1", 8000), timeout=1):
        print("  ⚠  Port 8000:            already in use — stop the existing process first")
        print("     Kill it with:  lsof -ti:8000 | xargs kill -9   (Linux/macOS)")
        print("                    netstat -ano | findstr 8000       (Windows)")
except (ConnectionRefusedError, socket.timeout):
    print("  ✓  Port 8000             available")
except Exception as e:
    print(f"  ?  Port 8000 check error: {e}")


print()
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓  GPU (CUDA)            {torch.cuda.get_device_name(0)}")
    else:
        print("  ○  GPU (CUDA)            not available — will use CPU (still works, ~20 FPS)")
except Exception:
    pass

print()
try:
    import cv2
    t = cv2.TrackerCSRT_create()
    print("  ✓  cv2.TrackerCSRT       available")
except AttributeError:
    print("  ✗  cv2.TrackerCSRT       MISSING")
    print("     You have plain opencv-python — fix with:")
    print("     pip uninstall opencv-python -y")
    print("     pip install opencv-contrib-python")
    all_ok = False

print()
print("═" * 60)
if all_ok:
    print("  ✓  All checks passed — start the server with:")
    print("     python app.py")
    print("     Then open: http://localhost:8000")
else:
    print("  ✗  Issues found — fix the items marked ✗ above then retry.")
print("═" * 60 + "\n")
