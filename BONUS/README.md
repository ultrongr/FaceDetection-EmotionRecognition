# BONUS — Real-time Face Detection + Emotion Recognition

A live webcam demo that combines the two models from this project:

1. the **ViT-YOLO face detector** (`final_vit_yolo.pth`) locates faces in each frame, then
2. the **CNN emotion classifier** (`best_model.pt`) labels each detected face with an emotion.

Detected faces are drawn with a box and an `Emotion (confidence)` label. Both weight files are
already included in this folder, so no training is needed.

Script: [`camera.py`](camera.py)

## Setup

```bash
pip install -r requirements.txt
```

(`camera.py` also tries to auto-install `pytorch-lightning`, `timm`, and `opencv-python` if
they're missing.)

## Run

```bash
python camera.py
```

A window opens showing the webcam feed with live detections. Press **`q`** to quit. Requires a
working camera (device index 0). Runs on GPU if available, otherwise CPU.

Tunable constants near the top of `run_webcam()`: `CONF_THRESH` (detection confidence) and
`IOU_THRESH` (non-max suppression).
