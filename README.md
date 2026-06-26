# Face Detection & Emotion Recognition

A computer-vision course project covering two tasks with two model families each, plus a
real-time demo that combines them:

| Task | Folder | Model | Dataset |
|------|--------|-------|---------|
| Face detection | [`FACE DETECTION ResNet/`](FACE%20DETECTION%20ResNet/) | ResNet-18 backbone + YOLO-style head | WIDER FACE |
| Face detection | [`FACE DETECTION ViT/`](FACE%20DETECTION%20ViT/) | DeiT-Small (ViT) backbone + YOLO-style head | WIDER FACE |
| Emotion recognition | [`Emotion Recognition CNN/`](Emotion%20Recognition%20CNN/) | VGG-style CNN | FER-2013 |
| Emotion recognition | [`Emotion Recognition ViT/`](Emotion%20Recognition%20ViT/) | ViT-Tiny | FER-2013 |
| Real-time demo | [`BONUS/`](BONUS/) | ViT-YOLO detector + CNN emotion classifier | webcam |

Both face detectors predict bounding boxes on a 14×14 / 7×7 grid (YOLO encoding). Both
emotion models classify a face into one of 7 classes: `angry, disgust, fear, happy,
neutral, sad, surprise`. The BONUS app wires the ViT face detector and the CNN emotion
classifier together over a live webcam feed.

## Layout

Each subfolder is **self-contained**: it has its own `requirements.txt`, its own
README with run instructions, and reads/writes data and weights relative to itself. Pick a
folder and follow its README — nothing depends on a sibling folder.

## Quick start

Every subproject runs the same way:

```bash
cd "<subfolder>"
python -m venv .venv && .venv\Scripts\activate     # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
jupyter notebook        # then open the .ipynb (BONUS is a plain script: python camera.py)
```

A GPU is recommended for training but not required — every notebook falls back to CPU, and
each one ships a pre-trained model so you can run inference without training.

See each folder's `README.md` for dataset download steps and the train-vs-load switch.
