# Face Detection — ResNet backbone

A YOLO-style face detector built on a **frozen ResNet-18** backbone (ImageNet pretrained)
with a small convolutional detection head. The image is split into a 7×7 grid and each cell
predicts 2 boxes `[x, y, w, h, confidence]`. Trained and evaluated on **WIDER FACE**.

Notebook: [`face_detection_WIDER.ipynb`](face_detection_WIDER.ipynb)

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download **WIDER FACE** and unzip it inside this folder so the layout is:

```
FACE DETECTION ResNet/
├── WIDER_train/images/...
├── WIDER_val/images/...
└── wider_face_split/
    ├── wider_face_train_bbx_gt.txt
    └── wider_face_val_bbx_gt.txt
```

Sources:
- Images: `WIDER_train.zip`, `WIDER_val.zip` — http://shuoyang1213.me/WIDERFACE/
  (mirror: `https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip`)
- Annotations: `wider_face_split.zip` — http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip

If the dataset is missing the notebook prints the download link and falls back to a tiny
random "dummy" dataset, so the cells still run end-to-end without it.

## Run

Open the notebook and run all cells. It trains for 10 epochs, saves the best weights to
`best_face_detector.pth`, plots the loss curves to `training_curves.png`, and provides a
`visualize_predictions(image_path, model, device)` helper to draw detected boxes on an image.
