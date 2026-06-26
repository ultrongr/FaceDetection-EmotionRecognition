# Face Detection — Vision Transformer backbone

The same YOLO-style face detector as the ResNet version, but with the backbone swapped for a
**DeiT-Small Vision Transformer** (ImageNet pretrained, last 6 blocks fine-tuned). The 196
patch tokens are reshaped into a 14×14 feature map and fed to a convolutional YOLO head that
predicts 2 boxes `[x, y, w, h, confidence]` per cell. Trained on **WIDER FACE**.

The notebook also includes a learning-rate / λ_coord hyperparameter search, a confidence-
threshold sweep (precision / recall / F1), detection visualisations vs. ground truth, and
self-attention map plots. Commentary is in Greek.

Notebook: [`ViT_YOLO.ipynb`](ViT_YOLO.ipynb) · Pretrained weights: `final_vit_yolo.pth`

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

**No manual download needed.** Cell 2 of the notebook fetches and unzips WIDER FACE with
`wget`/`unzip`:

```
WIDER_train.zip, WIDER_val.zip   (HuggingFace mirror of WIDER FACE)
wider_face_split.zip             (CUHK annotations)
```

> These cells use `wget`/`unzip`, so they run as-is on Linux / Google Colab. On Windows,
> either run them under WSL/Git-Bash or download the three zips manually (see the
> `FACE DETECTION ResNet` README for the URLs) and unzip them into this folder.

## Run

Open the notebook and run all cells to reproduce training and evaluation. Full training
saves `final_vit_yolo.pth` (already included here, so you can skip training and run only the
evaluation / visualisation cells against the shipped weights).
