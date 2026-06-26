# Emotion Recognition — Vision Transformer

A **ViT-Tiny** (`vit_tiny_patch16_224`, via `timm`) fine-tuned to classify a face into one of
7 emotions (`angry, disgust, fear, happy, neutral, sad, surprise`). FER-2013 faces are
resized to 224×224 and converted to 3-channel grayscale. Training uses mixed precision (AMP),
gradient clipping, AdamW + cosine annealing, label smoothing, class weights, and early
stopping. The notebook also renders confusion matrix, learning curves, a prediction grid, and
ViT self-attention maps.

Notebook: [`ViT.ipynb`](ViT.ipynb) · Pretrained weights: `outputs/vit_final_model/best_model.pth`

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

```bash
python download.py
```

This pulls FER-2013 from Kaggle (`msambare/fer2013`) into `data/FER2013/`, giving
`data/FER2013/train/<class>/` and `data/FER2013/test/<class>/`. Requires a Kaggle
account/token configured for `kagglehub`.

## Run

Open the notebook. The switch is `TRAIN_MODE` in the **Main Execution Pipeline** cell:

- `TRAIN_MODE = False` (default) — loads the bundled weights from
  `outputs/vit_final_model/best_model.pth` and runs evaluation only. (No `history.pt` is
  shipped, so the learning-curve plots are skipped on first run; the confusion matrix and
  attention maps still render.)
- `TRAIN_MODE = True` — fine-tunes from the pretrained ImageNet ViT and overwrites the weights
  in `outputs/vit_final_model/`.

Paths are resolved relative to this folder, so everything (data, outputs, weights) stays here.
