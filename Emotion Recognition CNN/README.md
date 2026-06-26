# Emotion Recognition — CNN

A VGG-style **CNN** that classifies a 48×48 grayscale face into one of 7 emotions
(`angry, disgust, fear, happy, neutral, sad, surprise`). Four conv blocks (64→128→256→512)
feed a global-average-pooled dense head. Trained on **FER-2013** with class-weighted
cross-entropy (the `disgust` class is tiny), augmentation, `ReduceLROnPlateau`, and early
stopping. The shipped run reaches ~68% test accuracy.

Notebook: [`cnn.ipynb`](cnn.ipynb)

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

```bash
python download.py
```

This pulls FER-2013 from Kaggle (`msambare/fer2013`) into `fer/`, giving `fer/train/<class>/`
and `fer/test/<class>/`. Requires a Kaggle account/token configured for `kagglehub`. The
`fer/` folder is git-ignored.

## Run

Open the notebook. The one switch that matters is in the **Config** cell:

- `LOAD_PRETRAINED = True` (default) — loads the latest saved run from `outputs/` and runs
  evaluation only. The repo ships one such run at `outputs/20260606_160733/`.
- `LOAD_PRETRAINED = False` — trains from scratch (~1 min/epoch on GPU) and saves a new
  timestamped run under `outputs/`.

Either path produces the test-set accuracy, a classification report, a normalised confusion
matrix, a grid of misclassified faces, and dumps every mistake to
`outputs/<run>/wrongly_categorized_<split>/`.
