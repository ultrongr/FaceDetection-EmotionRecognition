import os
import shutil

import kagglehub

# The notebook reads from data/FER2013/{train,test} relative to this folder.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "FER2013")

if os.path.isdir(DATA_DIR):
    print("Dataset already present at:", DATA_DIR)
else:
    path = kagglehub.dataset_download("msambare/fer2013")
    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
    shutil.move(path, DATA_DIR)
    print("Dataset downloaded to:", DATA_DIR)
