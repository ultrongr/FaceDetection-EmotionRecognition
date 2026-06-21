import os
import shutil

import kagglehub

FER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fer")

if os.path.isdir(FER_DIR):
    print("Dataset already present at:", FER_DIR)
else:
    path = kagglehub.dataset_download("msambare/fer2013")
    shutil.move(path, FER_DIR)
    print("Dataset downloaded to:", FER_DIR)
