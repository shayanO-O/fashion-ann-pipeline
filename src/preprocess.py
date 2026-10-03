"""Stage 2 - preprocess: normalize pixels to [0, 1] and split off a validation set.

Run from the repository root:  python src/preprocess.py
"""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")

with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

train = np.load(os.path.join(RAW_DIR, "train.npz"))
test = np.load(os.path.join(RAW_DIR, "test.npz"))
x_train, y_train = train["x"], train["y"]
x_test, y_test = test["x"], test["y"]

# --- Normalization: per-image min-max to [0, 1], then global standardization ---
def minmax(x):
    x = x.astype("float32")
    lo = x.min(axis=(1, 2), keepdims=True)
    hi = x.max(axis=(1, 2), keepdims=True)
    return (x - lo) / (hi - lo + 1e-7)

x_train = minmax(x_train)
x_test = minmax(x_test)
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
# --- end of normalization ---

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train,
    y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=y_train,
)

os.makedirs(OUT_DIR, exist_ok=True)
np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), x=x_tr, y=y_tr)
np.savez_compressed(os.path.join(OUT_DIR, "val.npz"), x=x_val, y=y_val)
np.savez_compressed(os.path.join(OUT_DIR, "test.npz"), x=x_test, y=y_test)

print(
    f"Saved processed data to {OUT_DIR}/  "
    f"train={x_tr.shape}  val={x_val.shape}  test={x_test.shape}  "
    f"pixel range=[{x_tr.min():.2f}, {x_tr.max():.2f}]"
)
