"""Stage 1 - prepare: download Fashion-MNIST and save the raw arrays to data/raw/.

Run from the repository root:  python src/prepare.py
"""
import os

import numpy as np
import tensorflow as tf

OUT_DIR = os.path.join("data", "raw")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

os.makedirs(OUT_DIR, exist_ok=True)
np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), x=x_train, y=y_train)
np.savez_compressed(os.path.join(OUT_DIR, "test.npz"), x=x_test, y=y_test)

print(f"Saved raw data to {OUT_DIR}/  train={x_train.shape}  test={x_test.shape}")
