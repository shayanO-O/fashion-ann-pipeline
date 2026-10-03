"""Stage 3 - train: build and train the fully connected ANN using params.yaml.

Architecture: Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax)
Run from the repository root:  python src/train.py
"""
import csv
import os

import numpy as np
import tensorflow as tf
import yaml

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model.h5")
HISTORY_PATH = os.path.join("models", "history.csv")

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

tf.keras.utils.set_random_seed(params["seed"])

train = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
val = np.load(os.path.join(PROCESSED_DIR, "val.npz"))

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params["dense_units"], activation="relu"),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    train["x"],
    train["y"],
    validation_data=(val["x"], val["y"]),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    verbose=2,
)

os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

keys = list(history.history.keys())
with open(HISTORY_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch"] + keys)
    for i in range(len(history.history[keys[0]])):
        writer.writerow([i + 1] + [history.history[k][i] for k in keys])

print(f"Saved model to {MODEL_PATH} and training history to {HISTORY_PATH}")
