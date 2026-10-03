"""Stage 4 - evaluate: score the model on the test set, write metrics.json and a confusion matrix.

Run from the repository root:  python src/evaluate.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # draw to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
TEST_PATH = os.path.join("data", "processed", "test.npz")
MODEL_PATH = os.path.join("models", "model.h5")

test = np.load(TEST_PATH)
x_test, y_test = test["x"], test["y"]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=CLASS_NAMES, xticks_rotation=45, ax=ax, colorbar=False
)
ax.set_title("Fashion-MNIST test set - confusion matrix")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=120)

metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"test_loss={test_loss:.4f}  test_accuracy={test_accuracy:.4f}")
print("Saved metrics.json and confusion_matrix.png")
