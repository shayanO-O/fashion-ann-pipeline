# fashion-ann-pipeline

End-to-end ML versioning with **Git**, **DVC** and **Google Drive**.
A fully connected ANN (not a CNN) classifies Fashion-MNIST images into 10 clothing categories.
The target is at least 85% test accuracy, reproducible with a single `dvc repro`.

## Pipeline

| Stage | Script | Output |
| --- | --- | --- |
| `prepare` | `src/prepare.py` | `data/raw/` (raw train/test arrays) |
| `preprocess` | `src/preprocess.py` | `data/processed/` (normalized train/val/test arrays) |
| `train` | `src/train.py` | `models/model.h5`, `models/history.csv` |
| `evaluate` | `src/evaluate.py` | `metrics.json`, `confusion_matrix.png` |

Model: `Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax)`, Adam optimizer,
sparse categorical crossentropy. All hyperparameters live in `params.yaml`.

## Setup (Windows, Command Prompt)

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

```
dvc repro          # runs only the stages whose code, data or params changed
dvc metrics show   # prints metrics.json
```

## Data and models

Raw data, processed data and models are versioned with DVC and stored in a Google Drive remote
(`gdrive_storage`). They are not stored in Git. To fetch them you need access to the Drive folder
and your own OAuth client (credentials are kept in `.dvc/config.local`, which is never committed):

```
dvc remote modify --local gdrive_storage gdrive_client_id <YOUR_CLIENT_ID>
dvc remote modify --local gdrive_storage gdrive_client_secret <YOUR_CLIENT_SECRET>
dvc pull
```

Or skip the download and rebuild everything with `dvc repro`.

## Versions

Git tags mark the pipeline states: `v1` (first reproduced pipeline) and `v2` (larger hidden layer).
Compare them with `dvc metrics diff v1 v2` and `dvc params diff v1 v2`.
