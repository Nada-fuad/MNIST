
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_images(dataframe):
    X = np.empty((len(dataframe), 784), dtype=np.float32)

    for i, image_data in enumerate(dataframe["image"]):
        if image_data["bytes"] is not None:
            image = Image.open(BytesIO(image_data["bytes"]))
        else:
            image = Image.open(DATA_DIR / image_data["path"])

        pixels = np.asarray(image.convert("L"), dtype=np.float32)
        X[i] = pixels.reshape(784) / 255.0
        image.close()

    return X


def one_hot(labels):
    T = np.zeros((len(labels), 10), dtype=np.float32)
    T[np.arange(len(labels)), labels] = 1.0
    return T


def load_mnist():
    train_files = sorted(DATA_DIR.rglob("train-*.parquet"))
    test_files = sorted(DATA_DIR.rglob("test-*.parquet"))

    if not train_files or not test_files:
        raise FileNotFoundError("MNIST-Dateien fehlen in data/")

    train_df = pd.concat(
        [pd.read_parquet(file) for file in train_files],
        ignore_index=True
    )
    test_df = pd.concat(
        [pd.read_parquet(file) for file in test_files],
        ignore_index=True
    )

    X = load_images(train_df)
    y = train_df["label"].to_numpy(dtype=np.int64)
    T = one_hot(y)

    X_test = load_images(test_df)
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    T_test = one_hot(y_test)

    # Zufällige, reproduzierbare Aufteilung
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))

    train_idx = indices[:50000]
    val_idx = indices[50000:]

    return (
        X[train_idx], T[train_idx],
        X[val_idx], T[val_idx],
        X_test, T_test
    )


if __name__ == "__main__":
    X_train, T_train, X_val, T_val, X_test, T_test = load_mnist()

    print("Training:", X_train.shape, T_train.shape)
    print("Validierung:", X_val.shape, T_val.shape)
    print("Test:", X_test.shape, T_test.shape)
    print("Pixelbereich:", X_train.min(), X_train.max())
