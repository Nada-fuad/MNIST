
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# Pfad zum Projektordner
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# MNIST-Dateien finden
train_files = sorted(DATA_DIR.rglob("train-*.parquet"))
test_files = sorted(DATA_DIR.rglob("test-*.parquet"))

if not train_files or not test_files:
    raise FileNotFoundError(
        "MNIST-Parquet-Dateien fehlen im Ordner data."
    )

# Daten laden
train_df = pd.concat(
    [pd.read_parquet(file) for file in train_files],
    ignore_index=True
)

test_df = pd.concat(
    [pd.read_parquet(file) for file in test_files],
    ignore_index=True
)

print("Trainingsbilder:", len(train_df))
print("Testbilder:", len(test_df))

# Erstes Bild auslesen
image_data = train_df.iloc[0]["image"]

if image_data["bytes"] is not None:
    image = Image.open(BytesIO(image_data["bytes"]))
else:
    image = Image.open(
        DATA_DIR / image_data["path"]
    )

pixels = np.array(image.convert("L"))

print("Bildform:", pixels.shape)
print("Label:", train_df.iloc[0]["label"])

# Bild anzeigen
plt.imshow(pixels, cmap="gray")
plt.title(f"MNIST – Label: {train_df.iloc[0]['label']}")
plt.axis("off")
plt.show()




# Erstes Bild in einen Vektor umwandeln
x = pixels.flatten()

# Pixelwerte normalisieren
x = x.astype(np.float32) / 255.0

print("Nach Flatten:", x.shape)
print("Kleinster Pixelwert:", x.min())
print("Größter Pixelwert:", x.max())

# One-Hot-Encoding
label = int(train_df.iloc[0]["label"])

target = np.zeros(10)
target[label] = 1

print("One-Hot-Target:", target)

