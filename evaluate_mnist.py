
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_mnist
from mlp.network import MLP
from mlp.vectorized_layers import VectorizedDenseLayer


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# 1. Gespeichertes Modell laden
model = MLP(
    [784, 30, 10],
    layer_class=VectorizedDenseLayer
)

with np.load(RESULTS / "mnist_model.npz") as saved:
    model.layers[0].W[:] = saved["W1"]
    model.layers[0].theta[:] = saved["theta1"]
    model.layers[1].W[:] = saved["W2"]
    model.layers[1].theta[:] = saved["theta2"]

# 2. Testdaten laden
_, _, _, _, X_test, T_test = load_mnist()

# 3. Vorhersagen
output = model.forward(X_test)
predictions = np.argmax(output, axis=1)
targets = np.argmax(T_test, axis=1)

accuracy = np.mean(predictions == targets)
print(f"Geladenes Modell - Test Accuracy: {accuracy:.2%}")

# 4. Confusion Matrix ohne scikit-learn
cm = np.zeros((10, 10), dtype=int)

for true_class, predicted_class in zip(targets, predictions):
    cm[true_class, predicted_class] += 1

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")

ax.set_title("MNIST - Confusion Matrix")
ax.set_xlabel("Vorhergesagte Klasse")
ax.set_ylabel("Tatsächliche Klasse")
ax.set_xticks(range(10))
ax.set_yticks(range(10))

for i in range(10):
    for j in range(10):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            fontsize=8,
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(RESULTS / "confusion_matrix.png", dpi=180)
plt.close(fig)

# 5. Trainingskurven
with np.load(RESULTS / "mnist_history.npz") as history:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

epochs = np.arange(1, len(train_loss) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, label="Training")
ax.plot(epochs, val_loss, label="Validation")
ax.set_title("MNIST - Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Categorical Cross-Entropy")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "loss_curve.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_acc * 100, label="Training")
ax.plot(epochs, val_acc * 100, label="Validation")
ax.set_title("MNIST - Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "accuracy_curve.png", dpi=180)
plt.close(fig)

print("Confusion Matrix und Trainingskurven gespeichert!")
