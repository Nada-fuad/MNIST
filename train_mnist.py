
from pathlib import Path

import numpy as np

from data_loader import load_mnist
from mlp.network import MLP
from mlp.vectorized_layers import VectorizedDenseLayer
from mlp.optimizers import Adam
from mlp.minibatch_trainer import MiniBatchTrainer


# 1. MNIST-Daten laden
print("Lade MNIST-Daten ...")

X_train, T_train, X_val, T_val, X_test, T_test = load_mnist()

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# 2. MLP erstellen: 784 -> 30 -> 10
model = MLP(
    [784, 30, 10],
    layer_class=VectorizedDenseLayer
)


# 3. Adam Optimizer
optimizer = Adam(learning_rate=0.001)


# 4. Mini-Batch-Trainer
trainer = MiniBatchTrainer(model, optimizer)


# 5. Training starten
history = trainer.train(
    X_train=X_train,
    T_train=T_train,
    X_val=X_val,
    T_val=T_val,
    epochs=20,
    batch_size=64,
    patience=5,
    seed=42
)


# 6. Finale Evaluation auf Testdaten
test_loss = trainer.loss(X_test, T_test)
test_accuracy = trainer.accuracy(X_test, T_test)

print("\n----- MNIST TESTERGEBNIS -----")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")


# 7. Gewichte speichern
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

np.savez(
    output_dir / "mnist_model.npz",
    W1=model.layers[0].W,
    theta1=model.layers[0].theta,
    W2=model.layers[1].W,
    theta2=model.layers[1].theta
)

print("Modell gespeichert: results/mnist_model.npz")


# 8. Trainingsverlauf speichern
np.savez(
    output_dir / "mnist_history.npz",
    **{key: np.asarray(value)
       for key, value in history.items()}
)

print("Training gespeichert: results/mnist_history.npz")
