
import numpy as np

from mlp.network import MLP
from mlp.vectorized_layers import VectorizedDenseLayer
from mlp.optimizers import Adam
from mlp.minibatch_trainer import MiniBatchTrainer


# XOR-Daten
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

T = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
], dtype=float)


# Vektorisiertes MLP
model = MLP(
    [2, 4, 2],
    layer_class=VectorizedDenseLayer
)

optimizer = Adam(learning_rate=0.01)

trainer = MiniBatchTrainer(model, optimizer)

# Ein Trainingstest
history = trainer.train(
    X_train=X,
    T_train=T,
    epochs=100,
    batch_size=4
)

assert len(history["train_loss"]) == 100
assert np.isfinite(history["train_loss"]).all()

print("MiniBatchTrainer-Test bestanden!")
print("Final Accuracy:", trainer.accuracy(X, T))
