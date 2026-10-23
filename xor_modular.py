
import numpy as np

from mlp.network import MLP
from mlp.optimizers import GradientDescent

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

# Unser gemeinsames MLP
mlp = MLP([2, 2, 2])

# Gleiche Gewichte wie im alten xor.py
mlp.layers[0].W = np.array([
    [0.2, 0.4],
    [0.3, 0.1]
])

mlp.layers[0].theta = np.array([0.1, 0.2])

mlp.layers[1].W = np.array([
    [0.3, 0.7],
    [0.5, 0.2]
])

mlp.layers[1].theta = np.array([0.1, 0.2])

# Optimizer
optimizer = GradientDescent(learning_rate=0.5)

# Dieselbe Instanz wie zuvor
x = X[2]
target = T[2]

# Forward
output_before = mlp.forward(x)
loss_before = -np.sum(
    target * np.log(output_before)
)

# Backpropagation
gradients = mlp.backward(target)

# Gradient Descent
optimizer.step(mlp.layers, gradients)

# Erneuter Forward
output_after = mlp.forward(x)
loss_after = -np.sum(
    target * np.log(output_after)
)

print("----- Vor dem Update -----")
print("Output:", output_before)
print("CCE:", loss_before)

print("----- Nach dem Update -----")
print("Output:", output_after)
print("CCE:", loss_after)

print("Fehler kleiner?", loss_after < loss_before)
