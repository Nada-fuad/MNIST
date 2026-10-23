
import numpy as np
from mlp.network import MLP

# XOR-Netz wie in der Vorlesung
mlp = MLP([2, 2, 2])

# Gewichte der versteckten Schicht
mlp.layers[0].W = np.array([
    [1.0, 1.0],
    [1.0, -1.0]
])
mlp.layers[0].theta = np.zeros(2)

# Gewichte der Ausgabeschicht
mlp.layers[1].W = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])
mlp.layers[1].theta = np.zeros(2)

x = np.array([1.0, 0.0])
target = np.array([0.0, 1.0])
eta = 0.5

# 1. Forward
output_before = mlp.forward(x)
print("Vorher:", output_before)

# 2. Backpropagation
gradients = mlp.backward(target)

# 3. Gradient Descent
for layer, (grad_W, grad_theta) in zip(
    mlp.layers, gradients
):
    layer.W -= eta * grad_W
    layer.theta -= eta * grad_theta

# 4. Forward mit neuen Gewichten
output_after = mlp.forward(x)
print("Nachher:", output_after)

# 5. Referenztest
assert np.allclose(
    output_after,
    [0.07585818, 0.92414182],
    atol=1e-6
)

print("MLP-Referenztest bestanden!")
