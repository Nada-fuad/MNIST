
import numpy as np

from mlp.network import MLP
from mlp.vectorized_layers import VectorizedDenseLayer


# Vier künstliche Testbilder
rng = np.random.default_rng(42)

X = rng.random((4, 784))

# Die vier richtigen Klassen: 0, 3, 5, 9
labels = np.array([0, 3, 5, 9])

T = np.zeros((4, 10))
T[np.arange(4), labels] = 1.0


# MLP mit vektorisierten Schichten
model = MLP(
    [784, 30, 10],
    layer_class=VectorizedDenseLayer
)

# 1. Forward
output = model.forward(X)

print("Input:", X.shape)
print("Output:", output.shape)
print("Softmax-Summen:", output.sum(axis=1))

assert output.shape == (4, 10)
assert np.allclose(output.sum(axis=1), 1.0)


# 2. Backpropagation
gradients = model.backward(T)

for i, (grad_W, grad_theta) in enumerate(gradients):
    print(
        f"Layer {i + 1}:",
        "grad_W =", grad_W.shape,
        "grad_theta =", grad_theta.shape
    )

assert gradients[0][0].shape == (784, 30)
assert gradients[0][1].shape == (30,)
assert gradients[1][0].shape == (30, 10)
assert gradients[1][1].shape == (10,)

print("Mini-Batch-Test bestanden!")
