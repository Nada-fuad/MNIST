
import numpy as np

from mlp.network import MLP
from mlp.vectorized_layers import VectorizedDenseLayer


# Kleine Testdaten: 3 Instanzen, 2 Eingaben
X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

T = np.array([
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

model = MLP(
    [2, 3, 2],
    layer_class=VectorizedDenseLayer
)

# Feste positive Gewichte für reproduzierbaren Test
model.layers[0].W[:] = [
    [0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7]
]
model.layers[0].theta[:] = 0.0

model.layers[1].W[:] = [
    [0.2, 0.4],
    [0.3, 0.1],
    [0.5, 0.2]
]
model.layers[1].theta[:] = 0.0


def loss():
    output = model.forward(X)
    return -np.mean(
        np.sum(T * np.log(output + 1e-12), axis=1)
    )


# Analytische Gradienten durch Backpropagation
model.forward(X)
gradients = model.backward(T)

# Numerische Gradienten
h = 1e-6
max_error = 0.0

for layer, (grad_W, grad_theta) in zip(
    model.layers, gradients
):
    for parameter, analytical in [
        (layer.W, grad_W),
        (layer.theta, grad_theta)
    ]:
        for index in np.ndindex(parameter.shape):
            old_value = parameter[index]

            parameter[index] = old_value + h
            loss_plus = loss()

            parameter[index] = old_value - h
            loss_minus = loss()

            parameter[index] = old_value

            numerical = (
                loss_plus - loss_minus
            ) / (2 * h)

            error = abs(numerical - analytical[index])
            max_error = max(max_error, error)

print("Maximaler Gradientenfehler:", max_error)

assert max_error < 1e-6

print("Mini-Batch-Gradientencheck bestanden!")
