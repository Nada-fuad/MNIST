
import numpy as np

from mlp.layers import DenseLayer
from mlp.vectorized_layers import VectorizedDenseLayer

# Gleiche Architektur
loop_layer = DenseLayer(2, 2, "relu")
vector_layer = VectorizedDenseLayer(2, 2, "relu")

# Gleiche Gewichte und Schwellwerte
W = np.array([
    [0.2, 0.4],
    [0.3, 0.1]
])

theta = np.array([0.1, 0.2])

loop_layer.W = W.copy()
vector_layer.W = W.copy()

loop_layer.theta = theta.copy()
vector_layer.theta = theta.copy()

# Gleiche Eingabe
x = np.array([1.0, 0.0])

# Forward vergleichen
output_loop = loop_layer.forward(x)
output_vector = vector_layer.forward(x)

print("Schleifen:", output_loop)
print("Vektorisiert:", output_vector)

assert np.allclose(output_loop, output_vector)

# Backward vergleichen
delta = np.array([-0.2, 0.15])

grad_loop = loop_layer.backward(delta)
grad_vector = vector_layer.backward(delta)

for a, b in zip(grad_loop, grad_vector):
    assert np.allclose(a, b)

print("Forward und Backward stimmen überein!")
