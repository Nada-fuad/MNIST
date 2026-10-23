
import numpy as np
from mlp.network import MLP

# Zufällige Gewichte, aber reproduzierbar
np.random.seed(42)

model = MLP([2, 4, 2])

rng = np.random.default_rng(42)

for layer in model.layers:
    layer.W = rng.uniform(
        -0.5, 0.5, size=layer.W.shape
    )
    layer.theta = rng.uniform(
        -0.2, 0.2, size=layer.theta.shape
    )

# Testeingabe und Target
x = np.array([1.0, 1.0])
target = np.array([0.0, 1.0])

def loss():
    output = model.forward(x)
    return -np.sum(target * np.log(output))

# Analytische Gradienten
model.forward(x)
gradients = model.backward(target)

h = 1e-6
max_error = 0.0

# Alle Gewichte und Schwellwerte prüfen
for layer, (grad_W, grad_theta) in zip(
    model.layers, gradients
):
    for parameter, gradient in [
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

            numerical = (loss_plus - loss_minus) / (2 * h)
            analytical = gradient[index]

            error = abs(numerical - analytical) / max(
                1e-12,
                abs(numerical) + abs(analytical)
            )

            max_error = max(max_error, error)

            assert error < 1e-6, (
                f"Gradientenfehler bei {index}: {error}"
            )

print("Maximaler relativer Fehler:", max_error)
print("Zufälliger Gradientencheck bestanden!")
