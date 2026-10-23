
import numpy as np


def relu(x):
    """ReLU-Aktivierungsfunktion."""
    return np.maximum(0.0, x)


def relu_derivative(x):
    """Ableitung der ReLU-Funktion."""
    return (x > 0).astype(float)


def softmax(x):
    """Stabile Softmax-Funktion."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values, axis=-1, keepdims=True
    )
