
import numpy as np
from .activations import relu, relu_derivative, softmax


class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation="relu"):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        # Gewichte zufällig initialisieren
        rng = np.random.default_rng()
        self.W = rng.normal(
            0, np.sqrt(2 / n_inputs),
            size=(n_inputs, n_neurons)
        )

        # Schwellwerte
        self.theta = np.zeros(n_neurons)

        # Speicher für Backpropagation
        self.x = None
        self.net = None
        self.output = None

    def forward(self, x):
        self.x = np.asarray(x, dtype=float)
        self.net = np.zeros(self.n_neurons)

        # Explizite Schleifen wie bei XOR
        for j in range(self.n_neurons):
            for i in range(self.n_inputs):
                self.net[j] += self.x[i] * self.W[i, j]
            self.net[j] -= self.theta[j]

        if self.activation == "relu":
            self.output = relu(self.net)
        elif self.activation == "softmax":
            self.output = softmax(self.net)
        else:
            raise ValueError("Unbekannte Aktivierungsfunktion")

        return self.output

    def backward(self, delta):
        # delta = dE/dnet dieser Schicht
        grad_W = np.zeros_like(self.W)
        grad_theta = -delta.copy()

        for i in range(self.n_inputs):
            for j in range(self.n_neurons):
                grad_W[i, j] = self.x[i] * delta[j]

        # Fehlersignal für die vorherige Schicht
        grad_input = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            for j in range(self.n_neurons):
                grad_input[i] += self.W[i, j] * delta[j]

        return grad_input, grad_W, grad_theta
