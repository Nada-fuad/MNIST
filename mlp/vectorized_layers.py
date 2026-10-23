
import numpy as np

from .layers import DenseLayer
from .activations import relu, softmax


class VectorizedDenseLayer(DenseLayer):

    def forward(self, x):
        self.x = np.asarray(x, dtype=float)

        # Matrixmultiplikation statt Schleifen
        self.net = self.x @ self.W - self.theta

        if self.activation == "relu":
            self.output = relu(self.net)
        elif self.activation == "softmax":
            self.output = softmax(self.net)
        else:
            raise ValueError("Unbekannte Aktivierungsfunktion")

        return self.output

    def backward(self, delta):
        delta = np.asarray(delta, dtype=float)

        if self.x.ndim == 1:
            # Einzelne Instanz
            grad_W = np.outer(self.x, delta)
            grad_theta = -delta.copy()
            grad_input = self.W @ delta

        else:
            # Mini-Batch: mittlere Gradienten
            batch_size = len(self.x)

            grad_W = self.x.T @ delta / batch_size
            grad_theta = -np.mean(delta, axis=0)
            grad_input = delta @ self.W.T

        return grad_input, grad_W, grad_theta
