
import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, layers, gradients):
        for layer, (grad_W, grad_theta) in zip(
            layers, gradients
        ):
            layer.W -= self.learning_rate * grad_W
            layer.theta -= self.learning_rate * grad_theta


class Adam:
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam speichert vergangene Gradienten
        self.t = 0
        self.m = None
        self.v = None

    def step(self, layers, gradients):

        # Speicher beim ersten Aufruf erstellen
        if self.m is None:
            self.m = [
                (
                    np.zeros_like(layer.W),
                    np.zeros_like(layer.theta)
                )
                for layer in layers
            ]

            self.v = [
                (
                    np.zeros_like(layer.W),
                    np.zeros_like(layer.theta)
                )
                for layer in layers
            ]

        self.t += 1

        for i, (layer, (grad_W, grad_theta)) in enumerate(
            zip(layers, gradients)
        ):
            mW, mTheta = self.m[i]
            vW, vTheta = self.v[i]

            # 1. Moment: Mittelwert der Gradienten
            mW[:] = (
                self.beta1 * mW
                + (1 - self.beta1) * grad_W
            )
            mTheta[:] = (
                self.beta1 * mTheta
                + (1 - self.beta1) * grad_theta
            )

            # 2. Moment: Quadratische Gradienten
            vW[:] = (
                self.beta2 * vW
                + (1 - self.beta2) * grad_W**2
            )
            vTheta[:] = (
                self.beta2 * vTheta
                + (1 - self.beta2) * grad_theta**2
            )

            # Bias-Korrektur
            mW_hat = mW / (1 - self.beta1**self.t)
            mTheta_hat = mTheta / (1 - self.beta1**self.t)

            vW_hat = vW / (1 - self.beta2**self.t)
            vTheta_hat = vTheta / (1 - self.beta2**self.t)

            # Gewichte aktualisieren
            layer.W -= (
                self.learning_rate * mW_hat
                / (np.sqrt(vW_hat) + self.epsilon)
            )

            # Schwellwerte aktualisieren
            layer.theta -= (
                self.learning_rate * mTheta_hat
                / (np.sqrt(vTheta_hat) + self.epsilon)
            )
