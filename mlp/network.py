
import numpy as np

from .layers import DenseLayer


class MLP:
    def __init__(self, sizes, layer_class=DenseLayer):
        self.layers = []

        for i in range(len(sizes) - 1):

            if i == len(sizes) - 2:
                activation = "softmax"
            else:
                activation = "relu"

            layer = layer_class(
                sizes[i],
                sizes[i + 1],
                activation
            )

            self.layers.append(layer)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, target):
        output = self.layers[-1].output

        # Softmax + CCE
        delta = output - target

        gradients = []

        # Von der letzten zur ersten Layer
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            grad_input, grad_W, grad_theta = (
                layer.backward(delta)
            )

            gradients.insert(
                0, (grad_W, grad_theta)
            )

            if i > 0:
                previous_layer = self.layers[i - 1]

                delta = grad_input * (
                    previous_layer.net > 0
                )

        return gradients
