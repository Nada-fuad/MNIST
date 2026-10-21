
import numpy as np

# Unsere XOR-Testdaten
x = np.array([1.0, 0.0])
target = np.array([0.0, 1.0])

# Gewichte und Schwellwerte
W1 = np.array([[0.2, 0.4],
               [0.3, 0.1]])
theta1 = np.array([0.1, 0.2])

W2 = np.array([[0.3, 0.7],
               [0.5, 0.2]])
theta2 = np.array([0.1, 0.2])


def forward():
    # Hidden Layer mit Schleifen
    net1 = np.zeros(2)

    for j in range(2):
        for i in range(2):
            net1[j] += x[i] * W1[i, j]
        net1[j] -= theta1[j]

    hidden = np.maximum(0, net1)

    # Output Layer mit Schleifen
    net2 = np.zeros(2)

    for j in range(2):
        for i in range(2):
            net2[j] += hidden[i] * W2[i, j]
        net2[j] -= theta2[j]

    # Softmax
    z = net2 - np.max(net2)
    exp_z = np.exp(z)
    output = exp_z / np.sum(exp_z)

    # CCE
    loss = -np.sum(target * np.log(output))

    return loss, net1, hidden, output


# 1. Analytische Gradienten (Backpropagation)
loss, net1, hidden, output = forward()

delta2 = output - target
grad_W2 = np.outer(hidden, delta2)
grad_theta2 = -delta2

delta1 = np.zeros(2)

for i in range(2):
    for j in range(2):
        delta1[i] += W2[i, j] * delta2[j]

    if net1[i] <= 0:
        delta1[i] = 0.0

grad_W1 = np.outer(x, delta1)
grad_theta1 = -delta1


# 2. Numerische Gradienten
h = 1e-6

def check_parameter(parameter, gradient, name):
    for index in np.ndindex(parameter.shape):
        original = parameter[index]

        parameter[index] = original + h
        loss_plus = forward()[0]

        parameter[index] = original - h
        loss_minus = forward()[0]

        parameter[index] = original

        numerical = (loss_plus - loss_minus) / (2 * h)
        analytical = gradient[index]

        relative_error = abs(numerical - analytical) / max(
            1e-12, abs(numerical) + abs(analytical)
        )

        print(name, index, "Fehler:", relative_error)

        assert relative_error < 1e-6


check_parameter(W1, grad_W1, "W1")
check_parameter(theta1, grad_theta1, "theta1")
check_parameter(W2, grad_W2, "W2")
check_parameter(theta2, grad_theta2, "theta2")

print("Gradientencheck bestanden!")
