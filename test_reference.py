
import numpy as np

# Werte aus der Vorlesung
x = np.array([1.0, 0.0])
target = np.array([0.0, 1.0])

W1 = np.array([
    [1.0, 1.0],
    [1.0, -1.0]
])

W2 = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

theta1 = np.zeros(2)
theta2 = np.zeros(2)

eta = 0.5


def forward():
    # Hidden Layer
    net1 = np.zeros(2)

    for j in range(2):
        for i in range(2):
            net1[j] += x[i] * W1[i, j]
        net1[j] -= theta1[j]

    hidden = np.maximum(0, net1)

    # Output Layer
    net2 = np.zeros(2)

    for j in range(2):
        for i in range(2):
            net2[j] += hidden[i] * W2[i, j]
        net2[j] -= theta2[j]

    # Softmax
    z = net2 - np.max(net2)
    exp_z = np.exp(z)
    output = exp_z / np.sum(exp_z)

    loss = -np.sum(target * np.log(output))

    return net1, hidden, output, loss


# 1. Vorwärts
net1, hidden, output, loss = forward()

print("Vor dem Training:", output)
print("CCE vorher:", loss)

# 2. Backpropagation
delta2 = output - target

grad_W2 = np.zeros_like(W2)
for i in range(2):
    for j in range(2):
        grad_W2[i, j] = hidden[i] * delta2[j]

grad_theta2 = -delta2

delta1 = np.zeros(2)

for i in range(2):
    for j in range(2):
        delta1[i] += W2[i, j] * delta2[j]

    if net1[i] <= 0:
        delta1[i] = 0.0

grad_W1 = np.zeros_like(W1)

for i in range(2):
    for j in range(2):
        grad_W1[i, j] = x[i] * delta1[j]

grad_theta1 = -delta1

# 3. Gewichte aktualisieren
W1 -= eta * grad_W1
W2 -= eta * grad_W2

theta1 -= eta * grad_theta1
theta2 -= eta * grad_theta2

# 4. Erneut vorwärts rechnen
net1_new, hidden_new, output_new, loss_new = forward()

print("Nach dem Training:", output_new)
print("CCE nachher:", loss_new)

# 5. Referenztest
assert np.allclose(
    output_new,
    [0.08, 0.92],
    atol=0.005
)

print("Referenztest bestanden!")
