import numpy as np


import numpy as np

# XOR-Eingaben: x1 und x2
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Richtige Klassen als One-Hot-Vektoren
T = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
], dtype=float)

print("Eingaben:")
print(X)

print("Targets:")
print(T)

print("Form von X:", X.shape)
print("Form von T:", T.shape)

# Hidden Layer: 2 Eingänge, 2 Neuronen

W1 = np.array([
    [0.2, 0.4],
    [0.3, 0.1]
])

theta1 = np.array([0.1, 0.2])

# Wir wählen die dritte XOR-Instanz
x = X[2]

# Forward Propagation mit Schleifen
hidden = np.zeros(2)

for j in range(2):
    net = 0.0

    for i in range(2):
        net += x[i] * W1[i, j]

    net -= theta1[j]

    hidden[j] = max(0, net)

print("Input:", x)
print("Hidden Output:", hidden)



# Gewichte und Schwellwerte der Output Layer
W2 = np.array([
    [0.3, 0.7],
    [0.5, 0.2]
])

theta2 = np.array([0.1, 0.2])

# Netzeingaben der Output Layer
net_output = np.zeros(2)

for j in range(2):
    net = 0.0

    for i in range(2):
        net += hidden[i] * W2[i, j]

    net -= theta2[j]
    net_output[j] = net

# Softmax (numerisch stabil)
shifted = net_output - np.max(net_output)

exp_values = np.exp(shifted)

output = np.zeros(2)
sum_exp = 0.0

for j in range(2):
    sum_exp += exp_values[j]

for j in range(2):
    output[j] = exp_values[j] / sum_exp

print("Net Output:", net_output)
print("Softmax Output:", output)
print("Vorhergesagte Klasse:", np.argmax(output))
print("Richtige Klasse:", np.argmax(T[2]))
target = T[2]

loss = 0.0

for j in range(2):
    loss -= target[j] * np.log(output[j])

print("CCE-Fehler:", loss)


# Fehlersignal der Output Layer
delta_output = np.zeros(2)

for j in range(2):
    delta_output[j] = output[j] - target[j]

print("Delta Output:", delta_output)

# Gradienten der Output Layer
grad_W2 = np.zeros((2, 2))
grad_theta2 = np.zeros(2)

for j in range(2):
    for i in range(2):
        grad_W2[i, j] = hidden[i] * delta_output[j]

    grad_theta2[j] = -delta_output[j]

print("Gradient W2:")
print(grad_W2)
print("Gradient theta2:", grad_theta2)



# Fehlersignal der Hidden Layer
delta_hidden = np.zeros(2)

for i in range(2):
    error_sum = 0.0

    for j in range(2):
        error_sum += W2[i, j] * delta_output[j]

    if hidden[i] > 0:
        delta_hidden[i] = error_sum
    else:
        delta_hidden[i] = 0.0

# Gradienten der Hidden Layer
grad_W1 = np.zeros((2, 2))
grad_theta1 = np.zeros(2)

for j in range(2):
    for i in range(2):
        grad_W1[i, j] = x[i] * delta_hidden[j]

    grad_theta1[j] = -delta_hidden[j]

print("Delta Hidden:", delta_hidden)
print("Gradient W1:")
print(grad_W1)
print("Gradient theta1:", grad_theta1)



# Gradientenabstieg
eta = 0.5

for i in range(2):
    for j in range(2):
        W1[i, j] -= eta * grad_W1[i, j]
        W2[i, j] -= eta * grad_W2[i, j]

for j in range(2):
    theta1[j] -= eta * grad_theta1[j]
    theta2[j] -= eta * grad_theta2[j]

print("Neue Gewichte W1:")
print(W1)

print("Neue Gewichte W2:")
print(W2)

print("Neue Schwellwerte theta1:", theta1)
print("Neue Schwellwerte theta2:", theta2)

# Forward Propagation nach dem Update

# 1. Hidden Layer neu berechnen
hidden_new = np.zeros(2)

for j in range(2):
    net = 0.0

    for i in range(2):
        net += x[i] * W1[i, j]

    net -= theta1[j]
    hidden_new[j] = max(0, net)

# 2. Output Layer neu berechnen
net_output_new = np.zeros(2)

for j in range(2):
    net = 0.0

    for i in range(2):
        net += hidden_new[i] * W2[i, j]

    net -= theta2[j]
    net_output_new[j] = net

# 3. Softmax neu berechnen
shifted_new = net_output_new - np.max(net_output_new)
exp_new = np.exp(shifted_new)

sum_exp_new = 0.0

for j in range(2):
    sum_exp_new += exp_new[j]

output_new = np.zeros(2)

for j in range(2):
    output_new[j] = exp_new[j] / sum_exp_new

# 4. CCE neu berechnen
loss_new = 0.0

for j in range(2):
    loss_new -= target[j] * np.log(
        max(output_new[j], 1e-15)
    )

# 5. Ergebnisse vergleichen
print("----- Vor dem Update -----")
print("Output:", output)
print("CCE:", loss)

print("----- Nach dem Update -----")
print("Hidden:", hidden_new)
print("Output:", output_new)
print("CCE:", loss_new)

print("Fehler kleiner?", loss_new < loss)
