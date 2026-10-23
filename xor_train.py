
import numpy as np

from mlp.network import MLP
##from mlp.optimizers import Adam
from mlp.trainer import Trainer
from mlp.optimizers import GradientDescent

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

T = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
], dtype=float)

# 2 Eingaben, 4 versteckte Neuronen, 2 Ausgaben
model = MLP([2, 4, 2])

# Adam als Optimizer
#optimizer = Adam(learning_rate=0.01)
optimizer = GradientDescent(learning_rate=0.1)
trainer = Trainer(model, optimizer)

# Trainieren
history = trainer.train(
    X,
    T,
    epochs=5000,
    verbose_every=100
)

# Alle XOR-Eingaben testen
predictions = trainer.predict(X)

print("\nXOR-Ergebnisse:")

for x, output, target in zip(X, predictions, T):
    print(
        f"Eingabe: {x.astype(int)} | "
        f"Vorhersage: {np.argmax(output)} | "
        f"Richtig: {np.argmax(target)} | "
        f"Softmax: {np.round(output, 4)}"
    )

print("\nEndgültige Accuracy:", trainer.accuracy(X, T))
