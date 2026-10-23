
import numpy as np

from mlp.network import MLP
from mlp.optimizers import Adam
from mlp.trainer import Trainer


# Vier XOR-Instanzen
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


def run_experiments(hidden_neurons, runs=20):
    successes = 0
    successful_epochs = []

    print(f"\n=== {hidden_neurons} Hidden Neuronen ===")

    for run in range(runs):

        # Neue MLP für jeden Trainingslauf
        model = MLP([2, hidden_neurons, 2])

        # Reproduzierbare Startgewichte
        rng = np.random.default_rng(1000 + run)

        for layer in model.layers:
            layer.W = rng.uniform(
                -0.7, 0.7, size=layer.W.shape
            )
            layer.theta = np.zeros_like(layer.theta)

        # Adam für jeden Lauf neu erstellen
        optimizer = Adam(learning_rate=0.01)
        trainer = Trainer(model, optimizer)

        history = trainer.train(
            X,
            T,
            epochs=2000,
            verbose_every=0
        )

        final_accuracy = trainer.accuracy(X, T)
        epochs_used = len(history["loss"])

        success = final_accuracy == 1.0

        if success:
            successes += 1
            successful_epochs.append(epochs_used)

        print(
            f"Lauf {run + 1:02d}: "
            f"{'Erfolg' if success else 'Fehler'} | "
            f"Epochen: {epochs_used} | "
            f"Accuracy: {final_accuracy:.0%}"
        )

    print("\n--- Zusammenfassung ---")
    print("Hidden Neuronen:", hidden_neurons)
    print("Erfolgreiche Läufe:", successes, "/", runs)
    print("Erfolgsquote:", f"{successes / runs:.0%}")

    if successful_epochs:
        print(
            "Mittlere Epochen bei Erfolg:",
            np.mean(successful_epochs)
        )

    return successes


# 20 Trainingsläufe pro Architektur
success_2 = run_experiments(2)
success_4 = run_experiments(4)

print("\n=== VERGLEICH ===")
print("2 Hidden Neuronen:", success_2, "/ 20")
print("4 Hidden Neuronen:", success_4, "/ 20")
