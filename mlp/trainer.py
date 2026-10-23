
import numpy as np


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def predict(self, X):
        """Vorhersagen für alle Eingaben."""
        predictions = []

        for x in X:
            output = self.model.forward(x)
            predictions.append(output.copy())

        return np.array(predictions)

    def accuracy(self, X, T):
        """Anteil richtig klassifizierter Instanzen."""
        predictions = self.predict(X)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(T, axis=1)

        return np.mean(predicted_classes == true_classes)

    def loss(self, X, T):
        """Mittlere Categorical Cross-Entropy."""
        total_loss = 0.0

        for x, target in zip(X, T):
            output = self.model.forward(x)

            # Schutz vor log(0)
            output = np.clip(output, 1e-15, 1.0)

            total_loss += -np.sum(
                target * np.log(output)
            )

        return total_loss / len(X)

    def train(
        self,
        X,
        T,
        epochs=1000,
        shuffle=True,
        verbose_every=100
    ):
        """Training mit einzelnen Instanzen."""

        if len(X) == 0 or len(X) != len(T):
            raise ValueError("X und T müssen gleich viele Instanzen haben.")

        rng = np.random.default_rng(42)

        history = {
            "loss": [],
            "accuracy": []
        }

        for epoch in range(1, epochs + 1):

            # Reihenfolge der Trainingsdaten
            if shuffle:
                indices = rng.permutation(len(X))
            else:
                indices = np.arange(len(X))

            # Training: Forward -> Backward -> Update
            for index in indices:
                x = X[index]
                target = T[index]

                self.model.forward(x)

                gradients = self.model.backward(target)

                self.optimizer.step(
                    self.model.layers,
                    gradients
                )

            # Ergebnisse nach jeder Epoche
            epoch_loss = self.loss(X, T)
            epoch_accuracy = self.accuracy(X, T)

            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_accuracy)

            if verbose_every and (
                epoch == 1
                or epoch % verbose_every == 0
                or epoch == epochs
            ):
                print(
                    f"Epoche {epoch:4d}: "
                    f"Loss = {epoch_loss:.6f}, "
                    f"Accuracy = {epoch_accuracy:.2%}"
                )

            # Bei XOR: alle Instanzen richtig klassifiziert
            if epoch_accuracy == 1.0:
                print(f"Alle Instanzen richtig nach {epoch} Epochen!")
                break

        return history
