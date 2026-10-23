
import numpy as np


class MiniBatchTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def predict(self, X):
        return self.model.forward(X)

    def loss(self, X, T):
        output = self.predict(X)
        output = np.clip(output, 1e-12, 1.0)

        return -np.mean(
            np.sum(T * np.log(output), axis=1)
        )

    def accuracy(self, X, T):
        output = self.predict(X)

        predictions = np.argmax(output, axis=1)
        targets = np.argmax(T, axis=1)

        return np.mean(predictions == targets)

    def train(
        self,
        X_train,
        T_train,
        X_val=None,
        T_val=None,
        epochs=20,
        batch_size=64,
        patience=5,
        seed=42
    ):
        rng = np.random.default_rng(seed)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        best_val_loss = float("inf")
        best_weights = None
        wait = 0

        n = len(X_train)

        for epoch in range(epochs):

            # 1. Daten mischen (Shuffling)
            indices = rng.permutation(n)

            # 2. Mini-Batches bilden
            for start in range(0, n, batch_size):
                batch_indices = indices[
                    start:start + batch_size
                ]

                X_batch = X_train[batch_indices]
                T_batch = T_train[batch_indices]

                # 3. Forward
                self.model.forward(X_batch)

                # 4. Backpropagation
                gradients = self.model.backward(T_batch)

                # 5. Gewichte aktualisieren
                self.optimizer.step(
                    self.model.layers,
                    gradients
                )

            # 6. Training bewerten
            train_loss = self.loss(X_train, T_train)
            train_acc = self.accuracy(X_train, T_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            message = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%}"
            )

            # 7. Validation und Early Stopping
            if X_val is not None and T_val is not None:
                val_loss = self.loss(X_val, T_val)
                val_acc = self.accuracy(X_val, T_val)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                message += (
                    f" | Val Loss: {val_loss:.4f}"
                    f" | Val Acc: {val_acc:.2%}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0

                    best_weights = [
                        (layer.W.copy(), layer.theta.copy())
                        for layer in self.model.layers
                    ]
                else:
                    wait += 1

            print(message)

            if (
                X_val is not None
                and T_val is not None
                and wait >= patience
            ):
                print("Early Stopping!")
                break

        # Beste Gewichte wiederherstellen
        if best_weights is not None:
            for layer, (W, theta) in zip(
                self.model.layers, best_weights
            ):
                layer.W[:] = W
                layer.theta[:] = theta

        return history
