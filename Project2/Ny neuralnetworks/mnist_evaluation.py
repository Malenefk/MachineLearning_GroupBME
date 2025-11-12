"""Utility script for training and evaluating the FFNN on the scikit-learn MNIST dataset."""

from __future__ import annotations

import matplotlib

# Use a non-interactive backend so the script can run in headless environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import autograd.numpy as np

from neuralnetwork import FFNN, Adam, MCE_multiclass, Sigmoid, Softmax


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Return a one-hot encoded matrix for the provided label vector."""

    y_int = y.astype(int)
    return np.eye(n_classes)[y_int]


def load_mnist_subset(sample_size: int | None = 20000, random_state: int = 42):
    """Fetch the MNIST dataset from scikit-learn and optionally select a stratified subset."""

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    if sample_size is not None and sample_size < X.shape[0]:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )

    return X, y


def train_and_evaluate(
    sample_size: int | None = 20000,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 15,
    batches: int = 50,
):
    """Train the FFNN on MNIST and report training/test accuracy with a confusion matrix."""

    X, y = load_mnist_subset(sample_size=sample_size, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    n_classes = len(np.unique(y))

    y_train_encoded = one_hot_encode(y_train, n_classes)
    y_test_encoded = one_hot_encode(y_test, n_classes)

    model = FFNN(
        nodes=(X_train.shape[1], 64, n_classes),
        hidden_activation=Sigmoid(),
        output_activation=Softmax(),
        cost_func=MCE_multiclass(),
        seed=random_state,
    )

    model.fit(
        X_train,
        y_train_encoded,
        scheduler=Adam(eta=0.001, rho=0.9, rho2=0.999),
        batches=batches,
        epochs=epochs,
        lam=0.0001,
    )

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("FFNN MNIST confusion matrix (Sigmoid in hidden layers)")
    plt.tight_layout()
    plt.savefig("mnist_confusion_matrix.png", dpi=300)
    plt.close()

    return train_accuracy, test_accuracy, cm


if __name__ == "__main__":
    train_and_evaluate()
