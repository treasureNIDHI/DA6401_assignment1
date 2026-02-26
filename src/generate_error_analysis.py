import json
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


MNIST_CLASS_NAMES = [str(i) for i in range(10)]
FASHION_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_model_from_files(config_path: str, model_path: str) -> tuple[NeuralNetwork, str]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    args = SimpleNamespace(**config)

    if getattr(args, "dataset", "mnist") == "fashion":
        args.dataset = "fashion_mnist"

    if not hasattr(args, "weight_decay"):
        args.weight_decay = 0.0

    model = NeuralNetwork(args)
    payload = np.load(model_path, allow_pickle=True).item()

    for index, layer in enumerate(model.layers):
        layer.W = payload["weights"][index]
        layer.b = payload["biases"][index]

    return model, args.dataset


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (Best Model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_failure_gallery(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: str,
    max_samples: int = 25,
) -> None:
    wrong_indices = np.where(y_true != y_pred)[0]

    if wrong_indices.size == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications found.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    selected = wrong_indices[:max_samples]
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for idx, axis in enumerate(axes.flat):
        if idx < len(selected):
            sample_index = selected[idx]
            image = x_test[sample_index].reshape(28, 28)
            axis.imshow(image, cmap="gray")
            true_label = class_names[int(y_true[sample_index])]
            pred_label = class_names[int(y_pred[sample_index])]
            axis.set_title(f"T:{true_label} / P:{pred_label}", fontsize=8)
            axis.axis("off")
        else:
            axis.axis("off")

    fig.suptitle("Failure Gallery: Misclassified Test Samples", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "best_config.json"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/best_model.npy"

    model, dataset = build_model_from_files(config_path, model_path)
    _, _, x_test, y_test = load_data(dataset)

    probabilities = model.forward(x_test)
    y_pred = probabilities.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    class_names = FASHION_CLASS_NAMES if dataset == "fashion_mnist" else MNIST_CLASS_NAMES

    os.makedirs("images", exist_ok=True)
    save_confusion_matrix(y_true, y_pred, class_names, "images/confusion_matrix.png")
    save_failure_gallery(x_test, y_true, y_pred, class_names, "images/failure_gallery.png")

    accuracy = (y_true == y_pred).mean()
    print(f"Dataset: {dataset}")
    print(f"Accuracy from loaded model: {accuracy:.4f}")
    print("Saved: images/confusion_matrix.png")
    print("Saved: images/failure_gallery.png")


if __name__ == "__main__":
    main()
