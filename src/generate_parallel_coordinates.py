import csv
import os

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode_category(values):
    unique_values = sorted(set(values))
    index = {value: idx for idx, value in enumerate(unique_values)}
    encoded = np.array([index[value] for value in values], dtype=float)
    return encoded, unique_values


def main() -> None:
    csv_path = "wandb_export_2026-02-26T22_26_44.673+05_30.csv"
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                rows.append(
                    {
                        "optimizer": row["optimizer"],
                        "activation": row["activation"],
                        "learning_rate": _to_float(row["learning_rate"]),
                        "batch_size": _to_float(row["batch_size"]),
                        "weight_init": row["weight_init"],
                        "test_accuracy": _to_float(row["test_accuracy"]),
                    }
                )
            except KeyError:
                continue

    if not rows:
        raise RuntimeError("No valid rows found in sweep CSV")

    optimizers = [row["optimizer"] for row in rows]
    activations = [row["activation"] for row in rows]
    inits = [row["weight_init"] for row in rows]

    optimizer_encoded, optimizer_labels = _encode_category(optimizers)
    activation_encoded, activation_labels = _encode_category(activations)
    init_encoded, init_labels = _encode_category(inits)

    learning_rates = np.array([row["learning_rate"] for row in rows], dtype=float)
    batch_sizes = np.array([row["batch_size"] for row in rows], dtype=float)
    accuracies = np.array([row["test_accuracy"] for row in rows], dtype=float)

    def normalize(values):
        min_value = np.min(values)
        max_value = np.max(values)
        if max_value - min_value < 1e-12:
            return np.zeros_like(values)
        return (values - min_value) / (max_value - min_value)

    matrix = np.vstack(
        [
            normalize(optimizer_encoded),
            normalize(activation_encoded),
            normalize(learning_rates),
            normalize(batch_sizes),
            normalize(init_encoded),
            normalize(accuracies),
        ]
    ).T

    x = np.arange(matrix.shape[1])
    labels = ["optimizer", "activation", "lr", "batch", "weight_init", "test_acc"]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    color_norm = Normalize(vmin=np.min(accuracies), vmax=np.max(accuracies))
    cmap = plt.get_cmap("viridis")

    for row, acc in zip(matrix, accuracies):
        ax.plot(x, row, color=cmap(color_norm(acc)), alpha=0.22, linewidth=0.9)

    cbar = fig.colorbar(ScalarMappable(norm=color_norm, cmap=cmap), ax=ax)
    cbar.set_label("Test Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Parallel Coordinates View of Sweep Hyperparameters")
    ax.set_ylabel("Normalized Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    os.makedirs("images", exist_ok=True)
    fig.savefig("images/parallel_coordinates.png", dpi=220)
    plt.close(fig)

    print(f"rows={len(rows)}")
    print(f"optimizer_labels={optimizer_labels}")
    print(f"activation_labels={activation_labels}")
    print(f"weight_init_labels={init_labels}")
    print("saved images/parallel_coordinates.png")


if __name__ == "__main__":
    main()
