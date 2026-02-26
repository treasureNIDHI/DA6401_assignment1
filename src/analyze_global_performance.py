import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    csv_path = "wandb_export_2026-02-26T22_26_44.673+05_30.csv"
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                rows.append(
                    {
                        "name": row["Name"],
                        "optimizer": row["optimizer"],
                        "activation": row["activation"],
                        "learning_rate": float(row["learning_rate"]),
                        "batch_size": int(row["batch_size"]),
                        "hidden_layer_sizes": row["hidden_layer_sizes"],
                        "weight_init": row["weight_init"],
                        "test_accuracy": float(row["test_accuracy"]),
                        "train_loss": float(row["train_loss"]),
                        "train_accuracy": float(row["train_accuracy"]) if row.get("train_accuracy") not in (None, "", "nan", "NaN") else None,
                    }
                )
            except (ValueError, TypeError, KeyError):
                continue

    if not rows:
        raise RuntimeError("No valid rows found in sweep CSV")

    test_acc = np.array([row["test_accuracy"] for row in rows], dtype=float)
    train_loss = np.array([row["train_loss"] for row in rows], dtype=float)
    train_acc_values = np.array(
        [row["train_accuracy"] for row in rows if row["train_accuracy"] is not None],
        dtype=float,
    )

    has_strict_train_accuracy = len(train_acc_values) == len(rows) and len(rows) > 0

    if has_strict_train_accuracy:
        train_metric = np.array([row["train_accuracy"] for row in rows], dtype=float)
        train_metric_label = "Train accuracy"
    else:
        train_metric = 1.0 - (train_loss / np.max(train_loss))
        train_metric_label = "Train quality proxy (1-normalized train loss)"

    sorted_indices = np.argsort(test_acc)
    x = np.arange(len(rows))

    plt.figure(figsize=(10, 5.5))
    plt.plot(x, train_metric[sorted_indices], label=train_metric_label, linewidth=1.4)
    plt.plot(x, test_acc[sorted_indices], label="Test accuracy", linewidth=1.4)
    plt.xlabel("Runs sorted by test accuracy")
    plt.ylabel("Score")
    plt.title("Global Performance Across Sweep Runs")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/global_performance_overlay.png", dpi=220)
    plt.close()

    candidates = []
    not_top_test_threshold = np.quantile(test_acc, 0.75)

    if has_strict_train_accuracy:
        high_train_accuracy_threshold = np.quantile(train_metric, 0.75)
        for row, train_value in zip(rows, train_metric):
            if train_value >= high_train_accuracy_threshold and row["test_accuracy"] <= not_top_test_threshold:
                candidates.append(
                    {
                        "name": row["name"],
                        "optimizer": row["optimizer"],
                        "activation": row["activation"],
                        "learning_rate": row["learning_rate"],
                        "batch_size": row["batch_size"],
                        "hidden_layer_sizes": row["hidden_layer_sizes"],
                        "weight_init": row["weight_init"],
                        "test_accuracy": row["test_accuracy"],
                        "train_accuracy": float(train_value),
                        "generalization_gap": float(train_value - row["test_accuracy"]),
                    }
                )
    else:
        low_loss_threshold = np.quantile(train_loss, 0.35)
        for row, proxy in zip(rows, train_metric):
            if row["train_loss"] <= low_loss_threshold and row["test_accuracy"] <= not_top_test_threshold:
                candidates.append(
                    {
                        "name": row["name"],
                        "optimizer": row["optimizer"],
                        "activation": row["activation"],
                        "learning_rate": row["learning_rate"],
                        "batch_size": row["batch_size"],
                        "hidden_layer_sizes": row["hidden_layer_sizes"],
                        "weight_init": row["weight_init"],
                        "test_accuracy": row["test_accuracy"],
                        "train_loss": row["train_loss"],
                        "train_proxy": float(proxy),
                    }
                )

    candidates = sorted(candidates, key=lambda item: (item["train_loss"], item["test_accuracy"]))[:10]

    with open("images/global_overfit_candidates.json", "w", encoding="utf-8") as file:
        json.dump(candidates, file, indent=2)

    print(f"rows={len(rows)}")
    print(f"strict_train_accuracy_available={has_strict_train_accuracy}")
    print("saved images/global_performance_overlay.png")
    print("saved images/global_overfit_candidates.json")
    print("top candidates:")
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
