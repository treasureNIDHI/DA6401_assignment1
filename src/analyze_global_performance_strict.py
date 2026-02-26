import json
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb


def main() -> None:
    entity = "nidhi-jagatpura-iit-madras"
    project = "assignment_1-global_strict"

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    rows = []
    for run in runs:
        summary = run.summary
        config = run.config

        train_accuracy = summary.get("train_accuracy")
        test_accuracy = summary.get("test_accuracy")

        if train_accuracy is None or test_accuracy is None:
            continue

        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "optimizer": config.get("optimizer"),
                "activation": config.get("activation"),
                "learning_rate": config.get("learning_rate"),
                "batch_size": config.get("batch_size"),
                "hidden_layer_sizes": config.get("hidden_layer_sizes"),
                "weight_init": config.get("weight_init"),
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "train_loss": float(summary.get("train_loss", np.nan)),
            }
        )

    if not rows:
        raise RuntimeError("No runs with both train_accuracy and test_accuracy found")

    train_acc = np.array([row["train_accuracy"] for row in rows], dtype=float)
    test_acc = np.array([row["test_accuracy"] for row in rows], dtype=float)
    gaps = train_acc - test_acc

    sorted_indices = np.argsort(test_acc)
    x = np.arange(len(rows))

    plt.figure(figsize=(10, 5.5))
    plt.plot(x, train_acc[sorted_indices], marker="o", linewidth=1.6, label="Train accuracy")
    plt.plot(x, test_acc[sorted_indices], marker="o", linewidth=1.6, label="Test accuracy")
    plt.xlabel("Runs sorted by test accuracy")
    plt.ylabel("Accuracy")
    plt.title("Global Performance (Strict): Train vs Test Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/global_performance_overlay_strict.png", dpi=220)
    plt.close()

    candidates = []
    high_train_threshold = np.quantile(train_acc, 0.6)
    high_gap_threshold = np.quantile(gaps, 0.6)

    for row in rows:
        gap = row["train_accuracy"] - row["test_accuracy"]
        if row["train_accuracy"] >= high_train_threshold and gap >= high_gap_threshold:
            candidate = dict(row)
            candidate["generalization_gap"] = float(gap)
            candidates.append(candidate)

    candidates = sorted(candidates, key=lambda item: item["generalization_gap"], reverse=True)

    with open("images/global_overfit_candidates_strict.json", "w", encoding="utf-8") as file:
        json.dump(candidates, file, indent=2)

    print(f"runs_with_strict_metrics={len(rows)}")
    print(f"mean_train_accuracy={float(np.mean(train_acc)):.4f}")
    print(f"mean_test_accuracy={float(np.mean(test_acc)):.4f}")
    print(f"mean_generalization_gap={float(np.mean(gaps)):.4f}")
    print("saved images/global_performance_overlay_strict.png")
    print("saved images/global_overfit_candidates_strict.json")
    print("top_gap_runs:")
    for row in sorted(rows, key=lambda item: item["train_accuracy"] - item["test_accuracy"], reverse=True)[:5]:
        gap = row["train_accuracy"] - row["test_accuracy"]
        print(f"- {row['run_name']} ({row['run_id']}): train={row['train_accuracy']:.4f}, test={row['test_accuracy']:.4f}, gap={gap:.4f}")


if __name__ == "__main__":
    main()
