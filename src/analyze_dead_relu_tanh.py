import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def build_args(activation: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="mnist",
        epochs=10,
        batch_size=64,
        learning_rate=0.1,
        optimizer="adam",
        hidden_layer_sizes=[128, 128, 128],
        activation=activation,
        loss="cross_entropy",
        weight_init="xavier",
        weight_decay=0.0,
        wandb_project="assignment_1-dead_neuron",
        model_save_path="",
        momentum_beta=0.9,
        rmsprop_beta=0.9,
        adam_beta1=0.9,
        adam_beta2=0.999,
        nadam_beta1=0.9,
        nadam_beta2=0.999,
        nag_beta=0.9,
        seed=42,
        log_interval=10,
        epsilon=1e-8,
    )


def load_model(model_path: str, activation: str) -> NeuralNetwork:
    args = build_args(activation)
    model = NeuralNetwork(args)
    payload = np.load(model_path, allow_pickle=True).item()

    for index, layer in enumerate(model.layers):
        layer.W = payload["weights"][index]
        layer.b = payload["biases"][index]

    return model


def get_hidden_activations(model: NeuralNetwork, inputs: np.ndarray) -> list[np.ndarray]:
    activations = []
    output = inputs
    hidden_layer_count = len(model.layers) - 1

    for idx in range(hidden_layer_count):
        output = model.layers[idx].forward(output)
        output = model.activations[idx].forward(output)
        activations.append(output.copy())

    return activations


def summarize_dead_relu(hidden_acts: list[np.ndarray]) -> list[dict]:
    summary = []
    for layer_idx, acts in enumerate(hidden_acts, start=1):
        dead_mask = np.all(acts == 0.0, axis=0)
        dead_pct = float(np.mean(dead_mask) * 100.0)
        zero_frac = float(np.mean(acts == 0.0) * 100.0)
        summary.append(
            {
                "layer": layer_idx,
                "dead_neuron_pct": dead_pct,
                "zero_activation_pct": zero_frac,
            }
        )
    return summary


def summarize_tanh_saturation(hidden_acts: list[np.ndarray]) -> list[dict]:
    summary = []
    for layer_idx, acts in enumerate(hidden_acts, start=1):
        sat_pos = np.mean(acts > 0.99, axis=0)
        sat_neg = np.mean(acts < -0.99, axis=0)
        sat_any = sat_pos + sat_neg
        saturated_neurons = sat_any > 0.95
        saturated_pct = float(np.mean(saturated_neurons) * 100.0)
        near_zero_pct = float(np.mean(np.abs(acts) < 1e-3) * 100.0)
        summary.append(
            {
                "layer": layer_idx,
                "saturated_neuron_pct": saturated_pct,
                "near_zero_activation_pct": near_zero_pct,
            }
        )
    return summary


def plot_activation_histograms(relu_acts: np.ndarray, tanh_acts: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.hist(relu_acts.ravel(), bins=80, alpha=0.65, label="ReLU (LR=0.1)", density=True)
    plt.hist(tanh_acts.ravel(), bins=80, alpha=0.65, label="Tanh (LR=0.1)", density=True)
    plt.xlim(-1.2, 3.0)
    plt.xlabel("Activation Value (Layer 1)")
    plt.ylabel("Density")
    plt.title("Layer-1 Activation Distribution: ReLU vs Tanh")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_dead_vs_saturated(relu_summary: list[dict], tanh_summary: list[dict], out_path: str) -> None:
    layers = [item["layer"] for item in relu_summary]
    relu_dead = [item["dead_neuron_pct"] for item in relu_summary]
    tanh_sat = [item["saturated_neuron_pct"] for item in tanh_summary]

    x = np.arange(len(layers))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, relu_dead, width=width, label="ReLU dead neurons (%)")
    plt.bar(x + width / 2, tanh_sat, width=width, label="Tanh saturated neurons (%)")
    plt.xticks(x, [f"Layer {layer}" for layer in layers])
    plt.ylabel("Percentage")
    plt.title("Dead ReLU Neurons vs Saturated Tanh Neurons")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    relu_model = load_model("models/dead_relu.npy", "relu")
    tanh_model = load_model("models/dead_tanh.npy", "tanh")

    _, _, x_test, _ = load_data("mnist")

    relu_hidden = get_hidden_activations(relu_model, x_test)
    tanh_hidden = get_hidden_activations(tanh_model, x_test)

    relu_summary = summarize_dead_relu(relu_hidden)
    tanh_summary = summarize_tanh_saturation(tanh_hidden)

    os.makedirs("images", exist_ok=True)
    plot_activation_histograms(relu_hidden[0], tanh_hidden[0], "images/dead_activation_distribution.png")
    plot_dead_vs_saturated(relu_summary, tanh_summary, "images/dead_vs_saturated_by_layer.png")

    payload = {
        "relu_dead": relu_summary,
        "tanh_saturated": tanh_summary,
    }

    with open("images/dead_neuron_summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print("Saved images/dead_activation_distribution.png")
    print("Saved images/dead_vs_saturated_by_layer.png")
    print("Saved images/dead_neuron_summary.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
