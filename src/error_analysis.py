import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
import json

# -----------------------------
# Load best config
# -----------------------------
with open("best_config.json", "r") as f:
    config_dict = json.load(f)

class Args:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

args = Args(config_dict)

# -----------------------------
# Initialize W&B
# -----------------------------
wandb.init(
    project=args.wandb_project,
    name="confusion_matrix_analysis",
    tags=["error_analysis"]
)

# -----------------------------
# Load test data
# -----------------------------
_, _, X_test, y_test = load_data(args.dataset)

# Convert one-hot to labels if needed
if len(y_test.shape) > 1:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test

# -----------------------------
# Rebuild model and load weights
# -----------------------------
model = NeuralNetwork(args)

saved = np.load(args.model_save_path, allow_pickle=True).item()
weights = saved["weights"]
biases = saved["biases"]

for layer, W, b in zip(model.layers, weights, biases):
    layer.W = W
    layer.b = b

# -----------------------------
# Get predictions
# -----------------------------
y_pred_probs = model.forward(X_test)

if len(y_pred_probs.shape) > 1:
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    y_pred = y_pred_probs

# -----------------------------
# Log Confusion Matrix
# -----------------------------
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred,
        class_names=[str(i) for i in range(10)]
    )
})

# -----------------------------
# Log Misclassified Examples (Creative Part)
# -----------------------------
table = wandb.Table(columns=["image", "true_label", "predicted_label"])

for img, true, pred in zip(X_test, y_true, y_pred):
    if true != pred:
        table.add_data(
            wandb.Image(img.reshape(28, 28)),
            int(true),
            int(pred)
        )

wandb.log({"misclassified_examples": table})

wandb.finish()