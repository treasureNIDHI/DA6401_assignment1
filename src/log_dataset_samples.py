import wandb
import numpy as np
from utils.data_loader import load_data

wandb.init(project="da6401-assignment-1", name="dataset_visualization")

X_train, y_train, _, _ = load_data("mnist")

class_names = [str(i) for i in range(10)]

table = wandb.Table(columns=["image", "label"])

count_per_class = {i: 0 for i in range(10)}

for img, label in zip(X_train, y_train):

    if isinstance(label, np.ndarray):
        label_idx = int(np.argmax(label))
    else:
        label_idx = int(label)

    if count_per_class[label_idx] < 5:

        image = img.reshape(28, 28)

        table.add_data(
            wandb.Image(image),
            class_names[label_idx]
        )

        count_per_class[label_idx] += 1

    if all(v == 5 for v in count_per_class.values()):
        break

wandb.log({"mnist_samples": table})
wandb.finish()