"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
import os
import wandb
import json
from sklearn.model_selection import train_test_split


def _parse_hidden_layer_sizes(values):
    if isinstance(values, list):
        raw_value = " ".join(str(value) for value in values).strip()
    else:
        raw_value = str(values).strip()

    if raw_value.startswith("[") and raw_value.endswith("]"):
        raw_value = raw_value[1:-1].strip()

    if not raw_value:
        raise argparse.ArgumentTypeError("hidden_layer_sizes cannot be empty")

    if "," in raw_value:
        tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    else:
        tokens = [token.strip() for token in raw_value.split() if token.strip()]

    try:
        return [int(token) for token in tokens]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid hidden_layer_sizes value: {values}"
        ) from error


def _positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed_value


def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    config_args, remaining_argv = config_parser.parse_known_args()

    config_defaults = {}
    if config_args.config is not None:
        with open(config_args.config, 'r') as config_file:
            loaded_config = json.load(config_file)
        if not isinstance(loaded_config, dict):
            raise ValueError("Config file must contain a JSON object")
        config_defaults = loaded_config

    parser = argparse.ArgumentParser(description='Train a neural network', parents=[config_parser])
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion', 'fashion_mnist'], default='mnist', help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=_positive_int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=_positive_int, default=64, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimizer to use')
    parser.add_argument('--hidden_layer_sizes', type=str, nargs='+', default=['128', '64'], help='Sizes of hidden layers (backward compatible alias)')
    parser.add_argument('-sz', '--hidden_size', type=str, nargs='+', default=None, help='Sizes of hidden layers (list of values)')
    parser.add_argument('-nhl', '--num_layers', type=_positive_int, default=None, help='Number of hidden layers')
    parser.add_argument('-a', '--activation', type=str, choices=['relu', 'sigmoid', 'tanh'], default='relu', help='Activation function to use')
    parser.add_argument('-l', '--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('-wi', '--weight_init', type=str, choices=['random', 'xavier', 'zeros'], default='random', help='Weight initialization method')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='L2 weight decay coefficient')
    parser.add_argument('--wandb_project', type=str, default='nn_training', help='W&B project name')
    parser.add_argument('--model_save_path', type=str, default='models/best_model.npy', help='Path to save trained model')


    parser.add_argument('--momentum_beta', type=float, default=0.9, help='Momentum beta for Momentum optimizer')
    parser.add_argument('--rmsprop_beta', type=float, default=0.9, help='Beta for RMSProp optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help ='Beta2 for Adam optimizer')
    parser.add_argument('--nadam_beta1', type=float, default=0.9, help='Beta1 for Nadam optimizer')
    parser.add_argument('--nadam_beta2', type=float, default=0.999, help='Beta2 for Nadam optimizer')
    parser.add_argument('--nag_beta', type=float, default=0.9, help='Beta for NAG optimizer')   

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training progress')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for optimizers to prevent division by zero')

    valid_keys = {action.dest for action in parser._actions}
    filtered_defaults = {
        key: value for key, value in config_defaults.items() if key in valid_keys
    }
    parser.set_defaults(**filtered_defaults)

    args = parser.parse_args(remaining_argv)

    hidden_sizes_source = args.hidden_size if args.hidden_size is not None else args.hidden_layer_sizes
    args.hidden_layer_sizes = _parse_hidden_layer_sizes(hidden_sizes_source)

    if args.num_layers is not None:
        if len(args.hidden_layer_sizes) == 1 and args.num_layers > 1:
            args.hidden_layer_sizes = args.hidden_layer_sizes * args.num_layers
        elif len(args.hidden_layer_sizes) != args.num_layers:
            raise ValueError(
                "num_layers must match number of hidden_size values, or provide a single hidden_size to repeat"
            )

    if args.dataset == 'fashion':
        args.dataset = 'fashion_mnist'

    return args



def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Initialize wandb correctly
    wandb.init(
        project=args.wandb_project,
        name=f"{args.optimizer}_{args.activation}_{args.learning_rate}",
        config=vars(args)
    )

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=args.seed,
        stratify=np.argmax(y_train, axis=1)
    )

    print("Initializing model...")
    model = NeuralNetwork(args)

    print("Starting training...")
    model.train(
        X_train_split,
        y_train_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_test=X_val,
        y_test=y_val
    )

    print("Evaluating model...")
    accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")

    # Log metrics to wandb
    wandb.log({
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("Saving model...")
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    weights = []
    biases = []

    for layer in model.layers:
        weights.append(layer.W)
        biases.append(layer.b)

    np.save(
        args.model_save_path,
        np.array(
            {
                "weights": weights,
                "biases": biases
            },
            dtype=object
        )
    )

    # Save config file
    with open("best_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Training complete!")
    print(f"Model saved to {args.model_save_path}")

    wandb.finish()

if __name__ == '__main__':
    main()