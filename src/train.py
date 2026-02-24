"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
import os


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
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimizer to use')
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[128, 64], help='Sizes of hidden layers')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh', 'softmax'], default='relu', help='Activation function to use')
    parser.add_argument('--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help='Loss function to use')
    parser.add_argument('--weight_init', type=str, choices=['random', 'xavier'], default='random', help='Weight initialization method')
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
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    print("Loading data...")

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    print("Initializing model...")
    model = NeuralNetwork(args)

    print("Starting training...")
    model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    print("Evaluating model...")
    accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")

    print("Saving model...")
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    

    weights = []
    biases = []

    for layer in model.layers:
        weights.append(layer.W)
        biases.append(layer.b)

    np.save(args.model_save_path, {
        "weights": weights,
        "biases": biases
    })
    print("Training complete!")
    print(f"Model saved to {args.model_save_path}")


if __name__ == '__main__':
    main()
