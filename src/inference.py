"""
Inference Script
Evaluate trained models on test sets
"""
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from ann.objective_functions import CrossEntropyLoss, MSELoss
import argparse


def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('--model_path', type=str, default='models/best_model.npy', help='Path to saved model weights')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--hidden_layer_sizes', type=int, nargs='+', default=[128, 64], help='Sizes of hidden layers')
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid', 'tanh', 'softmax'], default='relu', help='Activation function to use')    
    parser.add_argument('--loss', type=str, choices=['cross_entropy', 'mse'], default='cross_entropy', help='Loss function to use') 
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer (not used in inference but required for model initialization')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimizer to use (not used in inference but required for model initialization)')


    parser.add_argument('--momentum_beta', type=float, default=0.9, help='Momentum beta for Momentum optimizer (not used in inference but required for model initialization)')
    parser.add_argument('--rmsprop_beta', type=float, default=0.9, help='Beta for RMSProp optimizer (not used in inference but required for model initialization)')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (not used in inference but required for model initialization)')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help ='Beta2 for Adam optimizer (not used in inference but required for model initialization)')  
    parser.add_argument('--nadam_beta1', type=float, default=0.9, help='Beta1 for Nadam optimizer (not used in inference but required for model initialization)')
    parser.add_argument('--nadam_beta2', type=float, default=0.999, help='Beta2 for Nadam optimizer (not used in inference but required for model initialization)')
    parser.add_argument('--nag_beta', type=float, default=0.9, help='Beta for NAG optimizer (not used in inference but required for model initialization)') 
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for optimizers to prevent division by zero (not used in inference but required for model initialization)') 

    return parser.parse_args()


def load_model(model_path, args):
    """
    Load trained model from disk.
    """
    model = NeuralNetwork(args)
    saved  = np.load(model_path, allow_pickle=True).item()

    weights = saved['weights']
    biases = saved['biases']

    for i, layer in enumerate(model.layers):
        layer.W = weights[i]
        layer.b = biases[i]

    return model


def evaluate_model(model, X_test, y_test, args): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    if args.loss == "cross_entropy":
        loss_function = CrossEntropyLoss()
    else:
        loss_function = MSELoss()
    loss = loss_function.forward(logits, y_test)
    accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    print("Loading model...")
    model = load_model(args.model_path, args)
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test, args)
    print("\nResults:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")

    return results
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
