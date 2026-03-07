#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, 'src')

from ann.neural_network import NeuralNetwork

# Load MNIST data directly from npz
def load_mnist_npz():
    data = np.load('data/mnist.npz')
    X_train = data['x_train'].reshape(data['x_train'].shape[0], -1).astype(np.float32) / 255.0
    X_test = data['x_test'].reshape(data['x_test'].shape[0], -1).astype(np.float32) / 255.0
    y_train = data['y_train']
    y_test = data['y_test']
    
    # One-hot encode
    y_train_onehot = np.zeros((y_train.size, 10))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    y_test_onehot = np.zeros((y_test.size, 10))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    return X_train, y_train_onehot, X_test, y_test_onehot, y_test

# Load MNIST data
X_train, y_train, X_test, y_test_onehot, y_test_raw = load_mnist_npz()
print(f'Test data shape: {X_test.shape}, {y_test_onehot.shape}')

# Create a mock args object for 3-layer network
class Args:
    hidden_size = [128, 64]
    num_layers = 2
    hidden_layer_sizes = [128, 64]
    loss = 'cross_entropy'
    optimizer = 'nadam'
    learning_rate = 0.001
    momentum_beta = 0.9
    rmsprop_beta = 0.999
    epsilon = 1e-8
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    nadam_beta1 = 0.9
    nadam_beta2 = 0.999
    nag_beta = 0.9
    weight_decay = 0
    activation = 'relu'
    weight_init = 'xavier'

args = Args()
model = NeuralNetwork(args)

# Test different 3-layer models
models_to_test = [
    ('models/smoke_model.npy', 'smoke_model'),
    ('models/global_strict_1.npy', 'global_strict_1'),
    ('models/global_strict_8.npy', 'global_strict_8'),
]

for model_path, model_name in models_to_test:
    try:
        data = np.load(model_path, allow_pickle=True).item()
        weights = data['weights']
        biases = data['biases']
        
        model.set_weights(weights, biases)
        
        # Evaluate
        y_pred = model.predict_proba(X_test)
        y_pred_labels = y_pred.argmax(axis=1)
        
        # Compute accuracy as a proxy for F1
        accuracy = np.mean(y_pred_labels == y_test_raw)
        print(f'{model_name}: Accuracy = {accuracy:.4f}')
    except Exception as e:
        print(f'{model_name}: Error - {e}')

