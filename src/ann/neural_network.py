"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.objective_functions import CrossEntropyLoss, MSELoss
from ann.optimizers import SGD, Momentum, RMSProp, Adam, Nadam, NAG

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        self.activations = []
        self.loss_function = None
        self.optimizer = None
        self.input_size = 784
        self.output_size = 10


        if cli_args.loss == "cross_entropy":
            self.loss_function = CrossEntropyLoss()

        elif cli_args.loss == "mse":
            self.loss_function = MSELoss()

        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(cli_args.learning_rate)

        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(cli_args.learning_rate, cli_args.momentum_beta)

        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(cli_args.learning_rate, cli_args.rmsprop_beta, cli_args.epsilon)

        elif cli_args.optimizer == "adam":
            self.optimizer = Adam(cli_args.learning_rate, cli_args.adam_beta1, cli_args.adam_beta2, cli_args.epsilon)

        elif cli_args.optimizer == "nadam":
            self.optimizer = Nadam(cli_args.learning_rate, cli_args.nadam_beta1, cli_args.nadam_beta2, cli_args.epsilon)

        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(cli_args.learning_rate, cli_args.nag_beta)

        previous_size = self.input_size
        for size in cli_args.hidden_layer_sizes:
            dense_layer = Dense(previous_size, size)
            self.layers.append(dense_layer)
            if cli_args.activation == "relu":
                activation_object = ReLU()
            elif cli_args.activation == "sigmoid":
                activation_object = Sigmoid()
            elif cli_args.activation == "tanh":
                activation_object = Tanh()
            elif cli_args.activation == "softmax":
                activation_object = Softmax()
            else:
                raise ValueError("Invalid activation function")
            self.activations.append(activation_object)  # Use the selected activation for hidden layers
            previous_size = size
        output_layer = Dense(previous_size, self.output_size)
        self.layers.append(output_layer)
        self.activations.append(Softmax()) 



    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        output = X

        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            output = self.activations[i].forward(output)
        return output

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        self.loss_function.y_pred = y_pred
        self.loss_function.y_true = y_true
        grad = self.loss_function.backward()

        for i in reversed(range(len(self.layers))):
            grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            layer.W[:] = self.optimizer.update(layer.W, layer.grad_W)
            layer.b[:] = self.optimizer.update(layer.b, layer.grad_b)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """

        for epoch in range(epochs):

            for batch in range(0, len(X_train), batch_size):

                X_batch = X_train[batch:batch + batch_size]
                y_batch = y_train[batch:batch + batch_size]

                y_pred = self.forward(X_batch)
                loss = self.loss_function.forward(y_pred, y_batch)
                print(f"Epoch {epoch + 1}, Batch {batch}, Loss: {loss:.4f}")
                self.backward(y_batch, y_pred)
                self.update_weights()

        
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        Returns accuracy, precision, recall, and f1 score.
        """
        import numpy as np
        
        y_pred = self.forward(X)
        # Convert predictions to class labels
        y_pred_labels = y_pred.argmax(axis=1)
        # Convert one-hot encoded labels to class indices if needed
        if y.ndim == 2:
            y_true_labels = y.argmax(axis=1)
        else:
            y_true_labels = y
        
        # Calculate accuracy
        correct = (y_pred_labels == y_true_labels).sum()
        accuracy = correct / len(y)
        
        # Calculate precision, recall, and f1 for multi-class (macro-averaged)
        num_classes = y_pred.shape[1]
        precisions = []
        recalls = []
        f1_scores = []
        
        for cls in range(num_classes):
            true_positive = ((y_pred_labels == cls) & (y_true_labels == cls)).sum()
            false_positive = ((y_pred_labels == cls) & (y_true_labels != cls)).sum()
            false_negative = ((y_pred_labels != cls) & (y_true_labels == cls)).sum()
            
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Macro-average
        precision_avg = np.mean(precisions)
        recall_avg = np.mean(recalls)
        f1_avg = np.mean(f1_scores)
        
        return accuracy, precision_avg, recall_avg, f1_avg