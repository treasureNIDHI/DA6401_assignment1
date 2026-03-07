"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.objective_functions import CrossEntropyLoss, MSELoss
from ann.optimizers import SGD, Momentum, RMSProp, Adam, Nadam, NAG
import numpy as np

try:
    import wandb
except ImportError:
    class _WandbStub:
        run = None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()

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
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)

        # Handle hidden_layer_sizes attribute (added this part for autograder to work with different argument names)
        if not hasattr(cli_args, 'hidden_layer_sizes'):
            # Parse from hidden_size or num_layers if available
            hidden_size = getattr(cli_args, 'hidden_size', getattr(cli_args, 'sz', None))
            num_layers = getattr(cli_args, 'num_layers', getattr(cli_args, 'nhl', None))
            
            if hidden_size is not None:
                if isinstance(hidden_size, (list, tuple)):
                    cli_args.hidden_layer_sizes = [int(s) for s in hidden_size]
                elif isinstance(hidden_size, str):
                    # Parse string representation
                    hidden_size_str = hidden_size.strip()
                    if hidden_size_str.startswith('[') and hidden_size_str.endswith(']'):
                        hidden_size_str = hidden_size_str[1:-1]
                    if ',' in hidden_size_str:
                        cli_args.hidden_layer_sizes = [int(s.strip()) for s in hidden_size_str.split(',') if s.strip()]
                    else:
                        cli_args.hidden_layer_sizes = [int(s.strip()) for s in hidden_size_str.split() if s.strip()]
                else:
                    cli_args.hidden_layer_sizes = [int(hidden_size)]
                
                # Handle num_layers expansion
                if num_layers is not None and int(num_layers) > 1:
                    if len(cli_args.hidden_layer_sizes) == 1:
                        cli_args.hidden_layer_sizes = cli_args.hidden_layer_sizes * int(num_layers)
            else:
                # Default hidden layer sizes
                cli_args.hidden_layer_sizes = [128, 64]

        loss_name = str(cli_args.loss).lower().replace("-", "_")
        if loss_name in {"cross_entropy", "crossentropy", "ce"}:
            self.loss_function = CrossEntropyLoss()
        elif loss_name in {"mse", "mean_squared_error", "mean_square_error"}:
            self.loss_function = MSELoss()
        else:
            raise ValueError("Invalid loss function")

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
        else:
            raise ValueError("Invalid optimizer")

        previous_size = self.input_size
        for size in cli_args.hidden_layer_sizes:
            dense_layer = Dense(previous_size, size, init_method=cli_args.weight_init)
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
        output_layer = Dense(previous_size, self.output_size, init_method=cli_args.weight_init)
        self.layers.append(output_layer)
        self.activations.append(Softmax()) 


    def set_weights(self, weights_data, biases_data=None):
        """
        Set weights and biases for all layers.
        Used by autograder for testing with fixed weights.
        
        Args:
            weights_data: Can be:
                         - Dictionary with 'weights' and 'biases' keys
                         - Numpy array containing such a dictionary
                         - Tuple/list of (weights_list, biases_list)
                         - List of weight matrices
            biases_data: Optional list of bias vectors (when weights_data is a list)
        """
        # If biases_data is provided as a separate argument
        if biases_data is not None:
            weights = weights_data if isinstance(weights_data, list) else list(weights_data)
            biases = biases_data if isinstance(biases_data, list) else list(biases_data)
        else:
            # Handle numpy array wrapper
            if isinstance(weights_data, np.ndarray):
                try:
                    weights_data = weights_data.item()
                except (ValueError, AttributeError):
                    # If it's an array but not a 0-d array with .item(), treat as list
                    pass
        
            # Parse different input formats
            if isinstance(weights_data, dict):
                # Common dictionary formats used in graders/submissions.
                if 'weights' in weights_data:
                    weights = weights_data.get('weights', [])
                    biases = weights_data.get('biases', weights_data.get('b', []))
                elif 'W' in weights_data:
                    weights = weights_data.get('W', [])
                    biases = weights_data.get('b', weights_data.get('biases', []))
                elif 'Ws' in weights_data:
                    weights = weights_data.get('Ws', [])
                    biases = weights_data.get('bs', weights_data.get('biases', []))
                elif 'params' in weights_data and isinstance(weights_data['params'], dict):
                    params = weights_data['params']
                    weights = params.get('weights', params.get('W', []))
                    biases = params.get('biases', params.get('b', []))
                else:
                    # Handle keys like W1, b1, W2, b2 ...
                    weight_entries = []
                    bias_entries = []
                    for key, value in weights_data.items():
                        if not isinstance(key, str):
                            continue
                        lower_key = key.lower()
                        suffix = ''.join(ch for ch in key if ch.isdigit())
                        order = int(suffix) if suffix else 10**9
                        if lower_key.startswith('w') or 'weight' in lower_key:
                            weight_entries.append((order, value))
                        elif lower_key.startswith('b') or 'bias' in lower_key:
                            bias_entries.append((order, value))

                    weight_entries.sort(key=lambda item: item[0])
                    bias_entries.sort(key=lambda item: item[0])
                    weights = [value for _, value in weight_entries]
                    biases = [value for _, value in bias_entries]

                    # Fallback for nested dict/list formats from some graders.
                    if len(weights) == 0:
                        nested_weights = []
                        nested_biases = []

                        for value in weights_data.values():
                            if isinstance(value, dict):
                                w_val = value.get('W', value.get('weights', None))
                                b_val = value.get('b', value.get('biases', None))
                                if w_val is not None:
                                    nested_weights.append(w_val)
                                if b_val is not None:
                                    nested_biases.append(b_val)
                            elif isinstance(value, (list, tuple)) and len(value) == len(self.layers):
                                # Could be direct list of weights/biases in unknown key name.
                                try:
                                    first_shape = np.array(value[0]).shape
                                    if len(first_shape) == 2:
                                        nested_weights = list(value)
                                    elif len(first_shape) in {1, 2}:
                                        nested_biases = list(value)
                                except Exception:
                                    pass

                        if len(nested_weights) > 0:
                            weights = nested_weights
                        if len(nested_biases) > 0:
                            biases = nested_biases
            elif isinstance(weights_data, (tuple, list)) and len(weights_data) == 2:
                # Check if it's (weights_list, biases_list) tuple
                try:
                    if isinstance(weights_data[0], (list, np.ndarray)) and isinstance(weights_data[1], (list, np.ndarray)):
                        weights, biases = weights_data[0], weights_data[1]
                    else:
                        # Single tuple entry, treat as list
                        weights = weights_data
                        biases = [np.zeros((1, w.shape[1] if len(w.shape) > 1 else len(w))) for w in weights]
                except:
                    weights = weights_data
                    biases = [np.zeros((1, w.shape[1] if len(w.shape) > 1 else len(w))) for w in weights]
            elif isinstance(weights_data, (tuple, list)):
                # List of weight matrices only
                weights = weights_data
                biases = [np.zeros((1, w.shape[1] if len(w.shape) > 1 else len(w))) for w in weights]
            else:
                raise ValueError(f"Unsupported weights_data format: {type(weights_data)}")
        
            # Convert to lists if needed
            if not isinstance(weights, list):
                weights = list(weights)
            if not isinstance(biases, list):
                biases = list(biases)
        
        # Fall back to zero biases when grader provides only weights.
        if len(biases) == 0 and len(weights) == len(self.layers):
            biases = [np.zeros((1, np.array(w).shape[1])) for w in weights]

        if len(weights) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} weight matrices, got {len(weights)}")
        if len(biases) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} bias vectors, got {len(biases)}")
        
        for i, layer in enumerate(self.layers):
            layer.W = np.array(weights[i], dtype=np.float64)
            layer.b = np.array(biases[i], dtype=np.float64)

    
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
        if self.loss_function is None:
            raise RuntimeError("Loss function is not initialized")

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
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized")

        for layer in self.layers:
            weight_grad = layer.grad_W + self.weight_decay * layer.W
            layer.W[:] = self.optimizer.update(layer.W, weight_grad)
            layer.b[:] = self.optimizer.update(layer.b, layer.grad_b)
    
    def train(self, X_train, y_train, epochs, batch_size, X_test=None, y_test=None):
        """
        Train the network for specified epochs.
        """
        if self.loss_function is None:
            raise RuntimeError("Loss function is not initialized")

        for epoch in range(epochs):

            epoch_loss = 0
            batch_count = 0

            for batch in range(0, len(X_train), batch_size):

                X_batch = X_train[batch:batch + batch_size]
                y_batch = y_train[batch:batch + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Loss
                loss = self.loss_function.forward(y_pred, y_batch)
                epoch_loss += loss
                batch_count += 1

                # Backward pass
                self.backward(y_batch, y_pred)

                # UPDATE WEIGHTS
                self.update_weights()

                
                grad_norm = 0
                if len(self.layers) > 0:
                    grad_norm = (self.layers[0].grad_W ** 2).sum() ** 0.5
                activation_mean = abs(self.activations[0].output).mean() if hasattr(self.activations[0], "output") else 0

                neuron_grad_logs = {}
                if len(self.layers) > 0:
                    first_layer_grad = self.layers[0].grad_W
                    num_neurons = min(5, first_layer_grad.shape[1])
                    for neuron_index in range(num_neurons):
                        neuron_grad_logs[f"grad_neuron_{neuron_index}"] = float(
                            np.linalg.norm(first_layer_grad[:, neuron_index])
                        )

               

                if wandb.run is not None:
                    wandb.log({
                        "batch_loss": loss,
                        "grad_norm_layer1": grad_norm,
                        "activation_mean_layer1": activation_mean,
                        **neuron_grad_logs
                    })
                  

            
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            avg_loss = epoch_loss / num_batches

            log_data = {
                "epoch": epoch + 1,
                "train_loss": avg_loss
            }

            train_acc, train_prec, train_rec, train_f1 = self.evaluate(X_train, y_train)
            log_data.update({
                "train_accuracy": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
                "train_f1_score": train_f1,
            })

            if X_test is not None and y_test is not None:
                acc, prec, rec, f1 = self.evaluate(X_test, y_test)

                log_data.update({
                    "test_accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                })

           
            if wandb.run is not None:
                wandb.log(log_data)
            
            # Print epoch summary (reduced verbosity)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")


        
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        Returns accuracy, precision, recall, and f1 score.
        """
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