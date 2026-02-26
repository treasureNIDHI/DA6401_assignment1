"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class Dense:
    def __init__(self, input_features, output_features, init_method="random" ):
        if init_method == "random":
            self.W = np.random.randn(input_features, output_features) * 0.01
        elif init_method == "xavier":
            limit = np.sqrt(6 / (input_features + output_features))
            self.W = np.random.uniform(-limit, limit, (input_features, output_features))
        elif init_method == "zeros":
            self.W = np.zeros((input_features, output_features))
        else:
            raise ValueError("Invalid initialization method")
        
        self.b = np.zeros((1, output_features))
        
        # Gradients (required by autograder)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        # Cache for backward pass
        self.cache_x = None

    def forward(self, x):
        self.cache_x = x
        return np.dot(x, self.W) + self.b

        
    
    def backward(self, dout):
        if self.cache_x is None:
            raise RuntimeError("forward must be called before backward")
        self.grad_W = np.dot(self.cache_x.T, dout)
        self.grad_b = np.sum(dout, axis=0, keepdims=True)
        grad_input = np.dot(dout, self.W.T)
        return grad_input
        
        
    
